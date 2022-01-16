'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops


class Dataset:
    def __init__(self, split, test_mode=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        # self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint

        if test_mode:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

        self.file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, split, '*' + self.filename_suffix)))

    def __getitem__(self, index):
        return torch.load(self.file_names[index])

    def __len__(self):
        return len(self.file_names)

    # Elastic distortion


def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32) // gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


def getInstanceInfo(xyz, instance_label):
    '''
    :param xyz: (n, 3)
    :param instance_label: (n), int, (0~nInst-1, -100)
    :return: instance_num, dict
    '''
    instance_info = np.ones((xyz.shape[0], 9),
                            dtype=np.float32) * -100.0  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
    instance_pointnum = []  # (nInst), int
    instance_num = int(instance_label.max()) + 1
    for i_ in range(instance_num):
        inst_idx_i = np.where(instance_label == i_)

        ### instance_info
        xyz_i = xyz[inst_idx_i]
        min_xyz_i = xyz_i.min(0)
        max_xyz_i = xyz_i.max(0)
        mean_xyz_i = xyz_i.mean(0)
        instance_info_i = instance_info[inst_idx_i]
        instance_info_i[:, 0:3] = mean_xyz_i
        instance_info_i[:, 3:6] = min_xyz_i
        instance_info_i[:, 6:9] = max_xyz_i
        instance_info[inst_idx_i] = instance_info_i

        ### instance_pointnum
        instance_pointnum.append(inst_idx_i[0].size)

    return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


def dataAugment(xyz, jitter=False, flip=False, rot=False):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if rot:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]])  # rotation
    return np.matmul(xyz, m)


def crop(xyz, full_scale, max_npoint):
    '''
    :param xyz: (n, 3) >= 0
    '''
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    assert valid_idxs.sum() == xyz.shape[0]

    full_scale = np.array([full_scale[1]] * 3)
    room_range = xyz.max(0) - xyz.min(0)
    while (valid_idxs.sum() > max_npoint):
        offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
        full_scale[:2] -= 32

    return xyz_offset, valid_idxs


def getCroppedInstLabel(instance_label, valid_idxs):
    instance_label = instance_label[valid_idxs]
    j = 0
    while (j < instance_label.max()):
        if (len(np.where(instance_label == j)[0]) == 0):
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label


def collate_train(batch, scale, full_scale, voxel_mode, max_npoint, batch_size):
    locs = []
    locs_float = []
    feats = []
    labels = []
    instance_labels = []

    instance_infos = []  # (N, 9)
    instance_pointnum = []  # (total_nInst), int

    batch_offsets = [0]

    total_inst_num = 0
    for i, item in enumerate(batch):
        xyz_origin, rgb, label, instance_label = item #fetch additional features here

        ### jitter / flip x / rotation
        xyz_middle = dataAugment(xyz_origin, True, True, True)
        # xyz_middle = xyz_origin

        ### scale
        xyz = xyz_middle * scale

        ### elastic
        xyz = elastic(xyz, 6 * scale // 50, 40 * scale / 50)
        xyz = elastic(xyz, 20 * scale // 50, 160 * scale / 50)

        ### offset
        xyz -= xyz.min(0)

        ### crop
        xyz, valid_idxs = crop(xyz, full_scale, max_npoint)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        label = label[valid_idxs]
        instance_label = getCroppedInstLabel(instance_label, valid_idxs)

        ### get instance information
        inst_num, inst_infos = getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

        instance_label[np.where(instance_label != -100)] += total_inst_num
        total_inst_num += inst_num

        ### merge the scene to the batch
        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_middle))
        feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1) # extend this one
        labels.append(torch.from_numpy(label))
        instance_labels.append(torch.from_numpy(instance_label))

        instance_infos.append(torch.from_numpy(inst_info))
        instance_pointnum.extend(inst_pointnum)

    ### merge all the scenes in the batchd
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

    locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    feats = torch.cat(feats, 0)  # float (N, C)
    labels = torch.cat(labels, 0).long()  # long (N)
    instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

    instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
    instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), full_scale[0], None)  # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, voxel_mode)

    return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
            'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
            'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape}


def collate_val(batch, scale, full_scale, voxel_mode, max_npoint, batch_size):
    locs = []
    locs_float = []
    feats = []
    labels = []
    instance_labels = []

    instance_infos = []  # (N, 9)
    instance_pointnum = []  # (total_nInst), int

    batch_offsets = [0]

    total_inst_num = 0
    for i, item in enumerate(batch):
        xyz_origin, rgb, label, instance_label = item

        ### flip x / rotation
        xyz_middle = dataAugment(xyz_origin, False, True, True)

        ### scale
        xyz = xyz_middle * scale

        ### offset
        xyz -= xyz.min(0)

        ### crop
        xyz, valid_idxs = crop(xyz, full_scale, max_npoint)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        label = label[valid_idxs]
        instance_label = getCroppedInstLabel(instance_label, valid_idxs)

        ### get instance information
        inst_num, inst_infos = getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

        instance_label[np.where(instance_label != -100)] += total_inst_num
        total_inst_num += inst_num

        ### merge the scene to the batch
        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_middle))
        feats.append(torch.from_numpy(rgb))
        labels.append(torch.from_numpy(label))
        instance_labels.append(torch.from_numpy(instance_label))

        instance_infos.append(torch.from_numpy(inst_info))
        instance_pointnum.extend(inst_pointnum)

    ### merge all the scenes in the batch
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

    locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    feats = torch.cat(feats, 0)  # float (N, C)
    labels = torch.cat(labels, 0).long()  # long (N)
    instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

    instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
    instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), full_scale[0], None)  # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, voxel_mode)

    return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
            'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
            'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'point_coords':xyz_origin,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape}


def collate_test(batch, scale, full_scale, voxel_mode, test_split, batch_size):
    locs = []
    locs_float = []
    feats = []

    batch_offsets = [0]

    for i, item in enumerate(batch):

        if test_split == 'val':
            xyz_origin, rgb, label, instance_label = item
        elif test_split == 'test':
            xyz_origin, rgb = item
        else:
            print("Wrong test split: {}!".format(test_split))
            exit(0)

        ### flip x / rotation
        xyz_middle = dataAugment(xyz_origin, False, True, True)

        ### scale
        xyz = xyz_middle * scale

        ### offset
        xyz -= xyz.min(0)

        ### merge the scene to the batch
        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_middle))
        feats.append(torch.from_numpy(rgb))

    ### merge all the scenes in the batch
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

    locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    feats = torch.cat(feats, 0)  # float (N, C)

    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), full_scale[0], None)  # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, voxel_mode)

    return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
            'locs_float': locs_float, 'feats': feats, 'point_coords':xyz_origin,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape}
