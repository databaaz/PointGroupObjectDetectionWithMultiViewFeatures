'''
PointGroup test.py
Written by Li Jiang
'''

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np
import json
from torch.utils.data import DataLoader

from util.config import cfg as CONF
from util.log import logger
import util.utils as utils
from data.scannetv2_inst import collate_train, collate_val
from data.scannet.model_util_scannet import ScannetDatasetConfig
from data.scanrefer import get_dataloader

from torch.utils.data import DataLoader

from util.config import cfg

cfg.task = 'test'
CONF.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval
from data.scannetv2_inst import collate_test
from util.map.ap_helper import APCalculator
import itertools
import json

import argparse
from copy import deepcopy


def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result',
                              'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH,
                                                                         cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
                              cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, test_dataloader, test_dataset, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    ######################################## RESET TO COLLATE TEST HERE

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}

        ap_calculator = APCalculator(ap_iou_thresh=0.5)

        # pred_all: map of {img_id: [(classname, bbox, score)]}
        # gt_all: map of {img_id: [(classname, bbox)]}

        for i, batch in enumerate(test_dataloader):  # itertools
            print(i)
            N = batch['feats'].shape[0]
            point_coords = batch['point_coords']#.cpu.numpy()
            # logger.info(f"point_coords shape:{point_coords.shape}")
            # test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]
            test_scene_name = test_dataset[0]['scene_id']
            print(test_scene_name)

            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1


            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

            pt_offsets = preds['pt_offsets']  # (N, 3), float32, cuda

            if (epoch > cfg.prepare_epochs):
                scores = preds['score']  # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds['proposals']
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                             device=scores_pred.device)  # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[
                    semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]]  # (nProposal), long

                ##### score threshold
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f,
                                            proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(),
                                                    cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]

                nclusters = clusters.shape[0]
                print("Clusters: ", nclusters)

                #bounding box 
                b_boxes = []
                b_boxes_map = []
                scores = cluster_scores.cpu().numpy()
                labels = cluster_semantic_id.cpu().numpy()
                clusters_np = clusters.cpu().numpy()


                for j,c in enumerate(clusters_np):
                    # logger.info(f"points in cluster from matrix {c.sum()}")
                    cluster_points = point_coords[c.astype(bool)]
                    # logger.info(f"points in cluster: {cluster_points.shape}")
                    center_x,center_y,center_z = cluster_points.mean(0)

                    # import pdb
                    # pdb.set_trace()

                    x_max,y_max,z_max = cluster_points.max(0)
                    x_min,y_min,z_min = cluster_points.min(0)
                    length = abs(x_max-x_min)
                    breadth = abs(y_max-y_min)
                    height = abs(z_max-z_min)
                    # logger.info(f"max_c:{(x_max,y_max,z_max)}\nmin_c:{(x_min,y_min,z_min)}\ncenter_c={center_x,center_y,center_z}")
                    bbox={"center_x":center_x,
                    "center_y":center_y,
                    "center_z":center_z,
                    "length":length,
                    "breadth":breadth,
                    "height":height,
                    "label":labels[j],
                    "score":scores[j]}

                    b_boxes.append(bbox)
                    x_000 = [center_x - length/2, center_y - breadth/2, center_z-height/2]
                    x_100 = [center_x + length/2, center_y - breadth/2, center_z-height/2]
                    x_110 = [center_x + length/2, center_y + breadth/2, center_z-height/2]
                    x_010 = [center_x - length/2, center_y + breadth/2, center_z-height/2]
                    x_001 = [center_x - length/2, center_y - breadth/2, center_z+height/2]
                    x_101 = [center_x + length/2, center_y - breadth/2, center_z+height/2]
                    x_111 = [center_x + length/2, center_y + breadth/2, center_z+height/2]
                    x_011 = [center_x - length/2, center_y + breadth/2, center_z+height/2]

                    b_boxes_map.append((bbox['label'], np.array([x_111, x_110, x_010, x_011, x_101, x_100, x_000, x_001]), bbox['score']))

                # ground truth boxes
                gt_file = os.path.join(cfg.data_root, 'scannetv2', cfg.split + '_gt', test_scene_name + '.txt')
                gt_instances = eval.get_gt_instances_from_file(gt_file)
                n_instances = [len(gt_instances[k]) for k in gt_instances]
                logger.info(f"no. of gt_instances = {sum(n_instances)}")
                gt_boxes = []
                gt_boxes_map = []
                for k in gt_instances:
                    instances = gt_instances[k]
                    if len(instances)==0:
                        continue
                    for inst in instances:
                        vertice_indices = inst['vertices']
                        # logger.info(f"n_vertices = {len(vertice_indices)}")
                        vertice_coords = point_coords[vertice_indices]
                        # logger.info(f"instance coords shape: {vertice_coords.shape}")
                        center_x,center_y,center_z = vertice_coords.mean(0)
                        x_max,y_max,z_max = vertice_coords.max(0)
                        x_min,y_min,z_min = vertice_coords.min(0)
                        length = abs(x_max-x_min)
                        breadth = abs(y_max-y_min)
                        height = abs(z_max-z_min)
                        bbox={"center_x":center_x,
                            "center_y":center_y,
                            "center_z":center_z,
                            "length":length,
                            "breadth":breadth,
                            "height":height,
                            "label":inst["label_id"]}

                        gt_boxes.append(bbox)
                        x_000 = [center_x - length/2, center_y - breadth/2, center_z-height/2]
                        x_100 = [center_x + length/2, center_y - breadth/2, center_z-height/2]
                        x_110 = [center_x + length/2, center_y + breadth/2, center_z-height/2]
                        x_010 = [center_x - length/2, center_y + breadth/2, center_z-height/2]
                        x_001 = [center_x - length/2, center_y - breadth/2, center_z+height/2]
                        x_101 = [center_x + length/2, center_y - breadth/2, center_z+height/2]
                        x_111 = [center_x + length/2, center_y + breadth/2, center_z+height/2]
                        x_011 = [center_x - length/2, center_y + breadth/2, center_z+height/2]
                        gt_boxes_map.append((bbox['label'], np.array([x_111, x_110, x_010, x_011, x_101, x_100, x_000, x_001])))

                ap_calculator.step(np.array([b_boxes_map]), np.array([gt_boxes_map]))

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, 'scannetv2', cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
            
            try:
                ##### save files
                start3 = time.time()
                if cfg.save_semantic:
                    os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                    semantic_np = semantic_pred.cpu().numpy()
                    np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

                if cfg.save_pt_offsets:
                    os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                    pt_offsets_np = pt_offsets.cpu().numpy()
                    coords_np = batch['locs_float'].numpy()
                    coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)  # (N, 6)
                    np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

                if(epoch > cfg.prepare_epochs and cfg.save_instance):
                    f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                    f1 = open(os.path.join(result_dir, test_scene_name +'_bbox'+ '.txt'), 'w')

                    for proposal_id in range(nclusters):
                        clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                        semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                        score = cluster_scores[proposal_id]
                        box = b_boxes[proposal_id]
                        f1.write(f"{box['center_x']},{box['center_y']},{box['center_z']},{box['length']},{box['breadth']},{box['height']},{box['label']}\n")
                        f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                        if proposal_id < nclusters - 1:
                            f.write('\n')
                        np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                    
                    with open(os.path.join(result_dir, test_scene_name +'_gtbbox'+ '.txt'), 'w') as f2:
                        for box in gt_boxes:
                            f2.write(f"{box['center_x']},{box['center_y']},{box['center_z']},{box['length']},{box['breadth']},{box['height']},{box['label']}\n")
                    
                    f.close()
                    f1.close()
            except:
                pass


            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            print(i)
            logger.info(
                "instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                    i + 1, len(test_dataset), N, nclusters, end, end1, end3))
        
        res = ap_calculator.compute_metrics()
        print(json.dumps(res, indent = 4))

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def eval_epoch(val_loader, model, model_fn, epoch, writer):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val,
                                                                         am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, CONF.epochs, am_dict['loss'].avg,
                                                                       time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(f'eval/{k}', am_dict[k].avg, epoch)
        writer.flush()

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()
def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    print("aargs", args)
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    # import pdb
    # pdb.set_trace()

    if args.debug:
        idx = [i for i, element in enumerate(SCANREFER_TRAIN) if element['scene_id'] == 'scene0296_00'][0]
        scanrefer_train = [SCANREFER_TRAIN[idx]]
        scanrefer_eval_train = [SCANREFER_TRAIN[idx]]
        scanrefer_eval_val = [SCANREFER_TRAIN[idx]]

    # import pdb
    # pdb.set_trace()

    if args.no_caption: #what does this arg imply ?
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0]) #why always taking first caption ?
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        # eval on train
        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)
        
        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0]) #WHY ?
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(args.dataset))
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train), len(new_scanrefer_eval_val)))

    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list


if __name__ == '__main__':
    # init()
    # ##### get model version and data version
    # exp_name = cfg.config.split('/')[-1][:-5]
    # model_name = exp_name.split('_')[0]
    # data_name = exp_name.split('_')[-1]

    # ##### model
    # logger.info('=> creating model ...')
    # logger.info('Classes: {}'.format(cfg.classes))

    # if model_name == 'pointgroup':
    #     from model.pointgroup.pointgroup import PointGroup as Network
    #     from model.pointgroup.pointgroup import model_fn_decorator
    # else:
    #     print("Error: no model version " + model_name)
    #     exit(0)
    # model = Network(cfg)

    # use_cuda = torch.cuda.is_available()
    # logger.info('cuda available: {}'.format(use_cuda))
    # assert use_cuda
    # model = model.cuda()

    # # logger.info(model)
    # logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    # ##### model_fn (criterion)
    # model_fn = model_fn_decorator(test=True)

    # ##### load model
    # utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False,
    #                          f=cfg.pretrain)  # resume from the latest epoch, or specify the epoch to restore

    # ##### evaluate
    # test(model, model_fn, data_name, cfg.test_epoch)



    ##### init
    init()

    ##### model
    logger.info('=> creating model ...')

    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator

    model = Network(CONF)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    writer = SummaryWriter(os.path.join(CONF.exp_path, 'logs'), flush_secs=1)

    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer(CONF)
    train_data_loader = get_dataloader(CONF, scanrefer_train, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION)
    val_data_loader = get_dataloader(CONF, scanrefer_eval_val, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION)

    ##### resume
    utils.checkpoint_restore(model, CONF.exp_path, CONF.config.split('/')[-1][:-5],
                            use_cuda)  # resume from the latest epoch, or specify the epoch to restore

    data_name = 'scannet'
    test(model, model_fn, data_name, train_data_loader, scanrefer_train, CONF.test_epoch)
    

