'''
PointGroup train.py
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

import argparse
from copy import deepcopy

def init():
    # copy important files to backup
    backup_dir = os.path.join(CONF.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(CONF.model_dir, backup_dir))
    os.system('cp {} {}'.format(CONF.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(CONF.config, backup_dir))

    # log the config
    logger.info(CONF)

    # summary writer
    # global writer
    # writer = SummaryWriter(cfg.exp_path, flush_secs=1)

    # random seed
    random.seed(CONF.manual_seed)
    np.random.seed(CONF.manual_seed)
    torch.manual_seed(CONF.manual_seed)
    torch.cuda.manual_seed_all(CONF.manual_seed)


def train_epoch(train_loader, model, model_fn, optimizer, epoch, writer):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, CONF.lr, epoch - 1, CONF.step_epoch, CONF.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = CONF.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
            (epoch, CONF.epochs, i + 1, len(train_loader), am_dict['loss'].val, am_dict['loss'].avg,
             data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        if (i == len(train_loader) - 1): print()

    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, CONF.epochs, am_dict['loss'].avg,
                                                                     time.time() - start_epoch))

    utils.checkpoint_save(model, CONF.exp_path, CONF.config.split('/')[-1][:-5], epoch, CONF.save_freq, use_cuda)

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(f'train/{k}', am_dict[k].avg, epoch)
    writer.flush()


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

    if args.debug:
        scanrefer_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_val = [SCANREFER_TRAIN[0]]

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
    ##### init
    init()

    # ##### get model version and data version
    # exp_name = cfg.config.split('/')[-1][:-5]
    # model_name = exp_name.split('_')[0]
    # data_name = exp_name.split('_')[-1]

    # ##### model
    logger.info('=> creating model ...')

    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator

    # if model_name == 'pointgroup':
    #     from model.pointgroup.pointgroup import PointGroup as Network
    #     from model.pointgroup.pointgroup import model_fn_decorator
    # else:
    #     print("Error: no model - " + model_name)
    #     exit(0)

    model = Network(CONF)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if CONF.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.lr)
    elif CONF.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.lr, momentum=CONF.momentum,
                              weight_decay=CONF.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()

    writer = SummaryWriter(os.path.join(CONF.exp_path, 'logs'), flush_secs=1)

    # ##### dataset
    # if cfg.dataset == 'scannetv2':
    #     if data_name == 'scannet':
    #         import data.scannetv2_inst

    #         train_dataset = data.scannetv2_inst.Dataset(split='train')
    #         val_dataset = data.scannetv2_inst.Dataset(split='val')
    #         # dataset.trainLoader()
    #         # dataset.valLoader()
    #         train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
    #                                        collate_fn=lambda batch: collate_train(batch, cfg.scale, cfg.full_scale,
    #                                                                               voxel_mode=cfg.mode,
    #                                                                               max_npoint=cfg.max_npoint,
    #                                                                               batch_size=cfg.batch_size),
    #                                        num_workers=cfg.train_workers, shuffle=True, sampler=None, drop_last=True,
    #                                        pin_memory=True)

    #         val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
    #                                      collate_fn=lambda batch: collate_val(batch, cfg.scale, cfg.full_scale,
    #                                                                           voxel_mode=cfg.mode,
    #                                                                           max_npoint=cfg.max_npoint,
    #                                                                           batch_size=cfg.batch_size),
    #                                      num_workers=cfg.train_workers, shuffle=False, sampler=None, drop_last=True,
    #                                      pin_memory=True)


    #     else:
    #         print("Error: no data loader - " + data_name)
    #         exit(0)

    # ##### resume
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer(CONF)

    train_data_loader = get_dataloader(CONF, scanrefer_train, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION)
    val_data_loader = get_dataloader(CONF, scanrefer_eval_val, all_scene_list, "val", DC, True, SCAN2CAD_ROTATION)
    start_epoch = utils.checkpoint_restore(model, CONF.exp_path, CONF.config.split('/')[-1][:-5],
                                           use_cuda)  # resume from the latest epoch, or specify the epoch to restore

    # ##### train and val
    for epoch in range(start_epoch, CONF.epochs + 1):
        train_epoch(train_data_loader, model, model_fn, optimizer, epoch, writer)

        if utils.is_multiple(epoch, CONF.save_freq) or utils.is_power2(epoch):
            eval_epoch(val_data_loader, model, model_fn, epoch, writer)

        writer.close()


    # debug
    # import data.scannetv2_inst
    # train_dataset = data.scannetv2_inst.Dataset(split='train')
    # print(len(train_dataset))
    # train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
    #             collate_fn=lambda batch: collate_train(batch, cfg.scale, cfg.full_scale,
    #                                                     voxel_mode=cfg.mode,
    #                                                     max_npoint=cfg.max_npoint,
    #                                                     batch_size=cfg.batch_size),
    #             num_workers=cfg.train_workers, shuffle=True, sampler=None, drop_last=True,
    #             pin_memory=True)
    

    # ScanREFER

    

    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    # torch.manual_seed(CONF.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(CONF.seed)

    # from data.scanrefer import get_dataloader
    # print("preparing data...")
    # scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer(CONF)
    # print(all_scene_list)
    # train_data_loader = get_dataloader(CONF, scanrefer_train, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION)


    # Inspect Batch
    # batch = next(iter(train_data_loader))
    # print(batch['locs'].shape)
    
    
    
    