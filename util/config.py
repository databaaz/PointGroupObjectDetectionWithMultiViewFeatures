'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os
from easydict import EasyDict

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scanrefer.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    args_cfg = EasyDict(vars(args_cfg))
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    
    #return as dict of dict
    # args_cfg = EasyDict(config)
    args_cfg.update(config)
    for key in config:
        if type(config[key]) is dict:
            for k, v in config[key].items():
                args_cfg[k] = v

    
    # for key in config:
    #     print(key)
    #     for k, v in config[key].items():
    #         setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
# setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
cfg['exp_path'] = os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5])