'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json, math
import pickle
import random
import numpy as np
import multiprocessing as mp

from itertools import chain
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
import scipy


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
# from lib.config import CONF
from util.config import cfg as CONF
from util.pc_utils import random_sampling, rotx, roty, rotz
from util.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from data.scannetv2_inst import dataAugment

import pointgroup_ops

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary.json")
VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json") # dataset_name
# SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary_weights.json")
VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "{}_vocabulary_weights.json") # dataset_name
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ReferenceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}

            # trim long descriptions
            tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2)) # start and end

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]
            
            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _build_vocabulary(self, dataset_name):
        vocab_path = VOCAB.format(dataset_name)
        if os.path.exists(vocab_path):
            self.vocabulary = json.load(open(vocab_path))
        else:
            if self.split == "train":
                all_words = chain(*[data["token"][:CONF.TRAIN.MAX_DES_LEN] for data in self.scanrefer])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1], reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(vocab_path, "w"), indent=4)

                self.vocabulary = vocab

    def _build_frequency(self, dataset_name):
        vocab_weights_path = VOCAB_WEIGHTS.format(dataset_name)
        if os.path.exists(vocab_weights_path):
            with open(vocab_weights_path) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            
            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights
            
            with open(vocab_weights_path, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self, dataset_name):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary(dataset_name)
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self._tranform_des()
        self._build_frequency(dataset_name)

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned #(N,9)
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy") 
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy") #(N,8)
            

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


class ScannetReferenceDataset(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        scan2cad_rotation=None):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        self._load_data(name)
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0 #255.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label = np.zeros(MAX_NUM_OBJ)
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))

        num_bbox = 1
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        
        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

        # construct the reference target label for each bbox
        for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
            if gt_id == object_id:
                ref_box_label[i] = 1
                ref_center_label = target_bboxes[i, 0:3]
                ref_heading_class_label = angle_classes[i]
                ref_heading_residual_label = angle_residuals[i]
                ref_size_class_label = size_classes[i]
                ref_size_residual_label = size_residuals[i]

                # construct ground truth box corner coordinates
                ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                ref_size_class_label, ref_size_residual_label)
                ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])
            
            # construct all GT bbox corners
            all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64), angle_residuals[:num_bbox],
                                    size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
            all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])
            
            # store
            gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
            gt_box_masks = np.zeros((MAX_NUM_OBJ,))
            gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

            gt_box_corner_label[:num_bbox] = all_box_corner_label
            gt_box_masks[:num_bbox] = 1
            gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,)) # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(instance_bboxes[:num_bbox,-1].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64) # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64) # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64) # valid bbox object ids
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start

        return data_dict

class ScannetReferenceTestDataset():
       
    def __init__(self, scanrefer_all_scene, 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False):

        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview

        # load data
        self.scene_data = self._load_data()
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.vocabulary = json.load(open(SCANREFER_VOCAB))
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer_all_scene)

    def __getitem__(self, idx):
        start = time.time()

        scene_id = self.scanrefer_all_scene[idx]

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["dataset_idx"] = idx
        data_dict["lang_feat"] = self.glove["sos"].astype(np.float32) # GloVE embedding for sos token
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _load_data(self):
        scene_data = {}
        for scene_id in self.scanrefer_all_scene:
            scene_data[scene_id] = {}
            scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned

        return scene_data

class ScannetObjectDataset(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        num_points=1024,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        is_caption=False,
        is_eval=False,
        whole_scene=False,
        use_pn_features=False
        ):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.is_caption = is_caption
        self.is_eval = is_eval
        self.whole_scene = whole_scene
        self.use_pn_features = use_pn_features

        # load data
        self._load_data()
        self.multiview_data = {}
        self.pn_feature_data = {}

        # filter data
        self.scanrefer = self.scanrefer if is_caption else self._filter_object(self.scanrefer)
        self.scanrefer = self._filter_scene(self.scanrefer) if is_eval and whole_scene else self.scanrefer

        # weights
        self.weights = np.ones((18))
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 

        # ------------------------------- DATA AUGMENTATION ------------------------------ 
        if self.augment and not self.use_pn_features:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

            # Translation
            point_cloud = self._translate(point_cloud)

        num_bbox = instance_bboxes.shape[0]
        target_masks = np.zeros((MAX_NUM_OBJ))
        scene_object_ids = np.zeros((MAX_NUM_OBJ))

        # find bbox parameters for target object
        unique_instance_ids = instance_bboxes[:, -1]
        unique_num_bbox = unique_instance_ids.shape[0]

        object_bbox_corners = np.zeros((MAX_NUM_OBJ, 8, 3))
        object_bbox_centers = np.zeros((MAX_NUM_OBJ, 3))

        if self.whole_scene:
            if self.use_pn_features: # load the pre-computed features
                pid = mp.current_process().pid
                if pid not in self.multiview_data:
                    database_path = os.path.join(CONF.PATH.PN_FEATURES, "{}.hdf5".format(self.split))
                    self.pn_feature_data[pid] = h5py.File(database_path, "r", libver="latest")

                # load PointNet++ features
                pn_features = self.pn_feature_data[pid][scene_id]
                assert pn_features.shape[0] == unique_num_bbox
                
                # load object bounding box information
                pn_corners = self.pn_feature_data[pid]["{}_box_corners".format(scene_id)]

                # pick out the features for the train split for a random epoch
                # the epoch pointer is always 0 in the eval mode for train split
                # this doesn't apply to val split
                if self.split == "train":
                    if self.is_eval:
                        epoch_id = 0
                    else:
                        epoch_id = random.choice(range(pn_features.shape[1]))
                    
                    pn_features = pn_features[:, epoch_id, :] # num_bboxes, 128
                    pn_corners = pn_corners[:, epoch_id, :, :] # num_bboxes, 8, 3

                # compute the object bounding box centers
                pn_centers = np.zeros((MAX_NUM_OBJ, 3))
                for i in range(unique_num_bbox):
                    target_bbox = pn_corners[i]
                    target_min = np.min(target_bbox, axis=0)
                    target_max = np.max(target_bbox, axis=0)
                    target_center = [(target_max[i] + target_min[i]) / 2 for i in range(3)]
                    pn_centers[i] = np.array(target_center)

                # indicate which feature is for the target object
                for i in range(unique_num_bbox):
                    if unique_instance_ids[i] == object_id:
                        target_idx = i

                # dump
                target_point_cloud = np.zeros((MAX_NUM_OBJ, 128))
                object_cat = np.zeros((MAX_NUM_OBJ))
                target_point_cloud[:unique_num_bbox] = pn_features
                object_cat[:unique_num_bbox] = instance_bboxes[:, -2]
                object_bbox_corners[:unique_num_bbox] = pn_corners
                object_bbox_centers = pn_centers

            else: # extract points in the object bounding boxes
                target_point_cloud = np.zeros((MAX_NUM_OBJ, self.num_points, point_cloud.shape[-1] + 1))
                object_cat = np.zeros((MAX_NUM_OBJ))
                for i in range(unique_num_bbox):
                    target_bbox, target_cat = self._get_object_bbox(point_cloud, instance_labels, semantic_labels, unique_instance_ids[i])
                    target_min = np.min(target_bbox, axis=0)
                    target_max = np.max(target_bbox, axis=0)
                    target_center = [(target_max[i] + target_min[i]) / 2 for i in range(3)]

                    object_bbox_corners[i] = target_bbox
                    object_bbox_centers[i] = np.array(target_center)

                    # object_point_cloud = self._get_object_pc(point_cloud, target_bbox)
                    object_point_cloud = self._get_object_pc(point_cloud, instance_labels, object_id)
                    target_point_cloud[i] = object_point_cloud
                    object_cat[i] = target_cat

                    if unique_instance_ids[i] == object_id:
                        target_idx = i

            target_masks[:unique_num_bbox] = 1
            scene_object_ids[:num_bbox] = unique_instance_ids
        else:
            if self.use_pn_features: # load the pre-computed features
                pid = mp.current_process().pid
                if pid not in self.multiview_data:
                    database_path = os.path.join(CONF.PATH.PN_FEATURES, "{}.hdf5".format(self.split))
                    self.pn_feature_data[pid] = h5py.File(database_path, "r", libver="latest")

                pn_features = self.pn_feature_data[pid][scene_id]
                assert pn_features.shape[0] == unique_num_bbox
                
                # pick out the features for the train split for a random epoch
                # the epoch pointer is always 0 in the eval mode for train split
                # this doesn't apply to val split
                if self.split == "train":
                    if self.is_eval:
                        epoch_id = 0
                    else:
                        epoch_id = random.choice(range(pn_features.shape[1]))
                    
                    pn_features = pn_features[:, epoch_id, :]

                # find the target object feature
                for i in range(unique_num_bbox):
                    if unique_instance_ids[i] == object_id:
                        target_idx = i

                # dump
                target_point_cloud = pn_features[target_idx]
            else: # extract points in the object bounding boxes
                # find bbox parameters for target object
                target_bbox, target_cat = self._get_object_bbox(point_cloud, instance_labels, semantic_labels, object_id)
                target_idx = 0 # placeholder   

                target_masks[:num_bbox] = 1   
                scene_object_ids[:num_bbox] = instance_bboxes[:num_bbox, -1]

                try:
                    # target_point_cloud = self._get_object_pc(point_cloud, target_bbox)
                    target_point_cloud = self._get_object_pc(point_cloud, instance_labels, object_id)
                except Exception:
                    with open("pc.obj", "w") as f:
                        for i in range(point_cloud.shape[0]):
                            f.write("v {} {} {} {} {} {}\n".format(
                                point_cloud[i, 0], 
                                point_cloud[i, 1], 
                                point_cloud[i, 2], 
                                point_cloud[i, 3], 
                                point_cloud[i, 4], 
                                point_cloud[i, 5]
                            ))

                    with open("bbox.obj", "w") as f:
                        for i in range(target_bbox.shape[0]):
                            f.write("v {} {} {} 255 0 0\n".format(
                                target_bbox[i, 0], 
                                target_bbox[i, 1], 
                                target_bbox[i, 2]
                            ))

                    exit()

            # object category
            object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = target_point_cloud.astype(np.float32) # point cloud data including features
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64) # object category 
        data_dict["target_idx"] = np.array(target_idx).astype(np.int64) # idx of the target object in the scene batch
        data_dict["target_masks"] = np.array(target_masks).astype(np.int64) # masks of valid objects in the scene batch
        data_dict["scene_object_ids"] = scene_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["object_bbox_corners"] = object_bbox_corners.astype(np.float32) # box corners of the bounding boxes
        data_dict["object_bbox_centers"] = object_bbox_centers.astype(np.float32) # box centers of the bounding boxes
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["load_time"] = time.time() - start

        return data_dict
    
    def _get_object_bbox(self, point_cloud, instance_labels, semantic_labels, object_id):
        # get object coordinates
        masks = instance_labels == object_id + 1
        coords = point_cloud[masks, 0:3]

        # get semantic label for the box
        counts = Counter(semantic_labels[masks], return_counts=True)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        bbox_sem = sorted_counts[0][0]

        # construct the bbox
        xmin = np.min(coords[:, 0])
        ymin = np.min(coords[:, 1])
        zmin = np.min(coords[:, 2])
        xmax = np.max(coords[:, 0])
        ymax = np.max(coords[:, 1])
        zmax = np.max(coords[:, 2])
        
        bbox = [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],

            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax]
        ]
        bbox = np.array(bbox)

        return bbox, bbox_sem

    # def _get_object_pc(self, point_cloud, target_bbox):
    #     # crop target object
    #     curmin = np.min(target_bbox, axis=0)
    #     curmax = np.max(target_bbox, axis=0)
    #     target_mask = np.sum((point_cloud[:, :3] >= (curmin - 0.05)) * (point_cloud[:, :3] <= (curmax + 0.05)), axis=1) == 3

    #     target_point_cloud, _ = random_sampling(point_cloud[target_mask], self.num_points, return_choices=True)

    #     return target_point_cloud

    def _get_object_pc(self, point_cloud, instance_labels, target_object_id):
        # random sampling
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]

        # create object masks
        target_object_masks = instance_labels == (target_object_id + 1) # 0: unannotated
        target_object_masks = target_object_masks.astype(np.float32)

        # concatenate to point cloud
        target_point_cloud = np.concatenate([point_cloud, target_object_masks[:, np.newaxis]], axis=1)

        return target_point_cloud

    def _filter_object(self, data):
        new_data = []
        cache = []
        for d in data:
            scene_id = d["scene_id"]
            object_id = d["object_id"]

            entry = "{}|{}".format(scene_id, object_id)

            if entry not in cache:
                cache.append(entry)
                new_data.append(d)

        return new_data

    def _filter_scene(self, data):
        new_data = []
        cache = []
        for d in data:
            scene_id = d["scene_id"]

            entry = "{}".format(scene_id)

            if entry not in cache:
                cache.append(entry)
                new_data.append(d)

        return new_data

    def _translate(self, point_set):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords

        return point_set


class ScannetForScan2CapPointGroupAllPoints(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        scan2cad_rotation=None):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        self._load_data(name)
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
        

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            # point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0 #255.0
            # replicate pointgroup behavior
            point_cloud[:,3:6] = point_cloud[:,3:6]/127.5 - 1
            pcl_color = point_cloud[:,3:6]

        point_cloud_mean = point_cloud[:, :3].mean(0)
        # normalize box centers
        # shift prediction, not gt bbox
        # instance_bboxes[:,:3] = instance_bboxes[:,:3] - point_cloud_mean
        # normalize points
        point_cloud[:,:3] = point_cloud[:,:3] - point_cloud_mean # replicate pointgroup behavior
        

        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]
        # pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label = np.zeros(MAX_NUM_OBJ)
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))

        num_bbox = 1
        point_votes = np.zeros([len(mesh_vertices), 3])
        point_votes_mask = np.zeros(len(mesh_vertices))
        
        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

        # construct the reference target label for each bbox
        for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
            if gt_id == object_id:
                ref_box_label[i] = 1
                ref_center_label = target_bboxes[i, 0:3]
                ref_heading_class_label = angle_classes[i]
                ref_heading_residual_label = angle_residuals[i]
                ref_size_class_label = size_classes[i]
                ref_size_residual_label = size_residuals[i]

                # construct ground truth box corner coordinates
                ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                ref_size_class_label, ref_size_residual_label)
                ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])
            
            # construct all GT bbox corners
            all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64), angle_residuals[:num_bbox],
                                    size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
            all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])
            
            # store
            gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
            gt_box_masks = np.zeros((MAX_NUM_OBJ,))
            gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

            gt_box_corner_label[:num_bbox] = all_box_corner_label
            gt_box_masks[:num_bbox] = 1
            gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,)) # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(instance_bboxes[:num_bbox,-1].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}
        #For PointGroup
        data_dict["locs"] = None#ToDo
        data_dict["voxel_locs"] = None#ToDo
        data_dict["p2v_map"] = None#ToDo
        data_dict["v2p_map"] = None#ToDo
        data_dict["locs_float"] = None#ToDo
        data_dict["feats"] = None#ToDo
        remapper = np.ones(150) * (-100)
        # for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        #     remapper[x] = i
        for i, x in enumerate(DC.nyu40id2class.keys()):
            remapper[x] = DC.nyu40id2class[x]
        remapper = np.array(remapper)
        data_dict["labels"] = remapper[semantic_labels]
        data_dict["instance_labels"] = instance_labels#ToDo
        data_dict["instance_info"] = None#ToDo
        data_dict["instance_pointnum"] = None#ToDo
        data_dict["offsets"] = None#ToDo
        data_dict["spatial_shape"] = None#ToDo
        bboxes = instance_bboxes.copy()
        bboxes[:,-2] = remapper[bboxes[:,-2].astype(int)]
        data_dict["instance_bboxes"] = bboxes

        

        # For Scan2Cap
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["point_clouds_mean"] = point_cloud_mean.astype(np.float32)
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64) # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64) # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64) # valid bbox object ids
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start
        return data_dict



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



def collate_train(batch, scale, full_scale, voxel_mode, max_npoint, batch_size, debug):
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
        data_dict = item
        pc = data_dict["point_clouds"]
        label = data_dict["labels"].astype(np.int32)
        instance_label = data_dict["instance_labels"].astype(np.int32)
        

        xyz_origin = pc[:,:3]

        if not debug:
            # ### jitter / flip x / rotation
            xyz_middle = dataAugment(xyz_origin, True, True, True)
        else:
            xyz_middle = xyz_origin

        features = pc[:,3:]
        ### scale
        xyz = xyz_middle * scale

        if not debug:
            ### elastic
            xyz = elastic(xyz, 6 * scale // 50, 40 * scale / 50)
            xyz = elastic(xyz, 20 * scale // 50, 160 * scale / 50)

        ### offset
        xyz -= xyz.min(0)

        ### crop
        xyz, valid_idxs = crop(xyz, full_scale, max_npoint)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        features = features[valid_idxs]
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
        # feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1) # extend this one
        feats.append(torch.from_numpy(features))
        
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
            'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, # 'point_coords': xyz_origin,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape}


def collate_val(batch, scale, full_scale, voxel_mode, max_npoint, batch_size, debug):
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
        data_dict = item
        pc = data_dict["point_clouds"]
        label = data_dict["labels"].astype(np.int32)
        instance_label = data_dict["instance_labels"].astype(np.int32)

        xyz_origin = pc[:,:3]

        if not debug:
            # ### jitter / flip x / rotation
            xyz_middle = dataAugment(xyz_origin, False, True, True)
        else:
            xyz_middle = xyz_origin

        features = pc[:,3:]
        ### scale
        xyz = xyz_middle * scale

        ### elastic
        # xyz = elastic(xyz, 6 * scale // 50, 40 * scale / 50)
        # xyz = elastic(xyz, 20 * scale // 50, 160 * scale / 50)

        ### offset
        xyz -= xyz.min(0)

        ### crop
        xyz, valid_idxs = crop(xyz, full_scale, max_npoint)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        features = features[valid_idxs]
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
        # feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1) # extend this one
        feats.append(torch.from_numpy(features))
        
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
            'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, # 'point_coords': xyz_origin,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape}



def collate_test(batch, scale, full_scale, voxel_mode, test_split, batch_size, debug):
    locs = []
    locs_float = []
    feats = []
    bboxes = [] #gt bbox
    batch_offsets = [0]

    for i, item in enumerate(batch):

        data_dict = item
        pc = data_dict["point_clouds"]
        label = data_dict["labels"].astype(np.int32)
        instance_label = data_dict["instance_labels"].astype(np.int32)

        xyz_origin = pc[:,:3]
        features = pc[:,3:]

        ### flip x / rotation
        if not debug:
            xyz_middle = dataAugment(xyz_origin, False, True, True)
        else:
            xyz_middle = xyz_origin

        ### scale
        xyz = xyz_middle * scale

        ### offset
        xyz -= xyz.min(0)

        ### merge the scene to the batch
        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_middle))
        feats.append(torch.from_numpy(features))
        bboxes.append(data_dict["instance_bboxes"])

    ### merge all the scenes in the batch
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

    locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    feats = torch.cat(feats, 0)  # float (N, C)

    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), full_scale[0], None)  # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, voxel_mode)
    

    return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
            'locs_float': locs_float, 'feats': feats, 'point_coords': xyz_origin,
            'offsets': batch_offsets, 'spatial_shape': spatial_shape,
            'gt_bbox': bboxes, 'point_coords_mean': data_dict['point_clouds_mean']}



def get_dataloader(args, scanrefer, all_scene_list, split, config, augment, scan2cad_rotation=None):
    dataset = ScannetForScan2CapPointGroupAllPoints(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=False,
        scan2cad_rotation=scan2cad_rotation
    )
    # Scan2C: dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=lambda batch: collate_train(batch, CONF.scale, CONF.full_scale,
                                                                        voxel_mode=CONF.mode,
                                                                        max_npoint=CONF.max_npoint,
                                                                        batch_size=CONF.batch_size, debug=CONF.debug),
                                num_workers=CONF.train_workers,
                                shuffle=True, sampler=None, drop_last=False,
                                pin_memory=True)

    elif split == 'val':
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=lambda batch: collate_val(batch, CONF.scale, CONF.full_scale,
                                                                    voxel_mode=CONF.mode,
                                                                    max_npoint=CONF.max_npoint,
                                                                    batch_size=CONF.batch_size, debug=CONF.debug),
                                num_workers=CONF.train_workers,
                                shuffle=True, sampler=None, drop_last=False,
                                pin_memory=True)
    elif split == 'test':
        dataloader = DataLoader(dataset, batch_size=1,
                                collate_fn=lambda batch: collate_test(batch, CONF.scale, CONF.full_scale,
                                                                    voxel_mode=CONF.mode,
                                                                    test_split='train',  # 'train' for overfitting, val
                                                                    batch_size=CONF.batch_size, debug=CONF.debug),
                                num_workers=CONF.test_workers,
                                shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise ValueError('dataloader not specified')

    return dataloader                                    
