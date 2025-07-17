import os
import random
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from annotator.content import ContentDetector

from .util import load_caption_dict, keep_and_drop, load_flo_file, adaptive_weighted_downsample, normalize_for_warping

class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 index_file,
                 local_type_list,
                 global_type_list=[],
                 resolution= 512,
                 drop_txt_prob=0.5,
                 keep_all_cond_prob=0.9,
                 drop_all_cond_prob=0.5,
                 drop_each_cond_prob=[0.3,0.3],
                 transform=False):

        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.transform = transform
        self.annos = load_caption_dict(anno_path)

        with open(index_file, 'r') as f:
            self.video_frames = f.read().splitlines()


        self.aug_targets ={}
        # Albumentations key remapping
        if 'r1' in self.local_type_list:
            self.aug_targets['r1'] = 'image'
        if 'r2' in self.local_type_list:
            self.aug_targets['r2'] = 'image'
        if 'depth' in self.local_type_list:
            self.aug_targets['depth'] = 'image'

        if self.transform:
            self.augmentation = A.Compose([
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.1, p=1.0
                ),
            ], additional_targets=self.aug_targets)

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, index):
        img_path = Path(self.video_frames[index])
        sequence_id = f"{img_path.parent.parent.name.zfill(5)}_{img_path.parent.name}"
        anno = self.annos[sequence_id]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # needs to expanded
        global_files = []
        for global_type in self.global_type_list:
            if global_type == 'r2':
                global_files.append(img_path.with_name('r2.npy'))

        local_files = {}
        for local_type in self.local_type_list:
            if local_type == 'r1':
                local_files['r1'] = img_path.with_name('r1.png')
            elif local_type == 'r2':
                local_files['r2'] = img_path.with_name('r2.png')
            elif local_type == 'depth':
                local_files['depth'] = img_path.parent / 'depth' / img_path.name.replace('.png', '_depth.png')
            elif local_type == 'flow':
                local_files['flow'] = img_path.parent / 'Flow' / img_path.name.replace('.png', '.flo')
            elif local_type == 'flow_b':
                local_files['flow_b'] = img_path.parent / 'Flow_b' / img_path.name.replace('.png', '.flo')

        # Prepare inputs for augmentation
        image_inputs = {'image': image}
        for key in local_files.keys():
            path = local_files.get(key, None)
            if key not in ['flow','flow_b'] and path.exists():
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"[INVALID IMAGE] Could not load {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_inputs[key] = img

        # Apply augmentation
        if self.transform:
            augmented = self.augmentation(**image_inputs)
            # print('aug done')
        else:
            augmented = {k: cv2.resize(v, (self.resolution, self.resolution)) for k, v in image_inputs.items()}

        # Normalize and prepare outputs
        jpg = augmented['image']
        jpg = cv2.resize(jpg, (self.resolution, self.resolution))
        jpg = (jpg.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1]

        local_conditions = []
        for k in ['r1','r2','depth']:
            if k in augmented:
                cond = cv2.resize(augmented[k], (self.resolution, self.resolution))
                cond = cond.astype(np.float32) / 255.0
                local_conditions.append(cond)

        # Handle flow (resize + normalize only)
        flow_conditions = []
        flow, flow_b = None,None
        if 'flow' in local_files and local_files['flow'].exists():
            flow = load_flo_file(local_files['flow'])
            flow = adaptive_weighted_downsample(flow, target_h=256, target_w=256)
            # flow = normalize_for_warping(flow)


        if 'flow_b' in local_files and local_files['flow_b'].exists():
            flow_b = load_flo_file(local_files['flow_b'])
            flow_b = adaptive_weighted_downsample(flow_b, target_h=256, target_w=256)
            # flow_b = normalize_for_warping(flow_b)

        if flow is not None and flow_b is not None:
            flow_conditions = np.concatenate([flow,flow_b])
        elif flow is not None:
            flow_conditions = normalize_for_warping(flow)
        else:
            print('no flow used')
            pass

        global_conditions = []
        for global_file in global_files:
            condition = np.load(str(global_file))
            global_conditions.append(condition)

        # Drop text or conditions as per policy
        if random.random() < self.drop_txt_prob:
            anno = ""

        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob,
                                        self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob,
                                        self.drop_all_cond_prob, self.drop_each_cond_prob)

        if len(local_conditions):
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions):
            global_conditions = np.concatenate(global_conditions)

        if local_conditions.shape != (self.resolution, self.resolution,6):
            raise ValueError(f"[ERROR] Condition shape mismatch for '{local_files}': got {local_conditions.shape}, expected (3, {self.resolution}, {self.resolution})")

        return {
            'jpg': jpg,
            'txt': anno,
            'local_conditions': local_conditions ,
            'global_conditions': global_conditions ,
            'flow': flow_conditions
        }