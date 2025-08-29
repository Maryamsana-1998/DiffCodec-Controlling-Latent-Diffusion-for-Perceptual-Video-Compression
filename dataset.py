import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T
import torch.nn.functional as F
import random
import albumentations as A
from torch.utils.data import Dataset
from controlnet.dataset import load_flo_file, load_caption_dict

# ---------- helpers ----------
def fast_downsample_flow(flow, target_h=128, target_w=128):
    """Vectorized flow downsampling using PyTorch adaptive pooling."""
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    if flow.ndim == 3:  # [2,H,W]
        flow = flow.unsqueeze(0)  # [1,2,H,W]
    flow_ds = F.adaptive_avg_pool2d(flow.float(), (target_h, target_w))
    return flow_ds.squeeze(0).numpy()  # back to np [2,h,w]

def load_flow_cached(path, target_h=128, target_w=128):
    """Load pre-saved .npy flow if exists, otherwise fall back to .flo"""
    npy_path = str(path).replace(".flo", ".npy")
    if Path(npy_path).exists():
        flow = np.load(npy_path)
    else:
        flow = load_flo_file(path)   # your old loader
    return fast_downsample_flow(flow, target_h, target_w)

# ---------- dataset ----------
class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 index_file,
                 local_type_list,
                 resolution=512,
                 drop_txt_prob=0.3,
                 global_type_list=[],
                 keep_all_cond_prob=0.9,
                 drop_all_cond_prob=0.3,
                 drop_each_cond_prob=[0.3,0.3],
                 transform=True):

        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.transform = transform
        self.annos = load_caption_dict(anno_path)

        with open(index_file, "r") as f:
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
        anno = self.annos.get(sequence_id, "")

        # --- load main RGB ---
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Missing jpg at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))

        # --- local conditions (r1, r2, depth) ---
        local_files = {}
        for local_type in self.local_type_list:
            if local_type == 'r1':
                local_files['r1'] = img_path.with_name('r1.png')
            elif local_type == 'r2':
                local_files['r2'] = img_path.with_name('r2.png')
            elif local_type == 'depth':
                local_files['depth'] = img_path.parent / "depth" / img_path.name.replace(".png", "_depth.png")

        # Prepare inputs for augmentation — ensure SAME size for all
        image_inputs = {'image': image}
        for key, path in local_files.items():
            if path.exists():
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"[INVALID IMAGE] Could not load {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.resolution, self.resolution))
                image_inputs[key] = img

        # Apply augmentation
        if self.transform:
            augmented = self.augmentation(**image_inputs)
        else:
            augmented = image_inputs

        # Normalize jpg
        jpg = augmented['image'].astype(np.float32)
        jpg = (jpg / 127.5) - 1.0  # [-1,1]

        # Normalize local conditions
        local_conditions = []
        for k in ['r1', 'r2', 'depth']:
            if k in augmented:
                cond = augmented[k].astype(np.float32) / 255.0
                local_conditions.append(cond)

        if local_conditions:
            local_conditions = np.concatenate(local_conditions, axis=2)  # [H,W,C]
        else:
            print(f"[WARN] No local conditions for {img_path}, using zeros")
            local_conditions = np.zeros((self.resolution, self.resolution, 6), dtype=np.float32)

        # --- flow conditions ---
        flow_conditions = []
        if "flow" in self.local_type_list:
            flow_path = img_path.parent / "Flow" / img_path.name.replace(".png", ".flo")
            if flow_path.exists():
                flow_conditions = load_flow_cached(flow_path, 256, 256)
        if "flow_b" in self.local_type_list:
            flow_b_path = img_path.parent / "Flow_b" / img_path.name.replace(".png", ".flo")
            if flow_b_path.exists():
                flow_b = load_flow_cached(flow_b_path, 256, 256)
                if isinstance(flow_conditions, np.ndarray) and flow_conditions.size > 0:
                    flow_conditions = np.concatenate([flow_conditions, flow_b], axis=0)
                else:
                    flow_conditions = flow_b

        if isinstance(flow_conditions, list) or flow_conditions == []:
            print(f"[WARN] Missing flow for {img_path}, using zeros")
            flow_conditions = np.zeros((4, 256, 256), dtype=np.float32)

        # --- text dropout ---
        if random.random() < self.drop_txt_prob:
            anno = ""

        return {
            "jpg": jpg,                        # [H,W,3], float32
            "txt": anno,
            "local_conditions": local_conditions, # [H,W,6]
            "flow": flow_conditions             # [2,256,256] or [4,256,256]
        }
