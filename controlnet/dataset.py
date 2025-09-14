import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T
import torch.nn.functional as F
import random
import albumentations as A
from torch.utils.data import Dataset
import struct
from controlnet.softsplat import softsplat
from controlnet.control_utils import compute_mask

# ---------- helpers ----------
def load_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIEH':
            raise Exception('Invalid .flo file')
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
        flow = np.resize(data, (2, height, width))
        return flow

def load_caption_dict(txt_path):
    caption_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # skip empty or malformed lines

            path, caption = line.split(':', 1)
            parts = path.strip().split('/')
            if len(parts) >= 3:
                parent1 = parts[-3].zfill(5)
                parent2 = parts[-2].zfill(4)
                key = f"{parent1}_{parent2}"
                caption_dict[key] = caption.strip()
    return caption_dict

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

# ---------- datasets ----------
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
                flow_conditions = load_flow_cached(flow_path,  self.resolution, self.resolution)
        if "flow_b" in self.local_type_list:
            flow_b_path = img_path.parent / "Flow_b" / img_path.name.replace(".png", ".flo")
            if flow_b_path.exists():
                flow_b = load_flow_cached(flow_b_path, self.resolution,  self.resolution)
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

class ResidueDataset(Dataset):
    """
    A Dataset wrapper that takes an existing dataset and transforms its output.

    It assumes the original dataset returns a dictionary with:
    - 'local_conditions': (512, 512, 6) -> Image1 and Image2 concatenated
    - 'flow_conditions': (512, 512, 4) -> Flow1 and Flow2 concatenated
    - 'ground_truth': (512, 512, 3) -> The target image

    This wrapper will:
    1. Extract Image1 and Flow1.
    2. Warp Image1 using Flow1.
    3. Calculate the residual between the Ground Truth and the Warped Image.
    4. Return a new dictionary with the processed data.
    """
    def __init__(self, original_dataset: Dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 1. Get the original data sample
        original_sample = self.original_dataset[idx]
        
        local_conditions = original_sample['local_conditions']
        flow_conditions = original_sample['flow']
        ground_truth = original_sample['jpg']

        # 2. Extract the first image and first flow
        # Assuming they are concatenated on the channel axis
        image1 = torch.from_numpy(local_conditions[:,:,:3]).permute(2,1,0).unsqueeze(0).to('cuda')
        # Shape: (3, 512, 512,)
        image2 = torch.from_numpy(local_conditions[:,:,3:]).permute(2,1,0).unsqueeze(0).to('cuda')
        #reshape image to 1,3,256,256 
        
        flow1 = torch.from_numpy(flow_conditions[:2]).unsqueeze(0).to('cuda')   # Shape: (2, 256, 256, )
        flow2 = torch.from_numpy(flow_conditions[2:]).unsqueeze(0).to('cuda')
        #reshape 1,2,256,256

        metric = torch.ones_like(flow1[:, :1]).to('cuda') # shape: [B, 1, H, W]
        conf_fwd, conf_bwd = metric, metric

        # print('metric', metric.shape,flow1.shape,image1.shape)

        with torch.cuda.amp.autocast(enabled=False):
            warped1 = softsplat(
                tenIn=image1,
                tenFlow=flow1,
                tenMetric=metric,
                strMode="soft"
            )  
            warped2 = softsplat(
                tenIn=image1,
                tenFlow=flow1,
                tenMetric=metric,
                strMode="soft"
            )  
            occ_fwd = compute_mask(flow1,flow2)
            occ_bwd = compute_mask(flow2,flow1)

        # === Fusion step (soft_fuse logic) ===
        eps = 1e-6
        conf = torch.cat([occ_fwd, occ_bwd], dim=1)   # [B,2,H,W]
        conf = torch.clamp(conf, min=0)
        w_sum = conf.sum(dim=1, keepdim=True) + eps
        w_norm = conf / w_sum

        fused = w_norm[:, :1] * warped1 + w_norm[:, 1:] * warped2

        # fusion needed 
        gt_tensor = torch.from_numpy(ground_truth).permute(2,1,0).unsqueeze(0).to('cuda')
        residual = gt_tensor - fused
        
        # 5. Return the new, processed sample
        processed_sample = {
            'warped_image': fused.squeeze(0) ,
            'flow': original_sample['flow'] , # Passing the original concatenated flow
            'txt': original_sample['txt'] ,
            'local_conditions': original_sample['local_conditions'],
            'residual': residual.squeeze(0) ,
        }
        
        return processed_sample