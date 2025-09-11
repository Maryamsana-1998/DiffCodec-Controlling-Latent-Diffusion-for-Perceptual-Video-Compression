import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import glob
from controlnet.dataset import UniDataset
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from controlnet.softsplat import softsplat
import torchvision
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tensor_to_displayable_image(tensor, is_residual=False):
    """
    Converts a PyTorch tensor (C, H, W) to a NumPy array (H, W, C)
    that can be displayed by Matplotlib.
    """
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    
    if is_residual:
        # Normalize the residual to the [0, 1] range for visualization
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
    
    # Clip to ensure the values are in the valid [0, 1] range for floats
    return np.clip(image, 0, 1)


def visualize_samples(dataset, num_samples=5):
    """
    Selects random samples from a dataset and plots the ground truth,
    warped image, and residual.
    """
    if len(dataset) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}.")
        num_samples = len(dataset)

    # Get random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create a subplot grid
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle("Dataset Visualization: Ground Truth vs. Warped vs. Residual", fontsize=16)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        gt = tensor_to_displayable_image(sample['ground_truth'].squeeze(0))
        warped = tensor_to_displayable_image(sample['warped_image'].squeeze(0))
        residual = tensor_to_displayable_image(sample['residual'].squeeze(0), is_residual=True)

        # Plot Ground Truth
        ax = axs[i, 0]
        ax.imshow(gt)
        ax.set_title(f"Sample #{idx}\nGround Truth")
        ax.axis('off')

        # Plot Warped Image
        ax = axs[i, 1]
        ax.imshow(warped)
        ax.set_title(f"Sample #{idx}\nWarped Image")
        ax.axis('off')

        # Plot Residual
        ax = axs[i, 2]
        ax.imshow(residual)
        ax.set_title(f"Sample #{idx}\nResidual (Normalized)")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_image_batch(tensor):
    """
    Visualizes a batch of 8 images from a PyTorch tensor.

    Args:
        tensor (torch.Tensor): A tensor of shape [8, 3, 256, 256].
    """
    # Ensure the tensor has the correct shape
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (8, 3, 256, 256):
        print(f"Error: Input must be a torch.Tensor with shape [8, 3, 256, 256], but got {tensor.shape}")
        return

    # Create a 2x4 grid for plotting
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Visualization of Noisy X Tensor", fontsize=16)

    # Move tensor to CPU and detach from computation graph
    tensor = tensor.cpu().detach()

    for i in range(8):
        # Determine the subplot's row and column
        row = i // 4
        col = i % 4

        # --- Convert Tensor to Displayable Image ---
        # 1. Select the i-th image from the batch
        # 2. Permute axes from (C, H, W) to (H, W, C) for Matplotlib
        # 3. Convert to NumPy array
        img_np = tensor[i].permute(1, 2, 0).numpy()
        
        # 4. Clip values to the valid [0, 1] range for display
        img_np = np.clip(img_np, 0, 1)

        # Display the image
        ax = axs[row, col]
        ax.imshow(img_np)
        ax.set_title(f"Image #{i + 1}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def compute_mask(flow_bwd_tensor, flow_fwd_tensor):
    metric = torch.ones_like(flow_fwd_tensor[:, :1])
    with torch.cuda.amp.autocast(enabled=False):
        warped_bwd = softsplat(tenIn=flow_bwd_tensor, tenFlow=flow_fwd_tensor, tenMetric=metric, strMode='soft')
    flow_diff = flow_fwd_tensor + warped_bwd
    occ_mask = (torch.norm(flow_diff, p=2, dim=1, keepdim=True) > 0.3).float()
    return occ_mask


class WarpingDatasetWrapper(Dataset):
    """
    A Dataset wrapper that takes an existing dataset and transforms its output.

    It assumes the original dataset returns a dictionary with:
    - 'local_conditions': (512, 512, 6) -> Image1 and Image2 concatenated
    - 'flow_conditions': (256, 256, 4) -> Flow1 and Flow2 concatenated
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
        image1 = torch.from_numpy(local_conditions[:,:,:3]).permute(2,1,0).unsqueeze(0).to(device) # Shape: (3, 512, 512,)
        image2 = torch.from_numpy(local_conditions[:,:,3:]).permute(2,1,0).unsqueeze(0).to(device)
        #reshape image to 1,3,256,256 
        
        flow1 = torch.from_numpy(flow_conditions[:2]).unsqueeze(0).to(device)   # Shape: (2, 256, 256, )
        flow2 = torch.from_numpy(flow_conditions[2:]).unsqueeze(0).to(device)
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
        conf = torch.cat([conf_fwd, conf_bwd], dim=1)   # [B,2,H,W]
        conf = torch.clamp(conf, min=0)
        w_sum = conf.sum(dim=1, keepdim=True) + eps
        w_norm = conf / w_sum

        fused = w_norm[:, :1] * warped1 + w_norm[:, 1:] * warped2

        # Handle double holes (both invalid)
        holes = (occ_fwd + occ_bwd) > 1.5   # both occluded
        if holes.any():
            avg = 0.5 * (warped1 + warped2)
            fused = torch.where(holes.expand_as(fused), avg, fused)

        # fusion needed 

        
        gt_tensor = torch.from_numpy(ground_truth).permute(2,1,0).unsqueeze(0).to('cuda')
        residual = gt_tensor - fused
        
        # 5. Return the new, processed sample
        processed_sample = {
            'warped_image': fused,
            'flow': flow1, # Passing the original concatenated flow
            'ground_truth': gt_tensor,
            'residual': residual
        }
        
        return processed_sample
