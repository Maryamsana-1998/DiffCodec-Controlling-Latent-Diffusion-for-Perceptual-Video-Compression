import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
import matplotlib.pyplot as plt
# from fvd_utils.my_utils import calculate_fvd


# Initialize LPIPS and FID models
lpips_model = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
fid_model = FrechetInceptionDistance(feature=64).to('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    lambda x: x * 255  # Scale to 0-255
])

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def calculate_metrics_batch(original_images, pred_images):
    """
    Calculates PSNR, MS-SSIM, LPIPS, and FID metrics for a batch of image pairs.
    
    Args:
        original_images (list): List of PIL images for the original images.
        pred_images (list): List of PIL images for the predicted images.
    
    Returns:
        dict: Dictionary with metrics for the batch.
    """
    psnr_values, ms_ssim_values, lpips_values = [], [], []
    fid_model.reset()  # Clear any previous FID data
    org_frames =[]
    pred_frames =[]
    # Loop over the image pairs
    for original_image, pred_image in zip(original_images, pred_images):
        # Transform images
        original_tensor = transform(original_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        pred_tensor = transform(pred_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate PSNR and MS-SSIM
        psnr_value = psnr(original_tensor, pred_tensor).item()
        # print(psnr_value)
        if psnr_value > 1000:
            continue
        else:
            psnr_values.append(psnr_value)
        ms_ssim_values.append(ms_ssim(original_tensor, pred_tensor, data_range=255, size_average=True).item())

        # Calculate LPIPS
        lpips_value = lpips_model(original_tensor / 255.0, pred_tensor / 255.0).item()
        lpips_values.append(lpips_value)

        # Add tensors to FID model
        fid_model.update(original_tensor.to(torch.uint8), real=True)
        fid_model.update(pred_tensor.to(torch.uint8), real=False)

        org_frames.append(original_tensor)
        pred_frames.append(pred_tensor )

    # # Prepare 5D tensors for FVD: (B, T, C, H, W)
    # org_video = torch.stack(org_frames, dim=0).permute(1, 0, 2, 3, 4).repeat(2, 1, 1, 1, 1)
    # pred_video = torch.stack(pred_frames, dim=0).permute(1, 0, 2, 3, 4).repeat(2, 1, 1, 1, 1)
    
    # Compute metrics
    fid_value = fid_model.compute().item()
    # fvd_value = calculate_fvd(org_video, pred_video)

    return {
        "PSNR": sum(psnr_values) / len(psnr_values),
        "MS-SSIM": sum(ms_ssim_values) / len(ms_ssim_values),
        "LPIPS": sum(lpips_values) / len(lpips_values),
        "FID": fid_value,
        # "FVD": fvd_value
    }
