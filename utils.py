import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# ---------------------------
# Helpers: image & .flo loaders
# ---------------------------
def read_flo(path: str) -> np.ndarray:
    """Middlebury .flo → [H,W,2] float32 (pixel units)."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, 1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {path} (magic={magic})")
        w = int(np.fromfile(f, np.int32, 1)[0])
        h = int(np.fromfile(f, np.int32, 1)[0])
        data = np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)
    return data

def resize_flow_to(flow_hw2: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """Resize flow with bilinear and scale vectors to remain in pixel units."""
    ft = torch.from_numpy(flow_hw2).permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    _, _, H, W = ft.shape
    ft = F.interpolate(ft, size=(target_h, target_w), mode="bilinear", align_corners=True)
    ft[:, 0] *= (target_w / max(W, 1))
    ft[:, 1] *= (target_h / max(H, 1))
    return ft  # [1,2,target_h,target_w]

def load_pair_to_sixch(path0, path1, size=(512, 512)) -> torch.Tensor:
    """Two RGB images → [1,6,H,W] in [0,1]."""
    def load_rgb(p):
        img = Image.open(p).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BICUBIC)
        return TF.to_tensor(img)  # [3,H,W], float32
    a = load_rgb(path0)
    b = load_rgb(path1)
    return torch.cat([a, b], dim=0).unsqueeze(0)  # [1,6,H,W]

def load_controls_and_flows(
    img0_path, img1_path, fwd_flo_path, bwd_flo_path, size=(512, 512), device="cuda", dtype=torch.float32
):
    H, W = size
    sixch = load_pair_to_sixch(img0_path, img1_path, size=size).to(device=device, dtype=dtype)  # [1,6,H,W]

    fwd = read_flo(fwd_flo_path)
    bwd = read_flo(bwd_flo_path)
    fwd_t = resize_flow_to(fwd, H, W)
    bwd_t = resize_flow_to(bwd, H, W)
    flow4 = torch.cat([fwd_t, bwd_t], dim=1).to(device=device, dtype=dtype)  # [1,4,H,W]
    return sixch, flow4

def get_pred_original_sample(noise_scheduler, timesteps, sample, model_output, vae):
    """get predicted x_0 from inputs"""
    prediction_type = noise_scheduler.config.prediction_type
    alphas_cumprod = noise_scheduler.alphas_cumprod

    # 1. compute sqrt_alpha_prod, sqrt_one_minus_alpha_prod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if prediction_type == "epsilon":
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod * model_output) / sqrt_alpha_prod
    elif prediction_type == "v_prediction":
        pred_original_sample = sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * model_output
    else:
        raise ValueError(
            f"prediction_type given as {prediction_type} must be one of `epsilon`, or"
            " `v_prediction` for the DDPMScheduler."
        )

    # 3. scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * pred_original_sample
    with torch.no_grad():
        image = vae.decode(latents).sample

    # 4. clip to [-1, 1]
    image = image.clamp(-1.0, 1.0)
    return image
