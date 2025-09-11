import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
# your classes
from controlnet.flownet import DualFlowControlNet
from pipeline import StableDiffusionDualFlowControlNetPipeline
from controlnet.utils import load_controls_and_flows

# ---------------------------
# Load models (aligned SD-1.5)
# ---------------------------
dtype = torch.float32
base = "stable-diffusion-v1-5/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
scheduler = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")

# --- ControlNet: load your subclass weights ---
controlnet = DualFlowControlNet(
    block_out_channels=tuple(unet.config.block_out_channels),     # (320, 640, 1280, 1280)
    layers_per_block=2,
    cross_attention_dim=768,   
 )
# controlnet.load_state_dict(torch.load("path/to/controlnet.safetensors" or ".pth", map_location="cpu"))

# sanity: cross-attn dims must match (768 for SD1.x)
assert unet.config.cross_attention_dim == text_encoder.config.hidden_size == 768
if hasattr(controlnet, "config") and hasattr(controlnet.config, "cross_attention_dim"):
    assert controlnet.config.cross_attention_dim == 768, f"ControlNet CAD={controlnet.config.cross_attention_dim}"

ckpt = load_file('experiments/controlnet/checkpoint-46000/controlnet/diffusion_pytorch_model.safetensors')
controlnet.load_state_dict(ckpt,strict=False)

safety_checker = None
feature_extractor = None

# ---------------------------
# Build pipeline
# ---------------------------
pipe = StableDiffusionDualFlowControlNetPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
)
pipe = pipe.to("cuda")

# Validation data:
device = pipe.device

local_conditions = []
flow_conditions = []
prompts = ["A beautiful blonde girl smiling with pink lipstick with black background",
           "A Yacht with a red flag ,sailing in front of the Bosphorus in Istanbul , and bridge with cars is in the background." , 
           "A German shepherd shakes off water in the middle of a forest trail",
           "Honeybees hover among blooming purple flowers"]

videos = ['Beauty', 'Bosphorus', 'ShakeNDry', 'HoneyBee']
for video in videos:
    local,flow = load_controls_and_flows(
    f'data/{video}/images/frame_0000.png',
    f'data/{video}/images/frame_0004.png',
    f'data/{video}/optical_flow/optical_flow_gop_4_raft/flow_0000_0003.flo',
    f'data/{video}/optical_flow_bwd/optical_flow_gop_4_raft/flow_0004_0003.flo',
    size=(512, 512),
    device=device,
    dtype=dtype,
)
    local_conditions.append(local) 
    flow_conditions.append(flow)

image_logs = []
spacing = 20
img_size = 512
# --- 2. Define Experiment Parameters ---
# Define the scales you want to test
controlnet_scales = [ 1.35, 1.7]
guidance_scales = [3.5, 5.5]

# Enable FreeU for enhanced quality
pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
img_size = 512

# Create a directory to save the results
output_dir = "benchmark_results/scale_experiment2"
os.makedirs(output_dir, exist_ok=True)


# --- 3. Main Experiment Loop ---
for i, video in enumerate(videos):
    print(f"--- Processing video: {video} ---")
    
    # Load the ground truth image once per video
    gt_path = f"data/{video}/images/frame_0003.png"
    gt = Image.open(gt_path).convert("RGB").resize((img_size, img_size))
    gt_tensor = torch.from_numpy(np.array(gt).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    
    # Store results for this video
    results = []

    # Nested loops to iterate through the parameter grid
    for control_scale in controlnet_scales:
        for guide_scale in guidance_scales:
            print(f"  Generating with control_scale={control_scale}, guidance_scale={guide_scale}...")
            
            # Call the generation pipeline with the current scales
            out = pipe(
                prompt=prompts[i],
                controlnet_cond=local_conditions[i],
                flow_cond=flow_conditions[i],
                height=img_size,
                width=img_size,
                num_inference_steps=40,
                guidance_scale=guide_scale, # Use the current guidance scale
                negative_prompt=None,
                num_images_per_prompt=1, # Generate one image per setting for efficiency
                controlnet_conditioning_scale=control_scale, # Use the current controlnet scale
                guess_mode=False,
                output_type="pil",
                return_dict=True,
            )
            
            pred_image = out.images[0]
            
            # Compute metrics
            pred_tensor = torch.from_numpy(np.array(pred_image).transpose(2,0,1)).float().unsqueeze(0) / 255.0
            
            ms_ssim_val = ms_ssim(pred_tensor, gt_tensor, data_range=1.0).item()
            mse = F.mse_loss(pred_tensor, gt_tensor).item()
            psnr_val = 10 * np.log10(1.0 / mse) if mse != 0 else float('inf')
            
            # Store the result
            results.append({
                "control_scale": control_scale,
                "guide_scale": guide_scale,
                "image": pred_image,
                "psnr": psnr_val,
                "msssim": ms_ssim_val
            })

    # --- 4. Plotting the Results Grid for the Video ---
    nrows = len(controlnet_scales)
    ncols = len(guidance_scales)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    fig.suptitle(f'Parameter Sweep for "{video}"', fontsize=16)

    for idx, result in enumerate(results):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]
        
        ax.imshow(result['image'])
        title = (
            f"Control: {result['control_scale']}, Guide: {result['guide_scale']}\n"
            f"PSNR: {result['psnr']:.2f}, MS-SSIM: {result['msssim']:.3f}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{video}_scale_comparison.svg")
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"  ✅ Saved comparison grid to {save_path}\n")
    plt.close(fig)
