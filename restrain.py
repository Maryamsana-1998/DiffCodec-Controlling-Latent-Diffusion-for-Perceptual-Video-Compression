import glob
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image

from controlnet.dataset import UniDataset, ResidueDataset
from controlnet.flow_resnet import ResControlNet

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

# ---------------------------
# Setup device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float32

# ---------------------------
# Dataset
# ---------------------------
video_frames = glob.glob("/data2/local_datasets/test_vimeo/*/*.png")

train_dataset = UniDataset(
    anno_path="data/final_captions.txt",
    index_file="data/index_file_vll5.txt",
    local_type_list=["r1", "r2", "flow", "flow_b"],
    resolution=512,
)

train_dataset.video_frames = video_frames

wrapped_dataset = ResidueDataset(train_dataset)

train_dataloader = torch.utils.data.DataLoader(
    wrapped_dataset, batch_size=1, shuffle=True
)

print(wrapped_dataset[0].keys())

# ---------------------------
# Scheduler
# ---------------------------
noise_scheduler = DDPMScheduler(
    num_train_timesteps=500,           # shorter horizon is fine for residuals
    beta_start=1e-4,                   # very low starting noise
    beta_end=0.02,                     # keep final noise moderate
    beta_schedule="squaredcos_cap_v2", # cosine schedule (better SNR retention)
    prediction_type="epsilon",         # predict noise instead of x0
    clip_sample=True,                  # keep samples bounded
    variance_type="fixed_small"        # stable for training
)

# ---------------------------
# Load pretrained models (SD-1.5 backbone)
# ---------------------------
base = "stable-diffusion-v1-5/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=weight_dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=weight_dtype).to(device)
text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")

# ControlNet
controlnet = ResControlNet(
    block_out_channels=tuple(unet.config.block_out_channels),  # (320, 640, 1280, 1280)
    layers_per_block=2,
    cross_attention_dim=768,
).to(device)

# ---------------------------
# Training step (debug pass)
# ---------------------------
for batch in train_dataloader:
    print("Batch keys:", batch.keys(),batch["residual"].shape , batch['local_conditions'].shape)

    # Ground-truth residual image
    img_gt = batch["residual"].to(device=device, dtype=weight_dtype)

    # VAE encode to latents
    latents = vae.encode(img_gt).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    # Sample random timesteps
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=device,
    ).long()

    # Forward diffusion (add noise)
    noisy_latents = noise_scheduler.add_noise(
        latents.float(), noise.float(), timesteps
    ).to(dtype=weight_dtype)

    # ---------------------------
    # Text embeddings
    # ---------------------------
    input_ids = tokenizer(
        batch["txt"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)  # FIX: move to device

    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

    # ---------------------------
    # ControlNet conditions
    # ---------------------------
    controlnet_image = batch["local_conditions"].to(device=device, dtype=weight_dtype)
    controlnet_image = controlnet_image.permute(0,3,1,2)
    flow_cond = batch["flow"].to(device=device, dtype=weight_dtype)
    warped_cond = batch["warped_image"].to(device=device, dtype=weight_dtype)

    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_image,
        flow_cond=flow_cond,
        warp_cond=warped_cond,
        return_dict=False,
    )

    # ---------------------------
    # U-Net prediction
    # ---------------------------
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=[
            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
        return_dict=False,
    )[0]

    print("Model prediction shape:", model_pred.shape)
    break
