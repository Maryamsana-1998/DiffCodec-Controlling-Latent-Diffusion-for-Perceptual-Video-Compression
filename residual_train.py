# %%
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel

from controlnet.dataset import UniDataset
from controlnet.softsplat import softsplat
from residual_utils import *

# -------------------------------
# Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "outputs"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# Dataset
# -------------------------------
video_frames = glob.glob('/data2/local_datasets/test_vimeo/*/*.png')

train_dataset = UniDataset(
    anno_path='data/final_captions.txt',
    index_file='data/index_file_vll5.txt',
    local_type_list=['r1', 'r2', 'flow', 'flow_b'],
    resolution=256
)
train_dataset.video_frames = video_frames
wrapped_dataset = WarpingDatasetWrapper(train_dataset)

visualize_samples(wrapped_dataset, num_samples=2)

train_dataloader = DataLoader(wrapped_dataset, batch_size=1, shuffle=True)

# -------------------------------
# Scheduler
# -------------------------------
noise_scheduler = DDPMScheduler(
    num_train_timesteps=500,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
    clip_sample=True,
    variance_type="fixed_small"
)

plt.figure()
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")
plt.savefig(os.path.join(save_dir, "noise_schedule.png"))
plt.close()

# -------------------------------
# Example batch visualization
# -------------------------------
xb = next(iter(train_dataloader))["residual"].to(device)[:8].squeeze(0)
plt.imshow(xb.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.savefig(os.path.join(save_dir, "clean_sample.png"))
plt.close()

# -------------------------------
# Model
# -------------------------------
model = UNet2DModel(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
losses = []

# -------------------------------
# Training loop
# -------------------------------
epochs = 30
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["residual"].squeeze(0).to(device)

        # Sample noise
        noise = torch.randn_like(clean_images)
        bs = clean_images.shape[0]

        # Sample timestep
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()

        # Add noise
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict noise
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 5 == 0:
        avg_loss = np.mean(losses[-len(train_dataloader):])
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# -------------------------------
# Save model
# -------------------------------
torch.save(model.state_dict(), os.path.join(save_dir, "unet_residual.pth"))

# -------------------------------
# Plot training loss
# -------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[0].set_title("MSE Loss")
axs[1].plot(np.log(losses))
axs[1].set_title("Log Loss")
plt.savefig(os.path.join(save_dir, "training_loss.png"))
plt.close()

# -------------------------------
# Run inference after training
# -------------------------------
model.eval()
with torch.no_grad():
    test_batch = next(iter(train_dataloader))["residual"].squeeze(0).to(device)
    noise = torch.randn_like(test_batch)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (1,), device=device).long()
    noisy_test = noise_scheduler.add_noise(test_batch, noise, timesteps)

    pred_residual = model(noisy_test, timesteps).sample

# Save outputs
show_image_batch(noisy_test).save(os.path.join(save_dir, "noisy_test.png"))
show_image_batch(pred_residual).save(os.path.join(save_dir, "pred_residual.png"))
