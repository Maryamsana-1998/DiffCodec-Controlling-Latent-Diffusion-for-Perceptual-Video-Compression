import sys
if './' not in sys.path:
	sys.path.append('./')
 
import os
import glob
from PIL import Image
from collections import defaultdict
import numpy as np
import pandas as pd
from test_utils import calculate_metrics_batch


# === Utility Functions ===
def get_png_paths(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])


def get_every_other(lst, skip=2):
    return [img for i, img in enumerate(lst) if i % skip != 0]

def load_image_pairs(original_paths, pred_paths, resize=False):
    images_a, images_b = [], []
    for p1, p2 in zip(original_paths, pred_paths):
        if os.path.exists(p1) and os.path.exists(p2):
            img_a = Image.open(p1).convert("RGB")
            img_b = Image.open(p2).convert("RGB")
            if resize:
                img_b = img_b.resize((1920, 1080), Image.Resampling.LANCZOS)
            images_a.append(img_a)
            images_b.append(img_b)
        else:
            print(f"⚠️ Missing: {p1} or {p2}")
    return images_a, images_b

def evaluate_video(original_folder, pred_folder):
    original_paths = get_every_other(get_png_paths(original_folder))
    pred_paths = get_every_other(get_png_paths(os.path.join(pred_folder)))
    print(pred_paths[:10])

    orig_imgs, pred_imgs = load_image_pairs(original_paths, pred_paths)

    if orig_imgs and pred_imgs:
        metrics = calculate_metrics_batch(orig_imgs, pred_imgs)
        print("✅ Metrics:", metrics)
        return metrics
    print("❌ No valid pairs.")
    return {}


# List of videos
videos = ['Beauty', 'Bosphorus', 'ShakeNDry', 'HoneyBee']

# GOP size and frame info
gop_size = 2
height, width = 1024, 1920
inter_frames_per_video = 48

# Bits per video
hevc_bits_dict = {
    'Beauty': [131347, 4820, 70935],
    'Bosphorus': [11285, 3787, 7356],
    'ShakeNDry': [102325, 4451, 11174],
    'HoneyBee': [8505, 4102, 5280]
}

model_bits_dict = {
    'Beauty': [1222, 1082],
    'Bosphorus': [950, 708],
    'ShakeNDry': [556, 564],
    'HoneyBee': [556, 740]
}

# Labels for HEVC folders
bpp_dict = [0.1, 0.006, 0.05]

# Store results as list of dicts per video
results = {}

for video in videos:
    results[video] = []

    # HEVC results
    for i, bits in enumerate(hevc_bits_dict[video]):
        bpp = bits * 8 / (inter_frames_per_video * height * width)
        folder = str(bpp_dict[i])

        metrics = evaluate_video(
            f'data/{video}/images/',
            f'hevc_gop{gop_size}/{video}/bpp_{folder}/'
        )

        results[video].append({
            'codec': 'hevc',
            'bpp': bpp,
            'bits': bits,
            **metrics   # merge PSNR, MS-SSIM, LPIPS, FID
        })

    # Our model results
    bits_total = sum(model_bits_dict[video]) * 8 * inter_frames_per_video
    bpp_model = bits_total / (inter_frames_per_video * height * width)

    metrics = evaluate_video(
        f'data/{video}/images/',
        f'preds_gop{gop_size}_q4/{video}/'
    )

    results[video].append({
        'codec': 'ours',
        'bpp': bpp_model,
        'bits': sum(model_bits_dict[video]),
        **metrics
    })

# Save results
np.save('inter_frame_results.npy', results)
