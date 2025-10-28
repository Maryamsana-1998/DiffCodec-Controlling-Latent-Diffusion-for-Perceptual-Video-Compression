import os
import json
import sys
if './' not in sys.path:
    sys.path.append('./')

import os
import argparse
import numpy as np
from PIL import Image
from test_utils import calculate_metrics_batch


# === Utility Functions ===
def get_png_paths(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])


def get_inter_frames(lst, gop_size=4):
    """
    Returns all inter-frames from a list of frame paths given a GOP size.
    Every gop_size-th frame is considered intra, and the rest are inter.
    """
    intra_frames = {i for i in range(0, len(lst), gop_size)}
    inter_frames = [img for i, img in enumerate(lst) if i not in intra_frames]
    return inter_frames



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


def evaluate_video(original_folder, pred_folder, all_frames=False,gop_size=4):
    if all_frames:
        original_paths = get_png_paths(original_folder)
        pred_paths = get_png_paths(pred_folder)
    else:
        original_paths = get_inter_frames(get_png_paths(original_folder), gop_size)
        pred_paths = get_inter_frames(get_png_paths(pred_folder), gop_size)

    orig_imgs, pred_imgs = load_image_pairs(original_paths, pred_paths)

    if orig_imgs and pred_imgs:
        metrics = calculate_metrics_batch(orig_imgs, pred_imgs)
        return metrics
    print("❌ No valid pairs.")
    return {}


root_dir = "benchmark_results/gop8_results/h264_class_b_gop8"
gop_size = 4
total_frames = 96
height, width = 1024, 1920
pixels_per_frame = height * width
intra_frames_per_video = total_frames // gop_size
inter_frames_per_video = total_frames - intra_frames_per_video

results = {}
inter_results = {}

for video in os.listdir(root_dir):
    print(f"Processing video: {video}")
    video_path = os.path.join(root_dir, video)
    if not os.path.isdir(video_path):
        continue

    results[video] = []
    inter_results[video] = []

    for bpp_folder in os.listdir(video_path):
        print(f"  Evaluating bpp folder: {bpp_folder}")
        bpp_path = os.path.join(video_path, bpp_folder)
        if not os.path.isdir(bpp_path):
            continue

        original_path = os.path.join('data', video, 'images')

        # Evaluate metrics (assuming evaluate_video is defined elsewhere)
        metrics = evaluate_video(
            original_path,
            bpp_path,
            all_frames=False,
            gop_size=gop_size
        )
        all_metrics = evaluate_video(
            original_path,
            bpp_path,
            all_frames=True,
            gop_size=gop_size
        )

        # Read intra/inter byte stats
        stats_path = os.path.join(bpp_path, "intra_inter_storage.txt")
        if not os.path.isfile(stats_path):
            continue

        with open(stats_path, "r") as f:
            values = {}
            for line in f:
                if ":" in line:
                    key, val = line.split(":")
                    values[key.strip()] = int(val.strip())

        inter_bytes = values.get('Inter bytes', 0)
        total_bytes = values.get('Total bytes', 0)

        total_bpp = (total_bytes * 8) / (total_frames * pixels_per_frame)
        inter_bpp = (inter_bytes * 8) / (inter_frames_per_video * pixels_per_frame)

        inter_results[video].append({
            'codec': 'UVC',
            'bpp_folder': bpp_folder,
            'inter_bpp': inter_bpp,
            **metrics
        })
        results[video].append({
            'codec': 'hevc',
            'bpp_folder': bpp_folder,
            'total_bpp': total_bpp,
            **all_metrics
        })
        
result_path = os.path.join(root_dir ,'results_fast.json')

inter_result_path = os.path.join(root_dir ,'inter_results_fast.json')
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)

with open(inter_result_path, "w") as f:
    json.dump(inter_results, f, indent=4)