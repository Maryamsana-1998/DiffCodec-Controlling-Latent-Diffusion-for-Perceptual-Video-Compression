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


root_dir = "../UniControl_Video_Interpolation/evaluations/gop4/uvg/"
gop_size = 8
total_frames = 96
height, width = 1024, 1920
pixels_per_frame = height * width
intra_frames_per_video = total_frames // gop_size
inter_frames_per_video = total_frames - intra_frames_per_video

results = {}
inter_results = {}

for bpp_folder in os.listdir(root_dir):
    print(f"Processing : {bpp_folder}")
    bpp_path = os.path.join(root_dir, bpp_folder)
    if not os.path.isdir(bpp_path):
        print(bpp_folder, "is not a directory. Skipping...")
        continue

    results[bpp_folder] = {}
    inter_results[bpp_folder] = {}

    for video in os.listdir(bpp_path):
        print(f"  Evaluating : {video}")
        video_path = os.path.join(root_dir, bpp_folder, video)
        if not os.path.isdir(video_path):
            print(video_path, "is not a directory. Skipping...")
            continue

        original_path = os.path.join('data', video, 'images')
        print(f"Original path: {original_path}", video_path)

        # Evaluate metrics (assuming evaluate_video is defined elsewhere)
        metrics = evaluate_video(
            original_path,
            video_path,
            all_frames=False,
            gop_size=gop_size
        )
        all_metrics = evaluate_video(
            original_path,
            video_path,
            all_frames=True,
            gop_size=gop_size
        )

        inter_results[bpp_folder][video]= {
            'codec': 'UVC',
            **metrics
        }
        # results[bpp_folder][video]= {
        #     'codec': 'UVC',
        #     **all_metrics
        # }
        
result_path = os.path.join(root_dir ,'results.json')

inter_result_path = os.path.join(root_dir ,'inter_results.json')
# with open(result_path, "w") as f:
#     json.dump(results, f, indent=4)

with open(inter_result_path, "w") as f:
    json.dump(inter_results, f, indent=4)