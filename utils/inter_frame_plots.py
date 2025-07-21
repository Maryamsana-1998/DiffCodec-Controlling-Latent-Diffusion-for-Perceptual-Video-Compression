# %%
import os
from PIL import Image
import glob
from test_utils import *
from collections import defaultdict
import numpy as np
import pandas as pd

def get_png_paths(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])

# %%
flow_storage = {
    "Beauty": 16525.653333,
    "Bosphorus": 15291.733333,
    "HoneyBee": 14057.813333,
    "Jockey": 28380.160000,
    "ReadySteadyGo": 16112.640000,
    "ShakeNDry": 14059.520000,
    "YachtRide": 31167.146667
}

# %%
def evaluate_video_uni(original_folder, pred_folder,gop =2):
    original_paths = get_png_paths(original_folder)[1::gop]
    pred_paths = get_png_paths(pred_folder)[1::gop]
    print(original_paths[0:10],len(original_paths),len(pred_paths))

    original_eval_images = []
    pred_eval_images = []

    for original_path, pred_path in zip(original_paths, pred_paths):
        if os.path.exists(original_path) and os.path.exists(pred_path):
            original_eval_images.append(Image.open(original_path).convert("RGB"))
            pred_eval_images.append(Image.open(pred_path).convert("RGB").resize((1920,1080),Image.Resampling.LANCZOS))
        else:
            print(f"Warning: Missing image for: {original_path} or {pred_path}")

    if original_eval_images and pred_eval_images:
        metrics = calculate_metrics_batch(original_eval_images, pred_eval_images)
        print("Evaluation metrics:", metrics)
        return metrics
    else:
        print("No valid image pairs for evaluation.")
        return {}

def evaluate_all_videos_uni(original_root, pred_root,gop=2, warp=False):
    all_metrics = defaultdict(list)
    video_names = sorted(os.listdir(pred_root))
    print(video_names)

    for video in video_names:
        orig_path = os.path.join(original_root, video, "images")
        pred_path = os.path.join(pred_root, video)

        if os.path.exists(orig_path) and os.path.exists(pred_path):
            print(f"Evaluating {video}...")
            metrics = evaluate_video_uni(orig_path, pred_path,gop)
            if warp:
                no = 96 - (96/gop)
                metrics['bpp'] = (flow_storage[video]*no*2)/(1920*1080*no)
            else:
                metrics['bpp'] = 0
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        else:
            print(f"Missing data for {video}. Skipping...")

    # Final mean metrics
    print("\n🔍 Mean Evaluation Over All UVG Videos:")
    mean_metrics = {}
    for key, values in all_metrics.items():
        mean_val = np.mean(values)
        mean_metrics[key] = mean_val
        print(f"{key}: {mean_val:.4f}")
    return mean_metrics

# %%
crf_4 = evaluate_all_videos_uni(
    original_root="data/UVG",
    pred_root="experiments/bi_warp_v3/preds_150k_gop2_q1/", gop = 2, warp= True
)
crf_4['CRF'] = 1

crf_1 = evaluate_all_videos_uni(
    original_root="data/UVG",
    pred_root="experiments/bi_warp_v3/preds_150k_gop2_q4/",gop = 2, warp= True
)
crf_1['CRF'] = 4

bi_df = pd.DataFrame([crf_4,crf_1])
bi_df

bi_df.to_csv('data/bi_warp_frame_interpolation.csv')


