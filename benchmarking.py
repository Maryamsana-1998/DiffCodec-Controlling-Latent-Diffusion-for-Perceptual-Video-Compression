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

def main(args):
    # List of videos
    videos = ['Beauty', 'Bosphorus', 'ShakeNDry', 'HoneyBee']

    # GOP size and frame info
    gop_size = args.gop
    total_frames = 96
    height, width = 1024, 1920
    intra_frames_per_video = int(total_frames / gop_size)
    inter_frames_per_video = total_frames - intra_frames_per_video

    # Dummy dicts (replace with yours)
    hevc_bits_gop2 = { 'inter': { 'Beauty': [131347, 4820, 70935], 'Bosphorus': [11285, 3787, 7356], 'ShakeNDry': [102325, 4451, 11174], 'HoneyBee': [8505, 4102, 5280] }, 'intra': { 'Beauty': [ 1782689 , 124592, 816837 ], 'Bosphorus': [1936592 , 173497, 960655], 'ShakeNDry': [ 1869760, 281731, 997810 ], 'HoneyBee': [1908508, 333640, 998749] } }
    hevc_bits_gop4 = { 'inter':{ 'Beauty': [ 342901, 13528 , 193302], 'Bosphorus': [ 41486, 5666, 17893], 'ShakeNDry': [401593, 8365, 87133], 'HoneyBee': [19772, 5314, 11341] }, 'intra':{ 'Beauty': [ 1457338 , 104885, 605498 ], 'Bosphorus': [1587414 , 155093, 814503], 'ShakeNDry': [ 1365120 , 205722, 813858 ], 'HoneyBee': [1559583, 217725, 819863]} }
    model_bits_dict = { 'inter': { 'Beauty': [1222, 1082], 'Bosphorus': [950, 708], 'ShakeNDry': [556, 564], 'HoneyBee': [556, 740] }, 'intra': { 'Beauty': 7759.64, 'Bosphorus': 22414.2, 'ShakeNDry':22499.55555, 'HoneyBee': 16640.0 } }

    # Select bits table based on GOP
    if gop_size == 2:
        hevc_bits = hevc_bits_gop2
    elif gop_size == 4:
        hevc_bits = hevc_bits_gop4
    else:
        raise ValueError(f"Unsupported GOP size {gop_size}")

    # Labels for HEVC folders
    bpp_dict = [0.1, 0.006, 0.05]

    results = {}

    for video in videos:
        results[video] = []

        for i in range(len(bpp_dict)):
            # ---------------- HEVC ----------------
            inter_bits = hevc_bits['inter'][video][i]
            intra_bits = hevc_bits['intra'][video][i]

            if args.inter:
                # Case 1: Inter-only
                bpp_inter = inter_bits * 8 / (inter_frames_per_video * height * width)
                metrics = evaluate_video(
                    f'data/{video}/images/',
                    f'benchmark_results/hevc_gop{gop_size}/{video}/bpp_{bpp_dict[i]}/',
                    all_frames=False,gop_size=gop_size
                )
                results[video].append({
                    'codec': 'hevc',
                    'bpp_inter': bpp_inter,
                    'inter_bits': inter_bits,
                    **metrics
                })
            else:
                # Case 2: Inter+Intra
                total_bits = (inter_bits + intra_bits) * 8
                bpp_total = total_bits / (total_frames * height * width)
                metrics = evaluate_video(
                    f'data/{video}/images/',
                    f'benchmark_results/hevc_gop{gop_size}/{video}/bpp_{bpp_dict[i]}/',
                    all_frames=True,gop_size=gop_size
                )
                results[video].append({
                    'codec': 'hevc',
                    'bpp_total': bpp_total,
                    'inter_bits': inter_bits,
                    'intra_bits': intra_bits,
                    **metrics
                })

        # ---------------- Our Model ----------------
        inter_bits_model = sum(model_bits_dict['inter'][video]) * inter_frames_per_video
        intra_bits_model = model_bits_dict['intra'][video] * intra_frames_per_video

        if args.inter:
            # Case 1: Inter-only
            bpp_inter_model = inter_bits_model * 8 / (inter_frames_per_video * height * width)
            metrics = evaluate_video(
                f'data/{video}/images/',
                f'benchmark_results/preds_gop{gop_size}_q4/{video}/',
                all_frames=False,gop_size=gop_size
            )
            results[video].append({
                'codec': 'ours',
                'bpp_inter': bpp_inter_model,
                'inter_bits': inter_bits_model,
                **metrics
            })
        else:
            # Case 2: Inter+Intra
            total_bits_model = (inter_bits_model + intra_bits_model) * 8
            bpp_total_model = total_bits_model / (total_frames * height * width)
            metrics = evaluate_video(
                f'data/{video}/images/',
                f'benchmark_results/preds_gop{gop_size}_q4/{video}/',
                all_frames=True,gop_size=gop_size
            )
            results[video].append({
                'codec': 'ours',
                'bpp_total': bpp_total_model,
                'inter_bits': inter_bits_model,
                'intra_bits': intra_bits_model,
                **metrics
            })

    # Save results
    out_file = f'benchmark_results/results_gop{gop_size}_{"inter" if args.inter else "all"}.npy'
    np.save(out_file, results)
    print(f"✅ Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gop", type=int, default=4, help="GOP size (2 or 4)")
    parser.add_argument("--inter", action="store_true", help="Evaluate inter-frames only")
    args = parser.parse_args()

    main(args)