import os
import json
from benchmarking import evaluate_video


root_dir = "benchmark_results/gop8_results/hevc_uvg_gop8"
gop_size = 8
total_frames = 96
height, width = 1024, 1920
pixels_per_frame = height * width
intra_frames_per_video = total_frames // gop_size
inter_frames_per_video = total_frames - intra_frames_per_video

results = {}
inter_results = {}

for video in os.listdir(root_dir):
    video_path = os.path.join(root_dir, video)
    if not os.path.isdir(video_path):
        continue

    results[video] = []
    inter_results[video] = []

    for bpp_folder in os.listdir(video_path):
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
            'codec': 'h264',
            'bpp_folder': bpp_folder,
            'inter_bpp': inter_bpp,
            **metrics
        })
        results[video].append({
            'codec': 'h264',
            'bpp_folder': bpp_folder,
            'total_bpp': total_bpp,
            **all_metrics
        })
        
result_path = os.path.join(root_dir ,'results.json')

inter_result_path = os.path.join(root_dir ,'inter_results.json')
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)

with open(inter_result_path, "w") as f:
    json.dump(inter_results, f, indent=4)