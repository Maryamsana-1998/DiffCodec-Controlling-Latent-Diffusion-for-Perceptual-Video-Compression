# --- PRINT RESULTS ---
# pprint.pprint(bits_ours_data_uvg)
# bits_hevc_data = {
# "flow_sparse_fwd":{'RitualDance': 0.8863882211538462,
#  'BQTerrace': 0.9921123798076923,
#  'MarketPlace': 0.8050030048076923,
#  'Cactus': 0.6074669471153846,
#  'BasketballDrive': 0.8023287259615385}, 
# "flow_sparse_bwd" : {'RitualDance': 0.8463942307692308,
#  'BQTerrace': 1.0167067307692308,
#  'MarketPlace': 0.8957782451923076,
#  'Cactus': 0.6270132211538462,
#  'BasketballDrive': 0.7910757211538462}, 
# "intra_frames": {}}

import os
import re
import pprint
import numpy as np
import json

# --- CONFIG ---
# Base directory for videos
# Set to 'UVG' or 'Class_B'
base_dataset = 'UVG'  
base_dir = f"../UniControl_Video_Interpolation/data/{base_dataset}"

# Video resolution and total frames
WIDTH, HEIGHT = 1920, 1080
TOTAL_FRAMES = 96

# GOP sizes to evaluate
GOPS = [2, 4, 8]

# --- REGEXES FOR PARSING REPORTS ---
regex_arrow = re.compile(r"→\s*([\d.]+)\s*(B|KB|MB|KIB|MIB)?", re.IGNORECASE)  # file 1 & 3
regex_colon = re.compile(r":\s*([\d.]+)\s*(B|KB|MB|KIB|MIB)?", re.IGNORECASE)   # file 2

# --- FUNCTION TO PARSE ANY COMPRESSION REPORT ---
def parse_avg_size_any(report_path):
    """
    Parse a compression report and return the average size in BYTES.
    Handles:
        - Arrow format: '- Frame: flow_0000_0001.flo → 1.94 KB'
        - Colon format: 'flow_0000_0001.flo: 1406 bytes'
        - Intra frames: '- Frame: frame_0000.png → 6.82 KB'
    """
    sizes = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Try both regex formats
            match = regex_arrow.search(line) or regex_colon.search(line)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                if unit:
                    unit = unit.upper()
                    if unit in ["KB", "KIB"]:
                        val *= 1024
                    elif unit in ["MB", "MIB"]:
                        val *= 1024 * 1024
                    # else: B, leave as is
                sizes.append(val)
    return float(np.mean(sizes)) if sizes else 0.0

# --- FUNCTION TO GET REPORT PATHS ---
def get_report_path(video, kind):
    """
    Return the path to the report file based on the kind.
    kind: 'intra_frame', 'flow_sparse_fwd', 'flow_sparse_bwd', 'dense_flow'
    """
    paths = {
        "intra_frame": f"{base_dir}/{video}/intra_frames/decoded_q1/compression_report.txt",
        "flow_sparse_fwd": f"{base_dir}/{video}/optical_flow/cmp_gop_8_64/compression_report.txt",
        "flow_sparse_bwd": f"{base_dir}/{video}/optical_flow_bwd/cmp_gop_8_64/compression_report.txt",
        "dense_flow": f"{base_dir}/{video}/optical_flow/optical_flow_gop_2_raft_decoded/compression_report.txt"
    }
    return paths.get(kind, None)

# --- FUNCTION TO CALCULATE AVERAGE SIZE PER VIDEO ---
def calculate_video_avg_size(video):
    """
    Calculate average storage in KB per video for:
    - Intra frames
    - Sparse forward flow
    - Sparse backward flow
    - Dense flow
    """
    result = {}
    for kind in ["flow_sparse_fwd", "flow_sparse_bwd", "dense_flow", "intra_frame"]:
        path = get_report_path(video, kind)
        if path and os.path.exists(path):
            avg_bytes = parse_avg_size_any(path)
            result[kind] = avg_bytes / 1024  # store in KB
        else:
            result[kind] = None
    return result

# --- MAIN SCRIPT: PARSE ALL VIDEOS ---
videos = [v for v in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, v))]

# Dictionary to store average sizes
bits_ours_data_uvg = {kind: {} for kind in ["flow_sparse_fwd", "flow_sparse_bwd", "dense_flow", "intra_frame"]}

# Parse all videos and store average sizes
for video in videos:
    avg_stats = calculate_video_avg_size(video)
    for kind, val in avg_stats.items():
        bits_ours_data_uvg[kind][video] = val

print(bits_ours_data_uvg)
# --- CALCULATE BPP FOR EACH GOP ---
bpp_results = {}

for gop in GOPS:
    bpp_results[gop] = {}
    intra_frames = TOTAL_FRAMES // gop  # number of intra frames
    inter_frames = TOTAL_FRAMES - intra_frames  # number of inter frames

    for video in videos:
        # Convert average KB to bits for total storage
        intra_bits = bits_ours_data_uvg["intra_frame"][video] * intra_frames * 1024 * 8
        sparse_bits = (bits_ours_data_uvg["flow_sparse_fwd"][video] + bits_ours_data_uvg["flow_sparse_bwd"][video]) * inter_frames * 1024 * 8
        dense_bits = bits_ours_data_uvg["dense_flow"][video] * 2 * inter_frames * 1024 * 8  # multiplied by 2 because forward+backward dense

        # Bits per pixel (bpp)
        total_pixels = TOTAL_FRAMES * WIDTH * HEIGHT
        bpp_none = intra_bits / total_pixels
        bpp_sparse = (intra_bits + sparse_bits) / total_pixels
        bpp_dense = (intra_bits + dense_bits) / total_pixels
        # bpp_sparse = (sparse_bits) / total_pixels
        # bpp_dense = (dense_bits) / total_pixels

        bpp_results[gop][video] = {
            "none":bpp_none ,
            "sparse": bpp_sparse,
            "dense": bpp_dense
        }

# --- OPTIONAL: SAVE RESULTS TO JSON ---
output_json = f"benchmark_results/{base_dataset}_bpp_results.json"
with open(output_json, "w") as f:
    json.dump(bpp_results, f, indent=4)

# --- PRINT RESULTS ---
pprint.pprint(bpp_results)

mean_bpp_results = {}

for gop, videos_dict in bpp_results.items():
    mean_bpp_results[gop] = {}
    for condition in ["none", "sparse", "dense"]:
        # Collect all values that are not None
        values = [v[condition] for v in videos_dict.values() if v[condition] is not None]
        mean_bpp_results[gop][condition] = float(np.mean(values)) if values else None

# Print mean BPPs across all videos
pprint.pprint(mean_bpp_results)
