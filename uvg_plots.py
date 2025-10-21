import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

# Load and process H.264 and HEVC datasets
def load_and_process_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    flattened_data = []
    for video_name, records in data.items():
        if records:
            for record in records:
                record['video_name'] = video_name
                flattened_data.append(record)
    df = pd.DataFrame(flattened_data)
    return df.groupby('bpp_folder')[['total_bpp', 'LPIPS', 'MS-SSIM', 'PSNR', 'FID']].mean().sort_values('total_bpp')

# Function to load and compute mean metrics from a JSON file for a specific case
def compute_mean_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    mean_metrics = {}
    metrics = set()  # Collect unique metrics across all videos
    
    # Gather all metrics
    for video_data in data.values():
        metrics.update(video_data.keys())
    
    for metric in metrics:
        values = [video_data[metric] for video_data in data.values() if video_data.get(metric) is not None]
        mean_metrics[metric] = float(np.mean(values)) if values else None
    
    return mean_metrics

# Paths to JSON files for different cases (adjust paths as needed)
case_paths = {
    "none": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/uvg/none/all_videos_metrics.json",
    "sparse": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/uvg/sparse/all_videos_metrics.json",
    "dense": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/uvg/dense/all_videos_metrics.json"
}

bpp_data = {2: {'dense': 0.04193950283944557,
     'none': 0.03260499372715867,
     'sparse': 0.03560338692346853},
 4: {'dense': 0.030304260532009675,
     'none': 0.016302496863579336,
     'sparse': 0.020800086658044132},
 8: {'dense': 0.02448663937829173,
     'none': 0.008151248431789668,
     'sparse': 0.013398436525331929}}

# Compute average metrics for each case
avg_metrics_by_case = {}
for case, path in case_paths.items():
    avg_metrics_by_case[case] = compute_mean_metrics(path)


# BPP values for each case
bpps = bpp_data[8]

data_rows = []
for case in ['none', 'sparse', 'dense']:
    row = {"total_bpp": bpps[case]}
    row.update(avg_metrics_by_case[case])
    data_rows.append(row)

ours = pd.DataFrame(data_rows)

# Sort by total_bpp
ours = ours.sort_values('total_bpp').reset_index(drop=True)

# Load data
h264_grouped = load_and_process_json('benchmark_results/gop8_results/h264_class_b_gop8/results_fast.json')
hevc_grouped = load_and_process_json('benchmark_results/gop8_results/hevc_class_b_gop8/results_fast.json')

# DVC
DVC_uvg = pd.DataFrame({
    'total_bpp': [0.05, 0.10, 0.15, 0.20],
    'LPIPS': [0.155, 0.13, 0.121, 0.105],
    'FID': [22, 15, 11.5, 8],
    'MS-SSIM': [0.939, 0.953, 0.964, 0.971],
    'PSNR': [33.1, 34.85, 36.3, 37.5],
    'FVD': [19000, 10002, 7000, 4000]
}).sort_values('total_bpp')

# RLVC
RLVC_uvg = pd.DataFrame({
    'total_bpp': [0.07, 0.125, 0.2],
    'MS-SSIM': [0.965, 0.973, 0.98],
    'LPIPS': [0.135, 0.125, 0.101],
    'FID': [13, 12.3, 7],
    'PSNR': [35.5, 36.8, 37.7],
    'FVD': [8966, 3491, 1878]
}).sort_values('total_bpp')

# PLVC: Load CSV and rename BPP to total_bpp
plvc = pd.read_csv('benchmark_results/plvc_metrics.csv')
plvc = plvc.rename(columns={'BPP': 'total_bpp'}).sort_values('total_bpp')

# Save each dataset to CSV
datasets = [
    ('h264', h264_grouped),
    ('hevc', hevc_grouped),
    ('dvc', DVC_uvg),
    ('rlvc', RLVC_uvg),
    ('plvc', plvc),
    ('ours', ours),
]


# Define datasets and their plotting styles
datasets = [
    {'data': h264_grouped, 'label': 'H.264', 'color': '#BF0606', 'marker': 'o'},
    {'data': hevc_grouped, 'label': 'HEVC', 'color': '#CC6704', 'marker': 's'},
    {'data': DVC_uvg, 'label': 'DVC', 'color': '#0F753D', 'marker': '^'},
    {'data': RLVC_uvg, 'label': 'RLVC', 'color': '#70CCCF', 'marker': 'D'},
     {'data': plvc, 'label': 'PLVC', 'color': '#4D2DA1', 'marker': '+'},
    {'data': ours, 'label': 'Ours', 'color': '#AD34BA', 'marker': '*'}
]

# Metrics to plot
metrics = ['PSNR', 'MS-SSIM', 'LPIPS', 'FID']
titles = {
    'PSNR': 'PSNR vs Bitrate',
    'MS-SSIM': 'MS-SSIM vs Bitrate',
    'LPIPS': 'LPIPS vs Bitrate',
    'FID': 'FID vs Bitrate'
}
y_labels = {
    'PSNR': 'PSNR (dB)',
    'MS-SSIM': 'MS-SSIM',
    'LPIPS': 'LPIPS',
    'FID': 'FID'
}

# Create subplots (2x2 grid for the four metrics)
fig, axes = plt.subplots(1, 4, figsize=(32, 8),dpi= 1200)
axes = axes.flatten()  # Flatten for easier iteration

# Create subplots (2x2 grid for the four metrics)
fig, axes = plt.subplots(1, 4, figsize=(10, 8),dpi=1200)
axes = axes.flatten()  # Flatten for easier iteration

for metric in metrics:
    # Create a new figure for each metric
    plt.figure(figsize=(8, 6))  # Good figure size: 10x6 inches
    
    # Plot each dataset
    for dataset in datasets:
        plt.plot(
            dataset['data']['total_bpp'],
            dataset['data'][metric],
            marker=dataset['marker'],
            label=dataset['label'],
            color=dataset['color'],
            linewidth=2,
            markersize=8
        )
    
    # Customize plot
    plt.title(titles[metric], fontsize=12)
    plt.xlabel('Bitrate (bpp)', fontsize=10)
    plt.ylabel(y_labels[metric], fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot as high-DPI PDF
    filename = f'benchmark_results/rd_curve/uvg_gop8_{metric}.pdf'
    plt.savefig(filename, dpi=1200, bbox_inches='tight', format='pdf')  # 300 DPI for high quality
    plt.close()  # Close figure to free memory and avoid overlap
