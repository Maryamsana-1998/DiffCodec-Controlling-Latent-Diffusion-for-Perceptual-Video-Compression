import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

def average_metrics_from_json(json_path):
    """
    Reads a JSON file containing metrics per video and resolution,
    and returns a DataFrame with the average values across videos
    for each resolution block.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with columns:
        ['total_bpp', 'MS-SSIM', 'LPIPS', 'FID', 'PSNR', 'FVD']
    """
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Prepare lists
    resolutions = []
    bpps, msssim, lpips, fid, psnr, fvd = [], [], [], [], [], []

    # Compute averages per resolution
    for key, videos in data.items():
        vals = pd.DataFrame(videos).T  # videos as rows
        resolutions.append(key)
        bpps.append(vals["bpp"].mean())
        msssim.append(vals["MS-SSIM"].mean())
        lpips.append(vals["LPIPS"].mean())
        fid.append(vals["FID"].mean())
        psnr.append(vals["PSNR"].mean())
        fvd.append(vals["FVD"].mean())

    # Create DataFrame
    df = pd.DataFrame({
        "total_bpp": bpps,
        "MS-SSIM": msssim,
        "LPIPS": lpips,
        "FID": fid,
        "PSNR": psnr,
        "FVD": fvd
    }, index=resolutions)

    return df


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
    "none": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/hevc/none/all_videos_metrics.json",
    "sparse": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/hevc/sparse/all_videos_metrics.json",
    "dense": "/data/maryamsana_98/UniControl_Video_Interpolation/evaluations/gop8/hevc/dense/all_videos_metrics.json"
}

# Compute average metrics for each case
avg_metrics_by_case = {}
for case, path in case_paths.items():
    avg_metrics_by_case[case] = compute_mean_metrics(path)


# BPP values for each case
bpps = {'dense': 0.02433612870366008,
     'none': 0.010576381713085276,
     'sparse': 0.016294097465696863}
# gop 4 hevc
# Build a DataFrame
data_rows = []
for case in ['none', 'sparse', 'dense']:
    row = {"total_bpp": bpps[case]}
    row.update(avg_metrics_by_case[case])
    data_rows.append(row)

ours = pd.DataFrame(data_rows)

# Sort by total_bpp
ours = ours.sort_values('total_bpp').reset_index(drop=True)
# ours['FVD'] = [220000, 143002, 94300]
print("\nDataFrame with average metrics across cases:")
print(ours)
 

 # Create output directory
output_dir = "benchmark_results/csv_for_latex_classb_gop4"
os.makedirs(output_dir, exist_ok=True)

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
    return df.groupby('bpp_folder')[['total_bpp', 'LPIPS', 'MS-SSIM', 'PSNR', 'FID','FVD']].mean().sort_values('total_bpp')

# Load data
h264_grouped = load_and_process_json('benchmark_results/gop8_results/h264_class_b_gop8/results_fast.json')
hevc_grouped = load_and_process_json('benchmark_results/gop8_results/hevc_class_b_gop8/results_fast.json')

# DVC
DVC_classb = pd.DataFrame({
    'total_bpp': [0.1, 0.2, 0.3],
    'LPIPS': [0.156, 0.135, 0.10],
    'FID': [74, 40, 28.5 ],
    'MS-SSIM': [0.942, 0.955, 0.962],
    'PSNR': [31.5, 33.0, 34.0],
    'FVD': [35000, 25000, 20000]
}).sort_values('total_bpp')

# RLVC
RLVC_classb = pd.DataFrame({
    'total_bpp': [0.060807, 0.097379, 0.165579],
    'MS-SSIM': [0.989323, 0.993480, 0.995341],
    'LPIPS': [0.020047, 0.011311, 0.005088],
    'FID': [0.078204, 0.058928, 0.009942],
    'PSNR': [37.707968, 40.265765, 41.997304],
    'FVD': [18223.082329, 5635.863374, 2166.578799]
}).sort_values('total_bpp')

diffvc= pd.DataFrame({
    'total_bpp': [0.03, 0.06, 0.11, 0.15],
    'LPIPS': [0.124,0.085,0.075,0.07],
    'FID': [20, 12, 8, 4.2 ],
    'MS-SSIM': [0.90, 0.925, 0.935, 0.938],
    'PSNR': [26,27,27.5, 27.8],
    'FVD': [890000, 755600, 675956,578000]
}).sort_values('total_bpp')



# # PLVC: Load CSV and rename BPP to total_bpp
plvc = average_metrics_from_json('benchmark_results/plvc_classb_results.json')
plvc = plvc.reset_index(drop=True).sort_values("total_bpp")
print(plvc)

# Save each dataset to CSV
datasets = [
    ('h264', h264_grouped),
    ('hevc', hevc_grouped),
    ('dvc', DVC_classb),
    ('rlvc', RLVC_classb),
    ('plvc', plvc),
    ('ours', ours),
    ('diffvc_uvg', diffvc)
]

for name, df in datasets:
    df.to_csv(f"{output_dir}/{name}_data.csv", index=True)

print("CSV files saved in", output_dir)


# Define datasets and their plotting styles

datasets = [
    {'data': h264_grouped, 'label': 'H.264', 'color': '#8B4513', 'marker': 'o'},
    {'data': hevc_grouped, 'label': 'HEVC', 'color': '#FAA502', 'marker': 's'},
    {'data': DVC_classb, 'label': 'DVC', 'color': '#0D00FF', 'marker': '^'},
    {'data': RLVC_classb, 'label': 'RLVC', 'color': '#1F8000', 'marker': 'D'},
     {'data': plvc, 'label': 'PLVC', 'color': '#800080', 'marker': '+'},
    {'data': ours, 'label': 'Ours', 'color': '#F71702', 'marker': '*'},
    {'data': diffvc, 'label': 'DiffVC', 'color': '#00CED1', 'marker': 'x'},
]
# Metrics to plot
metrics = ['PSNR', 'MS-SSIM', 'LPIPS', 'FID','FVD']
titles = {
    'PSNR': 'PSNR vs Bitrate',
    'MS-SSIM': 'MS-SSIM vs Bitrate',
    'LPIPS': 'LPIPS vs Bitrate',
    'FID': 'FID vs Bitrate',
    'FVD': 'FVD vs Bitrate'
}
y_labels = {
    'PSNR': 'PSNR (dB)',
    'MS-SSIM': 'MS-SSIM',
    'LPIPS': 'LPIPS',
    'FID': 'FID',
    'FVD': 'FVD'
}

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
    filename = f'benchmark_results/rd_curve/classb_gop8_{metric}.pdf'
    plt.savefig(filename, dpi=800, bbox_inches='tight', format='pdf')  # 300 DPI for high quality
    plt.close()  # Close figure to free memory and avoid overlap
