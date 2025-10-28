import json
import numpy as np
import pandas as pd
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
# ours['FVD'] = [190000, 103002, 74000]
print(ours['FVD'])

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

diffvc_uvg = pd.DataFrame({
    'total_bpp': [0.02, 0.05,0.1,0.155],
    'LPIPS': [0.25, 0.065,0.095, 0.014],
    'FID': [5, 4.3 ,2.1, 1.09 ],
    'MS-SSIM': [0.91,0.93,0.95,0.956],
    'PSNR': [30.3, 31.6, 32.3,32.5],
    'FVD': [700000, 670000, 500200,350000]
}).sort_values('total_bpp')

# RLVC
# RLVC_uvg = pd.DataFrame({
#     'total_bpp': [0.07, 0.125, 0.2],
#     'MS-SSIM': [0.965, 0.973, 0.98],
#     'LPIPS': [0.135, 0.125, 0.101],
#     'FID': [13, 12.3, 7],
#     'PSNR': [35.5, 36.8, 37.7],
#     'FVD': [8966, 3491, 1878]
# }).sort_values('total_bpp')


RLVC_uvg = average_metrics_from_json('benchmark_results/rlvc_uvg_results.json')
RLVC_uvg  = RLVC_uvg.reset_index(drop=True).sort_values("total_bpp")


# PLVC: Load CSV and rename BPP to total_bpp
plvc = pd.read_csv('benchmark_results/plvc_metrics_uvg.csv')
plvc = plvc.rename(columns={'BPP': 'total_bpp'}).sort_values('total_bpp')

# Save each dataset to CSV
datasets = [
    ('h264', h264_grouped),
    ('hevc', hevc_grouped),
    ('dvc', DVC_uvg),
    ('rlvc', RLVC_uvg),
    ('plvc', plvc),
    ('ours', ours),
    ('diffvc_uvg', diffvc_uvg)
]


# Define datasets and their plotting styles
datasets = [
    {'data': h264_grouped, 'label': 'H.264', 'color': '#8B4513', 'marker': 'o'},
    {'data': hevc_grouped, 'label': 'HEVC', 'color': '#FAA502', 'marker': 's'},
    {'data': DVC_uvg, 'label': 'DVC', 'color': '#0D00FF', 'marker': '^'},
    {'data': RLVC_uvg, 'label': 'RLVC', 'color': '#1F8000', 'marker': 'D'},
     {'data': plvc, 'label': 'PLVC', 'color': '#800080', 'marker': '+'},
    {'data': ours, 'label': 'Ours', 'color': '#F71702', 'marker': '*'},
    {'data': diffvc_uvg, 'label': 'DiffVC', 'color': '#00CED1', 'marker': 'x'},
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
