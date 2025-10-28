import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Configuration ===
GOP_SIZE = 4  # Configurable GOP size (e.g., 4, 8)
DATASET = "class_b"  # Configurable dataset (e.g., "uvg", "class_b")

# Global font style
plt.rcParams['text.usetex'] = True

# Function to load and process JSON data
def load_and_process_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    flattened_data = []
    for video_name, records in data.items():
        if records:
            for record in records:
                record['video_name'] = video_name
                flattened_data.append(record)
    df = pd.DataFrame(flattened_data)
    return df.groupby('bpp_folder')[['inter_bpp', 'LPIPS', 'MS-SSIM', 'PSNR', 'FID']].mean().sort_values('inter_bpp')

# Load JSON data based on GOP size and dataset
base_path = f'benchmark_results/gop{GOP_SIZE}_results'
h264_grouped = load_and_process_json(f'{base_path}/h264_{DATASET}_gop{GOP_SIZE}/inter_results.json')
hevc_grouped = load_and_process_json(f'{base_path}/hevc_{DATASET}_gop{GOP_SIZE}/inter_results.json')

# BPP values for Ours (interpolated) based on dataset and GOP size
inter_bpp_uvg = {
 2: {'dense': 0.009334509112286891,
     'none': 0.0,
     'sparse': 0.002998393196309863},
 4: {'dense': 0.014001763668430336,
     'none': 0.0,
     'sparse': 0.004497589794464794},
 8: {'dense': 0.01633539094650206, 'none': 0.0, 'sparse': 0.00524718809354226}}

inter_bpp_class_b = {
2: {'dense': 0.007862712566042745,
     'none': 0.0,
     'sparse': 0.0032672661443494773},
 4: {'dense': 0.011794068849064119,
     'none': 0.0,
     'sparse': 0.004900899216524217},
 8: {'dense': 0.013759746990574803,
     'none': 0.0,
     'sparse': 0.005717715752611587}
     }

inter_bpp = inter_bpp_uvg if DATASET == "uvg" else inter_bpp_class_b
ours_bpp = inter_bpp[GOP_SIZE]

# Load Ours data
ours = pd.read_csv(f'benchmark_results/gop{GOP_SIZE}_results/ours_{DATASET}_inter.csv')

# Ensure all datasets have the same columns
for df in [h264_grouped, hevc_grouped, ours]:
    if 'inter_bpp' not in df.columns:
        df['inter_bpp'] = df.index  # Use index as inter_bpp if not present
    df = df.sort_values('inter_bpp').reset_index(drop=True)

# Define datasets with plotting styles
datasets = [
    {'data': h264_grouped, 'label': 'H.264', 'color': '#BF0606', 'marker': 'o'},
    {'data': hevc_grouped, 'label': 'HEVC', 'color': '#CC6704', 'marker': 's'},
    {'data': ours, 'label': 'Ours', 'color': '#AD34BA', 'marker': '*'}

]

# Metrics and their display info
metrics = ['PSNR', 'MS-SSIM', 'LPIPS', 'FID']
titles = {
    'PSNR': f'PSNR vs Bitrate (GOP {GOP_SIZE}, {DATASET})',
    'MS-SSIM': f'MS-SSIM vs Bitrate (GOP {GOP_SIZE}, {DATASET})',
    'LPIPS': f'LPIPS vs Bitrate (GOP {GOP_SIZE}, {DATASET})',
    'FID': f'FID vs Bitrate (GOP {GOP_SIZE}, {DATASET})'
}
y_labels = {
    'PSNR': 'PSNR (dB) ↑',
    'MS-SSIM': 'MS-SSIM ↑',
    'LPIPS': 'LPIPS ↓',
    'FID': 'FID ↓'
}

# Generate and save each plot individually
for metric in metrics:
    fig, ax = plt.subplots(figsize=(6, 5))  # Single figure per metric
    
    for dataset in datasets:
        ax.plot(
            dataset['data']['inter_bpp'],
            dataset['data'][metric],
            marker=dataset['marker'],
            label=dataset['label'],
            color=dataset['color'],
            linewidth=2,
            markersize=7
        )
    
    # Customize appearance
    ax.set_title(titles[metric], fontsize=13, fontweight='bold')
    ax.set_xlabel('Bitrate (bpp)', fontsize=11)
    ax.set_ylabel(y_labels[metric], fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9, loc='best', frameon=True)
    
    plt.tight_layout()
    
    # Save individual file
    save_path = f'benchmark_results/inter_plots/{DATASET}_gop{GOP_SIZE}_{metric.lower()}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', format='pdf')
    print(f"Saved {save_path}")
    
    plt.close(fig)  # Close to save memory

