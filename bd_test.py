import pandas as pd
import numpy as np
from bjontegaard import bd_rate
from scipy.interpolate import interp1d

# Input data (your exact datasets)
ours_uvg_8 = pd.DataFrame({
    'bpp': [0.008151, 0.013398, 0.024487],
    'PSNR': [24.753654, 25.256767, 24.750165],
    'MS-SSIM': [0.858723, 0.870163, 0.851365],
    'LPIPS': [0.115910, 0.113702, 0.124080],
    'FID': [1.255306, 1.268365, 1.277536]
})

hevc_uvg_8 = pd.DataFrame({
    'bpp': [0.00733, 0.00935, 0.0387],
    'PSNR': [24.693132, 24.924867, 25.312276],
    'MS-SSIM': [0.861621, 0.873813, 0.891709],
    'LPIPS': [0.175702, 0.134636, 0.060229],
    'FID': [2.115607, 1.148821, 0.104496]
})

h264_uvg_8 = pd.DataFrame({
    'bpp': [0.00511, 0.00862, 0.0469],
    'PSNR': [24.36296, 24.84366, 25.6169],
    'MS-SSIM': [0.79214, 0.80378, 0.80968],
    'LPIPS': [0.16716, 0.12614, 0.07888],
    'FID': [1.25122, 0.63817, 0.05134]
})

ours_uvg_4 = pd.DataFrame({
    'bpp': [0.0163, 0.0208, 0.0303],
    'PSNR': [26.00568, 27.3825, 25.95249],
    'MS-SSIM': [0.90668, 0.92182, 0.90234],
    'LPIPS': [0.12945, 0.12173, 0.1365],
    'FID': [2.23526, 2.13443, 2.2206]
})

hevc_uvg_4 = pd.DataFrame({
    'bpp': [0.00873, 0.01087, 0.0387],
    'PSNR': [24.10989, 24.32727, 25.19724],
    'MS-SSIM': [0.79202, 0.79862, 0.80845],
    'LPIPS': [0.21517, 0.17804, 0.09064],
    'FID': [3.75336, 2.2539, 0.18493]
})

h264_uvg_4 = pd.DataFrame({
    'bpp': [0.00511, 0.00862, 0.0469],  # Identical to h264_uvg_8, should differ
    'PSNR': [24.36296, 24.84366, 25.6169],
    'MS-SSIM': [0.79214, 0.80378, 0.80968],
    'LPIPS': [0.16716, 0.12614, 0.07888],
    'FID': [1.25122, 0.63817, 0.05134]
})

# Function to extrapolate RD curves with validation
def extrapolate_rd_curve(bpp, quality, n_points=7, extend_factor=0.1):
    # Sort by bpp
    sort_idx = np.argsort(bpp)
    bpp, quality = bpp[sort_idx], quality[sort_idx]
    
    # Create interpolation function
    interp_func = interp1d(bpp, quality, kind='linear', fill_value='extrapolate')
    
    # Define extended range, ensuring no negative bpp
    min_bpp = max(bpp.min() * (1 - extend_factor), 0.001)  # Minimum 0.001 to avoid log(0)
    max_bpp = bpp.max() * (1 + extend_factor)
    new_bpp = np.linspace(min_bpp, max_bpp, n_points)
    
    # Extrapolate quality
    new_quality = interp_func(new_bpp)
    
    # Ensure monotonicity (force increasing for PSNR/MS-SSIM, decreasing for LPIPS/FID)
    if np.all(np.diff(quality) > 0):  # Increasing (e.g., PSNR, MS-SSIM)
        new_quality = np.maximum.accumulate(new_quality)
    elif np.all(np.diff(quality) < 0):  # Decreasing (e.g., LPIPS, FID after inversion)
        new_quality = np.minimum.accumulate(new_quality[::-1])[::-1]
    
    return new_bpp, new_quality

# Function to calculate BD-Rate with extrapolated data
def calculate_bd_rate_with_extrapolation(rate_anchor, dist_anchor, rate_test, dist_test, higher_better=True):
    # Extrapolate both curves
    rate_anchor_ext, dist_anchor_ext = extrapolate_rd_curve(rate_anchor, dist_anchor)
    rate_test_ext, dist_test_ext = extrapolate_rd_curve(rate_test, dist_test)
    
    # Validate data
    if not (np.all(np.diff(rate_anchor_ext) > 0) and np.all(np.diff(rate_test_ext) > 0)):
        print("Warning: Extrapolated rates are not strictly increasing")
        return np.nan
    if np.any(rate_anchor_ext <= 0) or np.any(rate_test_ext <= 0):
        print("Warning: Extrapolated rates include non-positive values")
        return np.nan
    
    # Invert distortion for lower-is-better metrics
    if not higher_better:
        dist_anchor_ext = -dist_anchor_ext
        dist_test_ext = -dist_test_ext
    
    try:
        # Calculate BD-Rate with extrapolated data
        bd_value = bd_rate(rate_anchor_ext, dist_anchor_ext, rate_test_ext, dist_test_ext, method='pchip', min_overlap=0)
        if np.isinf(bd_value) or np.abs(bd_value) > 1000:  # Arbitrary threshold for sanity
            print(f"Warning: BD-Rate {bd_value} is unstable, returning NaN")
            return np.nan
        return bd_value
    except Exception as e:
        print(f"Error: {e}")
        return np.nan

# Define comparisons
comparisons = [
    ('Ours vs H.264 (GOP8)', ours_uvg_8, h264_uvg_8),
    ('Ours vs HEVC (GOP8)', ours_uvg_8, hevc_uvg_8),
    ('Ours vs H.264 (GOP4)', ours_uvg_4, h264_uvg_4),
    ('Ours vs HEVC (GOP4)', ours_uvg_4, hevc_uvg_4)
]

# Metrics configuration: (name, higher_better)
metrics_config = [
    ('PSNR', True),
    ('MS-SSIM', True),
    ('LPIPS', False),
    ('FID', False)
]

print("BD-Rate Calculations using bjontegaard library with Extrapolated Curves")
print("=" * 60)

for comparison_name, ours_df, ref_df in comparisons:
    print(f"\n{comparison_name}:")
    print("-" * 40)
    
    for metric_name, higher_better in metrics_config:
        # Extract rate and distortion
        rate_anchor = ref_df['bpp'].values
        dist_anchor = ref_df[metric_name].values
        rate_test = ours_df['bpp'].values
        dist_test = ours_df[metric_name].values
        
        # Calculate BD-Rate with extrapolation
        bd_value = calculate_bd_rate_with_extrapolation(rate_anchor, dist_anchor, rate_test, dist_test, higher_better)
        
        # Format output
        if np.isnan(bd_value):
            print(f"  {metric_name}: NaN")
        else:
            print(f"  {metric_name}: {bd_value:+.2f}%")

print("\n" + "=" * 60)
print("Note: Positive = Ours uses MORE bitrate | Negative = Ours uses LESS bitrate")
print("Warning: Results are based on extrapolated data and may be unstable due to limited overlap.")