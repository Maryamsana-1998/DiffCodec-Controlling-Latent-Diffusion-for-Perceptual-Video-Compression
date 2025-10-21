import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
import bjontegaard as bd

def bd_rate(R1, Q1, R2, Q2, higher_better=True):
    """
    Robust Bjøntegaard delta-rate computation.
    Handles few points, non-monotonic data, and avoids crazy results.
    Returns BD-rate percentage (Ours vs Baseline).
    """

    R1, Q1, R2, Q2 = map(np.array, (R1, Q1, R2, Q2))

    # If lower = better (LPIPS, FID) → flip sign so "higher=better"
    if not higher_better:
        Q1, Q2 = -Q1, -Q2

    # Sort by quality
    sort1, sort2 = np.argsort(Q1), np.argsort(Q2)
    Q1, R1 = Q1[sort1], R1[sort1]
    Q2, R2 = Q2[sort2], R2[sort2]

    # Clamp to overlapping quality range
    minQ = max(Q1.min(), Q2.min())
    maxQ = min(Q1.max(), Q2.max())
    if maxQ <= minQ:
        return np.nan  # no overlap → can't compute BD-rate

    # Take logs of rate
    logR1, logR2 = np.log(R1), np.log(R2)

    # Select interpolation type depending on number of points
    if len(Q1) >= 3:
        f1 = PchipInterpolator(Q1, logR1)
    else:
        f1 = interp1d(Q1, logR1, fill_value="extrapolate")
    if len(Q2) >= 3:
        f2 = PchipInterpolator(Q2, logR2)
    else:
        f2 = interp1d(Q2, logR2, fill_value="extrapolate")

    # Integrate over common quality range
    Qs = np.linspace(minQ, maxQ, 100)
    int1 = np.trapz(f1(Qs), Qs)
    int2 = np.trapz(f2(Qs), Qs)

    avg_diff = (int2 - int1) / (maxQ - minQ)
    return (np.exp(avg_diff) - 1) * 100

def bd_rate_safe(R1, Q1, R2, Q2, higher_better=True):
    R1, Q1, R2, Q2 = map(np.array, (R1, Q1, R2, Q2))

    if not higher_better:
        Q1, Q2 = -Q1, -Q2

    sort1, sort2 = np.argsort(Q1), np.argsort(Q2)
    Q1, R1 = Q1[sort1], R1[sort1]
    Q2, R2 = Q2[sort2], R2[sort2]

    minQ = min(Q1.min(), Q2.min()) * 0.95  # Extend lower bound
    maxQ = max(Q1.max(), Q2.max()) * 1.05  # Extend upper bound

    logR1, logR2 = np.log(R1), np.log(R2)

    if len(Q1) >= 3:
        f1 = PchipInterpolator(Q1, logR1, extrapolate=True)
    else:
        f1 = interp1d(Q1, logR1, fill_value="extrapolate")
    if len(Q2) >= 3:
        f2 = PchipInterpolator(Q2, logR2, extrapolate=True)
    else:
        f2 = interp1d(Q2, logR2, fill_value="extrapolate")

    Qs = np.linspace(minQ, maxQ, 100)
    int1 = np.trapz(f1(Qs), Qs)
    int2 = np.trapz(f2(Qs), Qs)

    avg_diff = (int2 - int1) / (maxQ - minQ)
    return (np.exp(avg_diff) - 1) * 100
# # -------------------------
# Input data
# -------------------------
ours_uvg_8 = pd.DataFrame({
    'bpp': [0.008151, 0.013398, 0.024487],
    'PSNR': [24.753654, 25.256767, 24.750165],
    'MS-SSIM': [0.858723, 0.870163, 0.851365],
    'LPIPS': [0.115910, 0.113702, 0.124080],
    'FID': [1.255306, 1.268365, 1.277536]
})

hevc_uvg_8 = pd.DataFrame({
    'bpp': [0.00733, 0.00935, 0.0387] ,
    'PSNR': [24.693132, 24.924867, 25.312276],
    'MS-SSIM': [0.861621, 0.873813, 0.891709],
    'LPIPS': [0.175702, 0.134636, 0.060229],
    'FID': [2.115607, 1.148821, 0.104496]
})

h264_uvg_8 = pd.DataFrame({
    'bpp': [0.00511, 0.00862, 0.0469] ,
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
    'MS-SSIM': [0.79202, 0.79862, 0.80845],
    'PSNR': [24.10989, 24.32727, 25.19724], 
    'LPIPS': [0.21517, 0.17804, 0.09064],
    'FID': [3.75336, 2.2539, 0.18493]
})

h264_uvg_4 = pd.DataFrame({
    'bpp': [0.00511, 0.00862, 0.0469] ,
    'PSNR': [24.36296, 24.84366, 25.6169],
    'MS-SSIM': [0.79214, 0.80378, 0.80968],
    'LPIPS': [0.16716, 0.12614, 0.07888],
    'FID': [1.25122, 0.63817, 0.05134]

})



# Calculate BD-Rate for all comparisons
metrics = [
    ('PSNR', True),
    ('MS-SSIM', True),
    ('LPIPS', False),
    ('FID', False)
]

# for comparison_name, ours_df, ref_df in comparisons:
#     print(f"\n{comparison_name}:")
#     for metric, higher_better in metrics:
#         bd_rate = bd_rate_safe(
#             ours_df['bpp'], ours_df[metric], 
#             ref_df['bpp'], ref_df[metric],
#             higher_better=higher_better
#         )
#         print(f"  {metric}: {bd_rate:.2f}%")
# # ours and hevc as pandas DataFrames (columns: 'bpp','PSNR','MS-SSIM','LPIPS','FID')
# Define comparisons
# Function to ensure monotonicity and calculate BD-Rate
def calculate_bd_rate(rate_anchor, dist_anchor, rate_test, dist_test, higher_better=True):
    # Sort by rate to ensure strictly increasing sequence
    sort_idx_anchor = np.argsort(rate_anchor)
    sort_idx_test = np.argsort(rate_test)
    rate_anchor, dist_anchor = rate_anchor[sort_idx_anchor], dist_anchor[sort_idx_anchor]
    rate_test, dist_test = rate_test[sort_idx_test], dist_test[sort_idx_test]

    # Remove duplicates if any
    _, unique_idx_anchor = np.unique(rate_anchor, return_index=True)
    _, unique_idx_test = np.unique(rate_test, return_index=True)
    rate_anchor, dist_anchor = rate_anchor[unique_idx_anchor], dist_anchor[unique_idx_anchor]
    rate_test, dist_test = rate_test[unique_idx_test], dist_test[unique_idx_test]

    # Invert distortion for lower-is-better metrics
    if not higher_better:
        dist_anchor = -dist_anchor
        dist_test = -dist_test

    try:
        # Calculate BD-Rate with PCHIP interpolation
        bd_value = bd_rate(rate_anchor, dist_anchor, rate_test, dist_test, higher_better=higher_better)
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

print("BD-Rate Calculations using bjontegaard library")
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
        
        # Calculate BD-Rate
        bd_value = calculate_bd_rate(rate_anchor, dist_anchor, rate_test, dist_test, higher_better)
        
        # Format output
        if np.isnan(bd_value):
            print(f"  {metric_name}: NaN")
        else:
            print(f"  {metric_name}: {bd_value:+.2f}%")

print("\n" + "=" * 60)
print("Note: Positive = Ours uses MORE bitrate | Negative = Ours uses LESS bitrate")
