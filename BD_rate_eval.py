import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

def bd_rate_safe(R1, Q1, R2, Q2, higher_better=True):
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

# -------------------------
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
comparisons = [
    ('Ours vs H.264 (GOP8)', ours_uvg_8, h264_uvg_8),
    ('Ours vs HEVC (GOP8)', ours_uvg_8, hevc_uvg_8),
    ('Ours vs H.264 (GOP4)', ours_uvg_4, h264_uvg_4),
    ('Ours vs HEVC (GOP4)', ours_uvg_4, hevc_uvg_4)
]

metrics = [
    ('PSNR', True),
    ('MS-SSIM', True),
    ('LPIPS', False),
    ('FID', False)
]

for comparison_name, ours_df, ref_df in comparisons:
    print(f"\n{comparison_name}:")
    for metric, higher_better in metrics:
        bd_rate = bd_rate_safe(
            ours_df['bpp'], ours_df[metric], 
            ref_df['bpp'], ref_df[metric],
            higher_better=higher_better
        )
        print(f"  {metric}: {bd_rate:.2f}%")