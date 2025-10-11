import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

def bd_rate(bpp_A, q_A, bpp_B, q_B, metric_higher_is_better=True, use_logbpp=True):
    """
    Compute BD-Rate of A relative to B (percentage).
    bpp_*: arrays of bitrate (bpp)
    q_*: arrays of quality metric (higher is better for monotonic increase)
    metric_higher_is_better: if False (e.g. LPIPS, FID), caller should pass transformed quality where higher==better.
    """
    # Ensure numpy arrays and sorted by quality (increasing)
    bpp_A = np.array(bpp_A); q_A = np.array(q_A)
    bpp_B = np.array(bpp_B); q_B = np.array(q_B)

    # sort by quality
    sortA = np.argsort(q_A); sortB = np.argsort(q_B)
    qA, rA = q_A[sortA], bpp_A[sortA]
    qB, rB = q_B[sortB], bpp_B[sortB]

    # overlapping quality interval
    q_min = max(qA.min(), qB.min())
    q_max = min(qA.max(), qB.max())
    if q_min >= q_max:
        raise ValueError("No overlapping quality range for BD-Rate computation")

    # interpolation (log10 of bitrate by default)
    kind = 'cubic'
    try:
        if use_logbpp:
            fA = interp1d(qA, np.log10(rA), kind=kind, bounds_error=True)
            fB = interp1d(qB, np.log10(rB), kind=kind, bounds_error=True)
        else:
            fA = interp1d(qA, rA, kind=kind, bounds_error=True)
            fB = interp1d(qB, rB, kind=kind, bounds_error=True)
    except Exception:
        # fallback to linear if cubic fails / produces issues
        if use_logbpp:
            fA = interp1d(qA, np.log10(rA), kind='linear', fill_value="extrapolate")
            fB = interp1d(qB, np.log10(rB), kind='linear', fill_value="extrapolate")
        else:
            fA = interp1d(qA, rA, kind='linear', fill_value="extrapolate")
            fB = interp1d(qB, rB, kind='linear', fill_value="extrapolate")

    qs = np.linspace(q_min, q_max, 200)
    if use_logbpp:
        diff = fB(qs) - fA(qs)   # log10(R_B) - log10(R_A)
        mean_diff = simps(diff, qs) / (q_max - q_min)
        bd_rate_percent = (10 ** mean_diff - 1) * 100.0
    else:
        # alternative BD-PSNR style (not recommended here)
        valA = fA(qs); valB = fB(qs)
        mean_rel = (simps(valB - valA, qs) / (q_max - q_min))
        bd_rate_percent = mean_rel  # in metric units, not percent

    return bd_rate_percent

# helper to transform metrics where lower-is-better:
def transform_lower_is_better(metric_array, method='neg'):
    if method == 'neg':
        return -np.array(metric_array)
    elif method == 'inv':
        eps = 1e-6
        return 1.0 / (np.array(metric_array) + eps)
    else:
        raise ValueError("Unknown transform method")


import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

def bd_rate(bpp_A, q_A, bpp_B, q_B, metric_higher_is_better=True, use_logbpp=True):
    """
    Compute BD-Rate of A relative to B (percentage).
    bpp_*: arrays of bitrate (bpp)
    q_*: arrays of quality metric (higher is better for monotonic increase)
    metric_higher_is_better: if False (e.g. LPIPS, FID), caller should pass transformed quality where higher==better.
    """
    # Ensure numpy arrays and sorted by quality (increasing)
    bpp_A = np.array(bpp_A); q_A = np.array(q_A)
    bpp_B = np.array(bpp_B); q_B = np.array(q_B)

    # sort by quality
    sortA = np.argsort(q_A); sortB = np.argsort(q_B)
    qA, rA = q_A[sortA], bpp_A[sortA]
    qB, rB = q_B[sortB], bpp_B[sortB]

    # overlapping quality interval
    q_min = max(qA.min(), qB.min())
    q_max = min(qA.max(), qB.max())
    if q_min >= q_max:
        raise ValueError("No overlapping quality range for BD-Rate computation")

    # interpolation (log10 of bitrate by default)
    kind = 'cubic'
    try:
        if use_logbpp:
            fA = interp1d(qA, np.log10(rA), kind=kind, bounds_error=True)
            fB = interp1d(qB, np.log10(rB), kind=kind, bounds_error=True)
        else:
            fA = interp1d(qA, rA, kind=kind, bounds_error=True)
            fB = interp1d(qB, rB, kind=kind, bounds_error=True)
    except Exception:
        # fallback to linear if cubic fails / produces issues
        if use_logbpp:
            fA = interp1d(qA, np.log10(rA), kind='linear', fill_value="extrapolate")
            fB = interp1d(qB, np.log10(rB), kind='linear', fill_value="extrapolate")
        else:
            fA = interp1d(qA, rA, kind='linear', fill_value="extrapolate")
            fB = interp1d(qB, rB, kind='linear', fill_value="extrapolate")

    qs = np.linspace(q_min, q_max, 200)
    if use_logbpp:
        diff = fB(qs) - fA(qs)   # log10(R_B) - log10(R_A)
        mean_diff = simps(diff, qs) / (q_max - q_min)
        bd_rate_percent = (10 ** mean_diff - 1) * 100.0
    else:
        # alternative BD-PSNR style (not recommended here)
        valA = fA(qs); valB = fB(qs)
        mean_rel = (simps(valB - valA, qs) / (q_max - q_min))
        bd_rate_percent = mean_rel  # in metric units, not percent

    return bd_rate_percent

# helper to transform metrics where lower-is-better:
def transform_lower_is_better(metric_array, method='neg'):
    if method == 'neg':
        return -np.array(metric_array)
    elif method == 'inv':
        eps = 1e-6
        return 1.0 / (np.array(metric_array) + eps)
    else:
        raise ValueError("Unknown transform method")


# Example RD arrays for dataset/gop (user must fill actual numbers)
import pandas as pd
import numpy as np 
ours = pd.read_csv('benchmark_results/csv_for_latex/ours_data.csv')
hevc = pd.read_csv('benchmark_results/csv_for_latex/hevc_data.csv')
h264 = pd.read_csv('benchmark_results/csv_for_latex/h264_data.csv')
bpp_ours = ours['total_bpp'].values
msssim_ours = ours['MS-SSIM'].values
lpips_ours = ours['LPIPS'].values
fid_ours = ours['FID'].values

bpp_hevc = hevc['total_bpp'].values
msssim_hevc = hevc['MS-SSIM'].values
lpips_hevc = hevc['LPIPS'].values
fid_hevc = hevc['FID'].values

lpips_ours_inv = 1 - lpips_ours
lpips_hevc_inv = 1 - lpips_hevc

fid_ours_inv = 1 - np.array(fid_ours) / np.max([np.max(fid_ours), np.max(fid_hevc)])
fid_hevc_inv = 1 - np.array(fid_hevc) / np.max([np.max(fid_ours), np.max(fid_hevc)])

import numpy as np
from scipy import interpolate, integrate

def bd_rate(bpp1, metric1, bpp2, metric2):
    log_bpp1 = np.log(bpp1)
    log_bpp2 = np.log(bpp2)

    # Polynomial fit
    p1 = np.polyfit(metric1, log_bpp1, 3)
    p2 = np.polyfit(metric2, log_bpp2, 3)

    # Integration limits
    min_int = max(min(metric1), min(metric2))
    max_int = min(max(metric1), max(metric2))

    # Integrate polynomials
    int1 = np.polyint(p1)
    int2 = np.polyint(p2)

    avg1 = np.polyval(int1, max_int) - np.polyval(int1, min_int)
    avg2 = np.polyval(int2, max_int) - np.polyval(int2, min_int)
    avg_diff = (avg2 - avg1) / (max_int - min_int)

    return (np.exp(avg_diff) - 1) * 100



bd_ms = bd_rate(bpp_hevc, msssim_hevc, bpp_ours, msssim_ours)
bd_lpips = bd_rate(bpp_hevc, lpips_hevc_inv, bpp_ours, lpips_ours_inv)
bd_fid = bd_rate(bpp_hevc, fid_hevc_inv, bpp_ours, fid_ours_inv)

print("BD-Rate vs HEVC:")
print(f"  MS-SSIM: {bd_ms:.2f}%")
print(f"  LPIPS: {bd_lpips:.2f}%")
print(f"  FID: {bd_fid:.2f}%")
