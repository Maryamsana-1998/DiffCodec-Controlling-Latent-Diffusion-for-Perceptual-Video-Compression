import os
import subprocess
from pathlib import Path
import numpy as np
import cv2
import json
from tqdm import tqdm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
vid ='YachtRide'
flow_dir          = Path(f"data/UVG/{vid}/optical_flow/optical_flow_gop_2_raft")         # input  .flo
png_dir           = Path(f"data/UVG/{vid}/optical_flow/tmp/png")                      # temp   HSV-PNG
bin_dir           = Path(f"data/UVG/{vid}/optical_flow/tmp/bin")                      # temp   .bin
decoded_png_dir   = Path(f"data/UVG/{vid}/optical_flow/tmp/decoded_png")              # temp   decoded PNG
decoded_flow_dir  = Path(f"data/UVG/{vid}/optical_flow/optical_flow_gop_2_raft_decoded")         # output .flo
report_path       = decoded_flow_dir / "compression_report.txt"

COMPRESSAI_PATH   = Path("../CompressAI/examples/codec.py")
MODEL             = "mbt2018-mean"
QUALITY           = 8
CUDA_FLAG         = "--cuda"          # ""  if you want CPU

for d in [png_dir, bin_dir, decoded_png_dir, decoded_flow_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Helper – basic .flo I/O
# ------------------------------------------------------------------
def read_flo(fname):
    with open(fname, 'rb') as f:
        magic = np.fromfile(f, np.float32, 1)[0]; assert magic == 202021.25
        w     = np.fromfile(f, np.int32,   1)[0]
        h     = np.fromfile(f, np.int32,   1)[0]
        data  = np.fromfile(f, np.float32, 2*w*h)
    return data.reshape(h, w, 2)

def write_flo(fname, flow):
    h, w, _ = flow.shape
    with open(fname, 'wb') as f:
        np.array([202021.25], np.float32).tofile(f)
        np.array([w],  np.int32).tofile(f)
        np.array([h],  np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)

# ------------------------------------------------------------------
# Flow  ↔  HSV-PNG
# ------------------------------------------------------------------
def flow_to_hsv(flow, max_flow=20.0, resize=None):
    """
    Convert optical flow to HSV visualization and optionally resize it.

    Args:
        flow (np.ndarray): Optical flow array of shape [H, W, 2].
        max_flow (float): Maximum magnitude to normalize saturation.
        resize (tuple or None): If provided, (width, height) for resizing output image.

    Returns:
        np.ndarray: RGB image of shape [H, W, 3] or resized.
    """
    u, v = flow[..., 0], flow[..., 1]
    mag  = np.sqrt(u**2 + v**2)
    ang  = np.arctan2(v, u)

    hsv        = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    hsv[...,0] = (ang + np.pi) / (2*np.pi)  # Hue: angle normalized to [0,1]
    hsv[...,1] = np.clip(mag / max_flow, 0, 1)  # Saturation: magnitude
    hsv[...,2] = 1.0  # Value channel

    hsv8 = (hsv * 255).astype(np.uint8)
    rgb  = cv2.cvtColor(hsv8, cv2.COLOR_HSV2RGB)

    if resize is not None:
        rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_LINEAR)

    return rgb


def hsv_png_to_flow(png_path, max_flow=20.0):
    bgr = cv2.imread(str(png_path))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)/255
    ang = hsv[...,0]*2*np.pi - np.pi
    mag = hsv[...,1]*max_flow
    u   = mag * np.cos(ang)
    v   = mag * np.sin(ang)
    return np.stack([u, v], axis=-1)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
with open(report_path, "w") as report:
    report.write(f"# Compression Report for model={MODEL}, quality={QUALITY}\n")

    for idx, flo_file in enumerate(tqdm(sorted(flow_dir.glob("*.flo")))):
        stem = flo_file.stem                         # e.g. flow_0001
        # 1. .flo → HSV-PNG
        flow = read_flo(flo_file)
        # flow = adaptive_weighted_downsample(flow)
        png_path = png_dir / f"{stem}.png"
        cv2.imwrite(str(png_path), flow_to_hsv(flow,resize=(512,512)))

        # 2. Compress PNG → .bin
        bin_path = bin_dir / f"{stem}.bin"
        encode_cmd = [
            "python3", str(COMPRESSAI_PATH), "encode",
            str(png_path), "-o", str(bin_path),
            "--model", MODEL, "-q", str(QUALITY)
        ]
        if CUDA_FLAG: encode_cmd.append(CUDA_FLAG)
        subprocess.run(encode_cmd, check=True)

        # 3. Decompress back to PNG
        dec_png = decoded_png_dir / f"{stem}.png"
        decode_cmd = [
            "python3", str(COMPRESSAI_PATH), "decode",
            str(bin_path), "-o", str(dec_png)
        ]
        if CUDA_FLAG: decode_cmd.append(CUDA_FLAG)
        subprocess.run(decode_cmd, check=True)

        # 4. PNG → .flo (decoded)
        rec_flow = hsv_png_to_flow(dec_png)
        write_flo(decoded_flow_dir / f"{stem}.flo", rec_flow)

        # 5. Log size
        size_kb = os.path.getsize(bin_path) / 1024
        report.write(f"- Frame: {stem}.flo → {size_kb:.2f} KB\n")
        print(f"Processed {stem} ({size_kb:.2f} KB)")

print(f"\n✅ Finished. Report saved to {report_path}")
