import os
from pathlib import Path
import numpy as np
from collections import Counter
import struct
import flowiz as fz
import cv2
from multiprocessing import Pool, cpu_count

def load_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIEH':
            raise Exception('Invalid .flo file')
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
        flow = np.resize(data, (height, width, 2))
        return flow

def compute_block_vector(block, method='mean'):
    flat = block.reshape(-1, 2)
    if method == 'mean':
        return flat.mean(axis=0)
    elif method == 'median':
        return np.median(flat, axis=0)
    elif method == 'mode':
        quantized = (flat * 10).astype(int)
        tupled = [tuple(v) for v in quantized]
        most_common = Counter(tupled).most_common(1)[0][0]
        return np.array(most_common) / 10.0
    elif method == 'weighted':
        mag = np.linalg.norm(flat, axis=1)
        return (flat * mag[:, None]).sum(0) / (mag.sum() + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")

def reconstruct_flow(flow, grid_size, method):
    H, W, _ = flow.shape
    h_blocks = H // grid_size
    w_blocks = W // grid_size
    out = np.zeros((h_blocks, w_blocks, 2), dtype=np.float32)
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = flow[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            out[i, j] = compute_block_vector(block, method)
    return out

def write_flo_file(flow, filename):
    with open(filename, 'wb') as f:
        f.write(struct.pack('f', 202021.25))
        f.write(struct.pack('i', flow.shape[1]))
        f.write(struct.pack('i', flow.shape[0]))
        flow.astype(np.float32).tofile(f)

def flow_to_color(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255                    # Saturation = fixed
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def color_to_flow(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)

    # Convert back to angle and magnitude
    ang = h * 2 * np.pi / 180.0  # angle in radians
    mag = v / 255.0              # relative magnitude (will lose scale!)

    # Convert polar to cartesian
    fx = mag * np.cos(ang)
    fy = mag * np.sin(ang)

    return np.stack([fx, fy], axis=-1).astype(np.float32)

