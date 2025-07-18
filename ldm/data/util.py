import random

import numpy as np
import struct

def load_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIEH':
            raise Exception('Invalid .flo file')
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
        flow = np.resize(data, (2, height, width))
        return flow

def load_caption_dict(txt_path):
    caption_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # skip empty or malformed lines

            path, caption = line.split(':', 1)
            parts = path.strip().split('/')
            if len(parts) >= 3:
                parent1 = parts[-3].zfill(5)
                parent2 = parts[-2].zfill(4)
                key = f"{parent1}_{parent2}"
                caption_dict[key] = caption.strip()
    return caption_dict


def normalize_for_warping(flow):
    h, w = flow.shape[1:]
    flow[..., 1] /= (w / 2)
    flow[..., 2] /= (h / 2)
    return flow  # [2, H, W]

def adaptive_weighted_downsample(flow, target_h=128, target_w=128):
    """
    Args:
        flow: np.ndarray of shape (2, H, W)
    Returns:
        output: np.ndarray of shape (2, target_h, target_w)
    """
    C, H, W = flow.shape
    assert C == 2, "Flow must have shape (2, H, W)"

    output = np.zeros((2, target_h, target_w), dtype=np.float32)

    # Compute bounds of each block
    h_bounds = np.linspace(0, H, target_h + 1, dtype=int)
    w_bounds = np.linspace(0, W, target_w + 1, dtype=int)

    for i in range(target_h):
        for j in range(target_w):
            h_start, h_end = h_bounds[i], h_bounds[i + 1]
            w_start, w_end = w_bounds[j], w_bounds[j + 1]

            block = flow[:, h_start:h_end, w_start:w_end]  # shape: (2, h, w)
            flat = block.reshape(2, -1)  # shape: (2, N)

            mag = np.linalg.norm(flat, axis=0)  # shape: (N,)
            if mag.sum() > 0:
                weighted_avg = (flat * mag).sum(axis=1) / (mag.sum() + 1e-6)
            else:
                weighted_avg = flat.mean(axis=1)

            output[:, i, j] = weighted_avg

    return output  # shape: (2, target_h, target_w)



def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, annos = [], []
    for line in lines:
        id, txt = line.split('\t')
        file_ids.append(id)
        annos.append(txt)
    return file_ids, annos


def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros(condition.shape))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros(conditions[i].shape))
            else:
                results.append(conditions[i])
    return results