import numpy as np
import os
import zlib
import argparse
from PIL import Image
import torch
from cmp.utils.flowlib import read_flo_file
from cmp.utils.data_utils import flow_sampler
import flowiz as fz


def visualize_sparse_flow(flow, expansion_size=1):
    """
    Expands each non-zero flow vector into a (2 * expansion_size + 1)^2 block.
    """
    h, w, c = flow.shape
    expanded_flow = np.zeros_like(flow)

    for y in range(h):
        for x in range(w):
            if not np.all(flow[y, x] == 0):  # valid flow point
                for dy in range(-expansion_size, expansion_size + 1):
                    for dx in range(-expansion_size, expansion_size + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            expanded_flow[ny, nx] = flow[y, x]
    
    return expanded_flow


def quantize_and_compress(array: np.ndarray, scale: float = 10.0) -> bytes:
    """
    Quantize a float32 array to int16, then compress with zlib.
    """
    array_q = (array * scale).astype(np.uint8)
    return zlib.compress(array_q.tobytes())


def save_compressed_flow_mask(flow, mask, out_path_base):
    flow_compressed = quantize_and_compress(flow)
    mask_compressed = quantize_and_compress(mask.astype(np.float32))

    with open(out_path_base + "_flow.zlib", 'wb') as f:
        f.write(flow_compressed)

    with open(out_path_base + "_mask.zlib", 'wb') as f:
        f.write(mask_compressed)


def main(args):
    os.makedirs(args.output, exist_ok=True)

    flow_files = [f for f in os.listdir(args.input) if f.endswith('.flo')]
    flow_files.sort()

    for fname in flow_files:
        input_path = os.path.join(args.input, fname)
        flow = read_flo_file(input_path)

        # Generate sparse flow & mask
        flow_sparse, flow_mask = flow_sampler(flow, strategy=['grid', 'watershed'])

        # Save compressed versions
        basename = os.path.splitext(fname)[0]
        output_base = os.path.join(args.output, basename)
        save_compressed_flow_mask(flow_sparse, flow_mask, output_base)

        print(f"[✓] Compressed saved for {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse Flow Quantizer + Compressor")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder containing .flo files")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder for compressed files")
    args = parser.parse_args()

    main(args)
