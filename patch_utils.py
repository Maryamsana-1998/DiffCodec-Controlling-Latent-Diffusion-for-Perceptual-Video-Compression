import os
import cv2
import numpy as np
from PIL import Image
from test_utils import *
import argparse
import json
import warnings

import torch
import torch.nn.functional as F

def merge_costiles(tiles, coords, full_shape, order="hwc", feather=64):
    """
    Reconstruct image from overlapping tiles with cosine feather blending.
    tiles: list of np.ndarray tiles
    coords: list of (y1, y2, x1, x2) tile coordinates
    full_shape: (h, w) if order="hwc" else (c, h, w)
    order: "hwc" or "chw"
    feather: feathering (pixels) at edges
    """
    if order == "hwc":
        h, w = full_shape
        c = tiles[0].shape[2]
        out = np.zeros((h, w, c), np.float32)
        weight = np.zeros((h, w, c), np.float32)
    else:  # chw
        c = tiles[0].shape[0]
        h, w = full_shape
        out = np.zeros((c, h, w), np.float32)
        weight = np.zeros((c, h, w), np.float32)

    def make_cosine_mask(h, w, feather):
        # 1D cosine window
        def cosine_window(L):
            x = np.linspace(-np.pi, np.pi, L)
            return (np.cos(x) + 1) / 2

        wy = np.ones(h)
        wx = np.ones(w)

        if feather > 0:
            f = min(feather, h//2)
            wy[:f] = cosine_window(f)[:f]
            wy[-f:] = cosine_window(f)[-f:]
            f = min(feather, w//2)
            wx[:f] = cosine_window(f)[:f]
            wx[-f:] = cosine_window(f)[-f:]

        mask = np.outer(wy, wx)
        return mask.astype(np.float32)

    for tile, (y1, y2, x1, x2) in zip(tiles, coords):
        target_h, target_w = (y2 - y1), (x2 - x1)

        # resize tile if needed
        if order == "hwc":
            if tile.shape[0] != target_h or tile.shape[1] != target_w:
                tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            mask = make_cosine_mask(target_h, target_w, feather)
            mask = np.repeat(mask[:, :, None], c, axis=2)

            out[y1:y2, x1:x2, :] += tile.astype(np.float32) * mask
            weight[y1:y2, x1:x2, :] += mask

        else:  # chw
            if tile.shape[1] != target_h or tile.shape[2] != target_w:
                tile = np.stack([
                    cv2.resize(tile[ch], (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    for ch in range(c)
                ])
            mask = make_cosine_mask(target_h, target_w, feather)
            mask = np.repeat(mask[None, :, :], c, axis=0)

            out[:, y1:y2, x1:x2] += tile.astype(np.float32) * mask
            weight[:, y1:y2, x1:x2] += mask

    out /= np.maximum(weight, 1e-8)
    return out.astype(np.uint8)


def merge_latent_tiles_from_pixel_coords(
    latents,               # list of torch tensors, each shape (1, C, th, tw)
    pixel_coords,          # list of (y1, y2, x1, x2) in original pixel space
    full_latent_shape,     # (1, C, H_lat, W_lat) target merged latent shape
    original_image_size,   # (H_px, W_px) e.g. (1080, 1920)
    eps: float = 1e-8
):
    """
    Merge a list of latent tiles (torch tensors) which were created
    from pixel-space crops given by `pixel_coords`. Returns merged latent
    tensor of shape full_latent_shape.

    The function:
      - maps pixel coords -> latent coords using full image -> full latent ratio
      - resizes tile latents if necessary to exactly match the target latent region
      - uses a Hann-window 2D mask per tile for smooth blending
      - accumulates weighted sums and divides by weights

    Notes:
      - latents must be torch tensors on same device & dtype.
      - pixel_coords length must equal len(latents).
    """
    assert len(latents) == len(pixel_coords), "latents and coords length mismatch"

    device = latents[0].device
    dtype  = latents[0].dtype

    _, C, H_lat, W_lat = full_latent_shape
    H_px, W_px = original_image_size

    out = torch.zeros(full_latent_shape, device=device, dtype=dtype)
    weight = torch.zeros_like(out)

    # helper to make 2D hann window (safe for tiny sizes)
    def make_2d_hann(h, w, device, dtype):
        if h <= 1:
            wy = torch.ones(1, device=device, dtype=dtype)
        else:
            wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
        if w <= 1:
            wx = torch.ones(1, device=device, dtype=dtype)
        else:
            wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
        m = wy.unsqueeze(1) * wx.unsqueeze(0)
        # normalize so max==1
        m = m / (m.max() + 1e-12)
        return m  # shape (h,w)

    for tile, (x1_px, x2_px, y1_px, y2_px) in zip(latents, pixel_coords):
        ly1 = int(round(y1_px * (H_lat / float(H_px))))
        ly2 = int(round(y2_px * (H_lat / float(H_px))))
        lx1 = int(round(x1_px * (W_lat / float(W_px))))
        lx2 = int(round(x2_px * (W_lat / float(W_px))))


        # clamp to canvas bounds
        ly1 = max(0, min(ly1, H_lat))
        ly2 = max(0, min(ly2, H_lat))
        lx1 = max(0, min(lx1, W_lat))
        lx2 = max(0, min(lx2, W_lat))

        target_h = ly2 - ly1
        target_w = lx2 - lx1

        if target_h <= 0 or target_w <= 0:
            # tile maps outside canvas (can happen with rounding) -> skip
            continue

        # tile shape and batch handling
        # ensure tile is (1, C, th, tw)
        assert tile.dim() == 4 and tile.size(0) == 1, "expected tile shape (1,C,H,W)"
        th, tw = tile.shape[-2:]

        # if tile size doesn't match target region, resize the tile latent
        if (th != target_h) or (tw != target_w):
            # use bilinear interpolation on latents (aligned=False)
            tile_resized = F.interpolate(tile, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            tile_resized = tile

        # make 2D blending mask sized (target_h, target_w)
        mask2d = make_2d_hann(target_h, target_w, device=device, dtype=dtype)  # (h,w)
        mask = mask2d.unsqueeze(0).unsqueeze(0)             # (1,1,h,w)
        mask = mask.expand(1, tile_resized.size(1), target_h, target_w)  # (1,C,h,w)

        out[:,:, ly1:ly2, lx1:lx2] += tile_resized * mask
        weight[:,:, ly1:ly2, lx1:lx2] += mask

    # final normalization
    denom = torch.maximum(weight, torch.tensor(eps, device=device, dtype=dtype))
    merged = out / denom
    return merged


def resize_to_match(img, target_shape, order="hwc"):
    """Resize img to match target_shape."""
    if order == "hwc":
        return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    elif order == "chw":
        c, h, w = img.shape
        tgt_h, tgt_w = target_shape
        resized = cv2.resize(img.transpose(1,2,0), (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        return resized.transpose(2,0,1)
    else:
        raise ValueError("order must be 'hwc' or 'chw'")

def crop_into_tiles(img, tile_size, overlap=0, order="hwc"):
    """Crop image into overlapping tiles of size tile_size."""
    if order == "hwc":
        h, w, c = img.shape
    else: # chw
        c, h, w = img.shape

    stride_y = tile_size[0] - overlap
    stride_x = tile_size[1] - overlap

    tiles, coords = [], []
    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            y2, x2 = min(y+tile_size[0], h), min(x+tile_size[1], w)
            if order == "hwc":
                tile = img[y:y2, x:x2, :]
            else:
                tile = img[:, y:y2, x:x2]
            tiles.append(tile)
            coords.append((y, y2, x, x2))
    return tiles, coords, (h, w)


def merge_tiles(tiles, coords, full_shape, order="hwc"):
    """Reconstruct image from overlapping tiles with feather blending."""
    if order == "hwc":
        h, w = full_shape
        c = tiles[0].shape[2]
        out = np.zeros((h, w, c), np.float32)
        weight = np.zeros((h, w, c), np.float32)
    else:  # chw
        c = tiles[0].shape[0]
        h, w = full_shape
        out = np.zeros((c, h, w), np.float32)
        weight = np.zeros((c, h, w), np.float32)

    for tile, (y1, y2, x1, x2) in zip(tiles, coords):
        # Ensure tile matches the target region
        target_h, target_w = (y2 - y1), (x2 - x1)

        if order == "hwc":
            if tile.shape[0] != target_h or tile.shape[1] != target_w:
                tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            mask = np.ones_like(tile, np.float32)
            out[y1:y2, x1:x2, :] += tile.astype(np.float32) * mask
            weight[y1:y2, x1:x2, :] += mask

        else:  # chw
            if tile.shape[1] != target_h or tile.shape[2] != target_w:
                tile = np.stack([
                    cv2.resize(tile[c], (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    for c in range(tile.shape[0])
                ])
            mask = np.ones_like(tile, np.float32)
            out[:, y1:y2, x1:x2] += tile.astype(np.float32) * mask
            weight[:, y1:y2, x1:x2] += mask

    out /= np.maximum(weight, 1e-8)
    return out.astype(np.uint8)
