"""
Author: Maryam Sana
Description: Generates high-quality zoom comparison figures for multiple datasets.
             Each figure shows GT, H.264, H.265, and Ours with zoomed regions,
             and bolds best metric values (lower is better).
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib as mpl

# =========================================================
# === GLOBAL CONSTANTS & SETTINGS ===
# =========================================================
PDF_DPI = 600
FONT_SIZE_TITLE = 22
FONT_SIZE_LABEL = 15

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =========================================================
# === HELPER FUNCTIONS ===
# =========================================================
def crop(img, box):
    """Crop an image given (x, y, w, h)."""
    x, y, w, h = box
    return img[y:y + h, x:x + w]

def style_zoom(ax, img, color, title=""):
    """Show cropped zoom region with colored border and optional title."""
    ax.imshow(img)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), img.shape[1], img.shape[0],
                           edgecolor=color, linewidth=1.2, fill=False))
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_LABEL, fontweight='normal', pad=12, y=1.05)

def bold_best(metrics):
    """
    Given a list of (name, [bpp, lpips, fid]),
    returns formatted labels with bold lowest values (for lpips and fid).
    """
    # Convert metrics into numbers (skip GT)
    nums = [vals for _, vals in metrics[1:]]  # skip GT
    best_lpips = min(m[1] for m in nums)
    best_fid = min(m[2] for m in nums)
    best_bpp = min(m[0] for m in nums)

    labels = []
    for name, (bpp, lpips, fid) in metrics:
        def fmt(v, best): return f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
        if name == "GT":
            labels.append(f"{name} \n(BPP ↓ / LPIPS ↓ / FID ↓)")
        else:
            label = (f"{name} \n({fmt(bpp, best_bpp)} / "
                     f"{fmt(lpips, best_lpips)} / {fmt(fid, best_fid)})")
            labels.append(label)
    return labels


def make_comparison_figure(title, image_paths, boxes, metrics, save_path):
    """
    Create and save a 2x5 zoom comparison figure for a given sequence.
    Args:
        title: figure title (dataset name)
        image_paths: dict with keys 'gt', 'h264', 'h265', 'ours'
        boxes: list of 2 tuples (x, y, w, h)
        metrics: list of (name, [bpp, lpips, fid]) including GT
        save_path: PDF filename
    """
    # === Load and convert images ===
    print(title)
    imgs = {k: cv2.cvtColor(cv2.imread(v), cv2.COLOR_BGR2RGB) for k, v in image_paths.items()}

    # === Prepare crops ===
    def get_crops(box):
        return [crop(imgs['gt'], box),
                crop(imgs['h264'], box),
                crop(imgs['h265'], box),
                crop(imgs['ours'], box)]

    blue_box, red_box = boxes
    blue_crops = get_crops(blue_box)
    red_crops = get_crops(red_box)

    # === Figure layout ===
    fig = plt.figure(figsize=(26, 7), dpi=PDF_DPI)
    gs = fig.add_gridspec(2, 5, width_ratios=[3.5, 1, 1, 1, 1], height_ratios=[1, 1])
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    # === Main GT image ===
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(imgs['gt'])
    ax_main.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax_main.axis("off")

    # Draw zoom boxes
    for box, color in [(blue_box, 'blue'), (red_box, 'red')]:
        ax_main.add_patch(Rectangle((box[0], box[1]), box[2], box[3],
                                    edgecolor=color, linewidth=1.5, fill=False))

    # === Create axes ===
    ax_b = [fig.add_subplot(gs[0, i]) for i in range(1, 5)]
    ax_r = [fig.add_subplot(gs[1, i]) for i in range(1, 5)]

    # === Add metrics and crops ===
    labels = bold_best(metrics)
    colors = ['blue', 'blue', 'blue', 'blue']
    for ax, img, label, color in zip(ax_b, blue_crops, labels, colors):
        style_zoom(ax, img, color, label)
    for ax, img, color in zip(ax_r, red_crops, ['red']*4):
        style_zoom(ax, img, color)

    # === Connection lines ===
    for src_box, dest_ax, color in [(blue_box, ax_b[0], 'blue'),
                                    (red_box, ax_r[0], 'red')]:
        x, y, w, h = src_box
        xyA = (x + w / 2, y + h / 2)
        con = ConnectionPatch(xyA=xyA, xyB=(0, 0),
                              coordsA="data", coordsB="axes fraction",
                              axesA=ax_main, axesB=dest_ax,
                              color=color, linewidth=1.0, arrowstyle="-")
        ax_main.add_artist(con)

    # === Save ===
    plt.savefig(save_path, bbox_inches="tight", dpi=PDF_DPI)
    plt.close(fig)
    print(f"✅ Saved: {save_path}")

# =========================================================
# === EXAMPLES ===
# =========================================================
datasets = [
    {
        "title": "BQTerrace",
        "paths": {
            "gt": "bqterrace_original.png",
            "h264": "bq_terrace_0.01_gop8_h264.png",
            "h265": "bq_terrace_0.01_gop8_h265.png",
            "ours": "bqterrace_sparse_ours_gop8_bpp0.025.png"
        },
        "boxes": [(200, 550, 200, 200), (1000, 800, 200, 200)],
        "metrics": [
            ("GT", [0, 0, 0]),
            ("H.264", [0.008, 0.114, 0.99]),
            ("H.265", [0.017, 0.113, 1.55]),
            ("Ours", [0.025, 0.089, 1.65])
        ],
        "save": "bqterrace_comparison.pdf"
    },
    {
        "title": "MarketPlace",
        "paths": {
            "gt": "marketplace_original.png",
            "h264": "marketplace_gop8_bpp0.01_h264.png",
            "h265": "marketplace_gop8_bpp0.01_h265.png",
            "ours": "marketplace_gop8_ours_bpp0.017.png"
        },
        "boxes": [(1000, 50, 180, 180), (1500, 400, 180, 180)],
        "metrics": [
            ("GT", [0, 0, 0]),
            ("H.264", [0.010, 0.25, 2.32]),
            ("H.265", [0.011, 0.17, 1.65]),
            ("Ours", [0.017, 0.19, 0.94])
        ],
        "save": "marketplace_comparison.pdf"
    },
    {
        "title": "Beauty",
        "paths": {
            "gt": "beauty_original.png",
            "h264": "beauty_h264_0.01.png",
            "h265": "beauty_265.png",
            "ours": "beauty_ours_bpp_lpips.png"
        },
        "boxes": [(1000, 50, 200, 200), (700, 400, 200, 200)],
        "metrics": [
            ("GT", [0, 0, 0]),
            ("H.264", [0.009, 0.12, 0.537]),
            ("H.265", [0.008, 0.11, 0.69]),
            ("Ours", [0.0104, 0.0719, 0.621])
        ],
        "save": "beauty_comparison.pdf"
    },
    {
        "title": "YachtRide",
        "paths": {
            "gt": "yachtride_original.png",
            "h264": "yacht_h264_0.01.png",
            "h265": "yachetride_h265_bpp0.00929_lpips0.14.png",
            "ours": "yachetride_ours_bpp_0.014lpips_0.108.png"
        },
        "boxes": [(200, 50, 200, 200), (1300, 400, 200, 200)],
        "metrics": [
            ("GT", [0, 0, 0]),
            ("H.264", [0.010, 0.144, 1.52]),
            ("H.265", [0.0099, 0.140, 1.264]),
            ("Ours", [0.014, 0.058, 0.287])
        ],
        "save": "yacht_comparison.pdf"
    }
]

# === RUN ALL ===
for d in datasets:
    make_comparison_figure(d["title"], d["paths"], d["boxes"], d["metrics"], d["save"])
