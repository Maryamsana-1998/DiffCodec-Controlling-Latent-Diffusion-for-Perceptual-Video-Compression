"""
Author: Maryam Sana
Description: Generates a 4-panel visual comparison figure
             (1 full frame + 3 zoomed crops) with bolded better stats.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Rectangle

# =========================================================
# === GLOBAL SETTINGS ===
# =========================================================
PDF_DPI = 600
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
def crop(img, x_range, y_range):
    """Crop image given x and y pixel ranges."""
    return img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

def bold_best(entries):
    """
    entries: list of (name, bpp, msssim)
    Returns LaTeX-formatted strings with bold best values.
    MS-SSIM: higher is better
    Bpp: lower is better
    """
    bpps = [e[1] for e in entries]
    ssims = [e[2] for e in entries]
    best_bpp = min(bpps)
    best_ssim = max(ssims)

    def fmt(val, best, better_high=False):
        return f"\\textbf{{{val:.3f}}}" if ((better_high and val == best) or (not better_high and val == best)) else f"{val:.3f}"

    captions = []
    for name, bpp, ssim in entries:
        captions.append(f"{name} ({fmt(bpp, best_bpp)} / {fmt(ssim, best_ssim, better_high=True)})")
    return captions

# =========================================================
# === MAIN FUNCTION ===
# =========================================================
def make_visual_comparison(
    img_paths,
    crop_x=(400, 700),
    crop_y=(320, 520),
    stats=[("H.264", 0.009, 0.8964), ("H.265", 0.008, 0.898), ("Ours", 0.010, 0.940)],
    save_name="visual_quality_comparison.pdf",
    title="Bosphorus"
):
    """Generates a 2x2 visual comparison figure with bolded metrics."""
    # === Load images ===
    imgs = [mpimg.imread(p) for p in img_paths]

    # === Crop regions ===
    crops = [crop(imgs[i][200:800, 200:1100], crop_x, crop_y) for i in range(1, 4)]

    # === Compute bolded captions ===
    captions = bold_best(stats)
    labels = [
        f"(a) Original frame (Bpp / MS-SSIM)",
        f"(b) {captions[0]}",
        f"(c) {captions[1]}",
        f"(d) {captions[2]}"
    ]

    # === Plot setup ===
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=PDF_DPI)
    axes = axes.flatten()

    # --- Original frame ---
    axes[0].imshow(imgs[0][200:800, 200:1100])
    axes[0].axis("off")
    axes[0].set_title(labels[0], fontsize=14, y=-0.15)

    # Add red rectangle showing cropped region
    rect = Rectangle(
        (crop_x[0], crop_y[0]),
        crop_x[1] - crop_x[0],
        crop_y[1] - crop_y[0],
        linewidth=2, edgecolor="r", facecolor="none"
    )
    axes[0].add_patch(rect)

    # --- Cropped views ---
    for ax, crop_img, label in zip(axes[1:], crops, labels[1:]):
        ax.imshow(crop_img)
        ax.axis("off")
        ax.set_title(label, fontsize=14, y=-0.15)

    # --- Layout adjustments ---
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.18)
    fig.suptitle(title, fontsize=18, y=1.02, fontweight="bold")

    # --- Save ---
    plt.savefig(save_name, bbox_inches="tight", dpi=PDF_DPI, format="pdf")
    plt.close(fig)
    print(f"✅ Saved: {save_name}")

# =========================================================
# === EXAMPLE USAGE ===
# =========================================================
make_visual_comparison(
    img_paths=[
        "bosphorus_original.png",
        "bosphorus_h264.png",
        "bosphorus_hevc_gop8.png",
        "bosphorus_sparse_gop8_ours.png"
    ],
    crop_x=(400, 700),
    crop_y=(320, 520),
    stats=[
        ("H.264", 0.009, 0.8964),
        ("H.265", 0.008, 0.898),
        ("Ours", 0.010, 0.940)
    ],
    save_name="bosphorus_comparison.pdf",
    title="Bosphorus"
)
