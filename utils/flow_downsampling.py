"""
Regenerate optical-flow grids for all UVG videos.

Directory layout expected:
data/UVG/<video>/1080p/optical_flow_gop_<4|8|16>/*.flo   (source files)
└─ for each GOP folder, create/refresh
   grid_3/, grid_9/, grid_15/    (holds new .flo + .png)

Change BASE_DIR or GRID_SIZES below if needed.
"""

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import flowiz as fz     #  assume these utilities are available
from utils.flow_utils import (      #  your own helpers
    load_flo_file,
    reconstruct_flow,
    write_flo_file,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR   = Path("data/UVG")
GRID_SIZES = (15,9)       # tweak if you want fewer / more grids
MAX_WORKERS = 8               # cap workers so you don’t overload the node
METHOD      = "weighted"      # reconstruct_flow argument
# --------------------------------------------------------------------------- #


# ---------- helper --------------------------------------------------------- #
def source_flo_files(base_dir: Path):
    """
    Yield every original .flo file under GOP folders,
    but **skip** directories that already start with 'grid_'.
    """
    for root, dirs, files in os.walk(base_dir):
        # prune out grid_* so os.walk won't descend into them
        dirs[:] = [d for d in dirs if not d.startswith("grid_")]
        for fname in files:
            if fname.endswith(".flo"):
                yield Path(root) / fname


# ---------- processing ----------------------------------------------------- #
def process_one(args):
    """Worker target: generate grid-flow + PNG if missing."""
    src_path, grid = args
    out_dir   = src_path.parent / f"grid_{grid}"
    out_dir.mkdir(exist_ok=True)
    dst_flo   = out_dir / src_path.name
    dst_png   = dst_flo.with_suffix(".png")
    decoded_png = dst_png.with_name(dst_png.stem + ".decoded.png")
    
    if decoded_png.exists() or dst_png.exists():
        # already done
        return


    flow          = load_flo_file(src_path)
    processed     = reconstruct_flow(flow, grid_size=grid, method=METHOD)
    write_flo_file(processed, dst_flo)

    # convert .flo → RGB image and save
    rgb = fz.convert_from_file(str(dst_flo))
    cv2.imwrite(str(dst_png), rgb[..., ::-1])
    print(f"Saved {dst_flo.relative_to(BASE_DIR)}")


def main():
    tasks = [(p, g) for p in source_flo_files(BASE_DIR) for g in GRID_SIZES]
    print(f"⏳ Preparing {len(tasks)} tasks...")

    workers = min(MAX_WORKERS, cpu_count())
    with Pool(workers) as pool:
        pool.map(process_one, tasks)

    print("✅ Done.")


if __name__ == "__main__":
    main()
