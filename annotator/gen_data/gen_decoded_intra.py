import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import os
import re

# === CONFIGURATION ===
DATASET_ROOT      = Path("/data2/local_datasets/vimeo_sequences")
COMPRESSAI_PATH   = Path("../CompressAI/examples/codec.py")
MODEL             = "mbt2018-mean"
QUALITY           = 4
NUM_GPUS          = 4  # Set this to the number of GPUs available
NUM_PROCESSES     = 16 # Typically multiple of NUM_GPUS is okay

def natural_sort_key(filename):
    if filename == "r1.png":
        return -1
    elif filename == "r2.png":
        return 1e6
    match = re.match(r'im(\d+)\.png', filename)
    return int(match.group(1)) if match else 1e5

def compress_intra_frames(args):
    video_folder, gpu_id = args
    try:
        all_pngs = sorted(
            [f for f in os.listdir(video_folder) if f.endswith((".png", ".jpg"))],
            key=natural_sort_key
        )
        print(f"\n[ALL FRAMES] in {video_folder}:", all_pngs)

        all_images = [video_folder / Path(i) for i in all_pngs[::2]]
        print(f"[INTRA FRAMES] Selected for encoding in {video_folder}:", [f.name for f in all_images])

        if not all_images:
            return

        for img_path in all_images:
            stem = img_path.stem
            bin_path = img_path.parent / f"{stem}.bin"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Encode
            print(f'[GPU {gpu_id}] Encoding: ', bin_path)
            encode_cmd = [
                "python3", str(COMPRESSAI_PATH), "encode",
                str(img_path), "-o", str(bin_path),
                "--model", MODEL, "-q", str(QUALITY),
                "--cuda"
            ]
            subprocess.run(encode_cmd, check=True, env=env)

            # Decode
            dec_png = img_path
            decode_cmd = [
                "python3", str(COMPRESSAI_PATH), "decode",
                str(bin_path), "-o", str(dec_png),
                "--cuda"
            ]
            subprocess.run(decode_cmd, check=True, env=env)
            print(f'[GPU {gpu_id}] Decoding: ', dec_png)

            bin_path.unlink()

    except Exception as e:
        print(f"[GPU {gpu_id}] Failed: {video_folder} | {e}")

# === MAIN PARALLEL LOOP ===
if __name__ == "__main__":
    all_dirs = sorted([p for p in DATASET_ROOT.glob("*/*") if (p / "r1.png").exists()])
    tasks = [(video_dir, i % NUM_GPUS) for i, video_dir in enumerate(all_dirs[30:])]

    with Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(compress_intra_frames, tasks), total=len(tasks)))
