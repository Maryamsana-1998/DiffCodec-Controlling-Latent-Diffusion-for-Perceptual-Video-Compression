import os
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from annotator.content import ContentDetector
from multiprocessing import Process, Queue, cpu_count

# === Collect all r2.png files in vimeo structure ===
def get_image_paths(root_dir):
    return sorted(list(Path(root_dir).glob("*/*/r2.png")))

# === Worker Function per GPU ===
def process_on_gpu(gpu_id, image_paths, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ContentDetector()

    for path in tqdm(image_paths, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            img = cv2.imread(str(path))
            if img is None:
                print(f"Failed to read: {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = model(img)
            np.save(path.parent / "r2.npy", feat)
        except Exception as e:
            print(f"Error: {e}, path: {path}")
    queue.put(f"GPU {gpu_id} done: {len(image_paths)} files.")

# === Main Launcher ===
def run_batched_content_extraction(root_dir, num_gpus=4):
    all_paths = get_image_paths(root_dir)
    chunks = [all_paths[i::num_gpus] for i in range(num_gpus)]
    processes, queues = [], []

    for i in range(num_gpus):
        q = Queue()
        p = Process(target=process_on_gpu, args=(i, chunks[i], q))
        p.start()
        processes.append(p)
        queues.append(q)

    for p in processes:
        p.join()

    for q in queues:
        print(q.get())

# === Call This ===
if __name__ == "__main__":
    run_batched_content_extraction("/data3/local_datasets/vimeo_sequences/", num_gpus=4)
