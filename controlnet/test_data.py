import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# import your dataset class
from dataset import UniDataset  # adjust path if needed

def safe_collate(batch):
    good = []
    for b in batch:
        try:
            good.append(b)
        except Exception as e:
            print("Bad batch item:", e, flush=True)
    return good


def test_dataset(start_batch=6000):
    dataset = UniDataset(
        anno_path='data/final_captions.txt',
        index_file='data/train_index.txt',
        local_type_list=['r1','r2','flow','flow_b'],
        resolution=512,
        transform=False   # skip heavy augmentations for faster check
    )

    print(f"Total samples: {len(dataset)}", flush=True)
    loader = DataLoader(dataset, batch_size=32, num_workers=8,
                        collate_fn=safe_collate)

    # wrap loader in enumerate with tqdm
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        if idx < start_batch:
            continue  # skip first N batches

        try:
            print(f"[Batch {idx}] OK", flush=True)
            # example: check shapes
            # jpg = batch[0]["jpg"]
            # cond = batch[0]["local_conditions"]
            # flow = batch[0]["flow"]
            # print(f"jpg {jpg.shape} | cond {cond.shape if hasattr(cond, 'shape') else type(cond)} | flow {flow.shape if hasattr(flow, 'shape') else type(flow)}")
        except Exception as e:
            print(f"❌ Error at batch {idx}: {e}", flush=True)
            continue


if __name__ == "__main__":
    print("Starting dataset test...", flush=True)
    test_dataset(start_batch=3000)
