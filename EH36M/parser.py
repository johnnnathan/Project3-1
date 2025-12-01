import os
import torch
import multiprocessing as mp
import numpy as np
from loader import load_sample
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CACHE_DIR = "cache_eh36m"
os.makedirs(CACHE_DIR, exist_ok=True)

ALLOWED_ACTIONS = [
    "Directions", "Discussion", "Greeting", "Phoning",
    "Posing", "Sitting", "SittingDown",
    "Waiting", "Walking", "WalkTogether"
]


def process_folder(folder):
    import torch
    import gc
    folder_name = os.path.basename(folder)
    save_base = os.path.join(CACHE_DIR, folder_name)

    if os.path.exists(save_base + "_0.pt"):
        return f"Already cached: {folder_name}"

    sample = load_sample(folder)
    chunks = []
    chunk_idx = 0
    batch_size = 200

    for skel_frame, event_slice in zip(sample['skeleton'], sample['events_aligned']):
        frame_data = {
            'skeleton': torch.tensor(skel_frame, dtype=torch.float32),
            'events': torch.tensor(
                np.vstack([event_slice['ts'], event_slice['x'], event_slice['y'], event_slice['pol']]).T,
                dtype=torch.float32
            )
        }
        chunks.append(frame_data)

        if len(chunks) >= batch_size:
            torch.save(chunks, f"{save_base}_{chunk_idx}.pt")
            chunks.clear()
            chunk_idx += 1
            gc.collect()

    if chunks:
        torch.save(chunks, f"{save_base}_{chunk_idx}.pt")
        gc.collect()

    return f"Cached safely: {folder_name} in {chunk_idx+1} chunks"


def cache_samples_parallel(root):
    all_folders = []

    for subdir, _, _ in os.walk(root):
        if "ch0GT50Hzskeleton" in subdir:
            all_folders.append(os.path.dirname(subdir))

    print(f"\nProcessing {len(all_folders)} folders...\n")

    with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
        for res in pool.imap(process_folder, all_folders, chunksize=1):
            print(res)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    root = r"C:\Users\alenm\OneDrive\Desktop\EH36M"
    cache_samples_parallel(root)
    print("\nCaching complete.")
