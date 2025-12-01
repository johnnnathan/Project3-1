import os
import numpy as np
import h5py
import re
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import imageio
from io import BytesIO


def load_skeleton(skel_file):
    pattern = re.compile(r'\d+ (\d+\.\d+) SKLT \((.*?)\)')
    skeleton_frames = []
    with open(skel_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            _, points_str = match.groups()
            points = np.array(points_str.split(), dtype=np.float32).reshape(-1, 2)
            skeleton_frames.append(points)
    return np.array(skeleton_frames, dtype=np.float32)


def load_events_lazy(h5_file):
    """Lazy loading of events — does NOT load full array into RAM."""
    f = h5py.File(h5_file, 'r')
    events = f['events']
    return {
        'ts': events[:, 0],
        'x': events[:, 1],
        'y': events[:, 2],
        'pol': events[:, 3].astype(bool)
    }


def align_events_lazy(events, skeleton, window=0.02):
    """Lazily yield event slices for each frame, low memory."""
    total_time = events['ts'][-1]
    frame_times = np.linspace(0, total_time, len(skeleton))

    for t in frame_times:
        mask = (events['ts'][:] >= t - window) & (events['ts'][:] <= t + window)
        yield {
            'ts': events['ts'][mask],
            'x': events['x'][mask],
            'y': events['y'][mask],
            'pol': events['pol'][mask]
        }


def load_sample(folder_path):
    skel_dir = os.path.join(folder_path, "ch0GT50Hzskeleton")
    skel_file = os.path.join(skel_dir, [f for f in os.listdir(skel_dir) if f.endswith('.log')][0])

    h5_dir = os.path.join(folder_path, "h5")
    h5_file = os.path.join(h5_dir, [f for f in os.listdir(h5_dir) if f.endswith('.h5')][0])

    skeleton = load_skeleton(skel_file)
    events = load_events_lazy(h5_file)

    return {
        'skeleton': skeleton,
        'events_aligned': align_events_lazy(events, skeleton),  # generator
        'folder_path': folder_path
    }


class EventH36MDataset(Dataset):
    def __init__(self, sample, max_events=5000):
        self.skeleton = sample['skeleton']
        self.event_slices = list(sample['events_aligned'])
        self.max_events = max_events

    def __len__(self):
        return len(self.skeleton)

    def __getitem__(self, idx):
        events = self.event_slices[idx]
        skel = torch.tensor(self.skeleton[idx], dtype=torch.float32)
        N = min(self.max_events, len(events['ts']))
        event_tensor = torch.tensor(
            np.vstack([events['ts'][:N], events['x'][:N], events['y'][:N], events['pol'][:N]]).T,
            dtype=torch.float32
        )
        return event_tensor, skel


def generate_motion_gif(sample, save_path="human_motion.gif", step=10):
    frames = []
    total_frames = len(sample['events_aligned'])
    for i in range(0, total_frames, step):
        events = sample['events_aligned'][i]
        skel = sample['skeleton'][i]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(events['x'], events['y'], s=1, alpha=0.3)
        ax.scatter(skel[:, 0], skel[:, 1], c='red', s=40)
        ax.invert_yaxis()
        ax.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)

        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(save_path, frames, fps=8)


if __name__ == "__main__":
    folder = r"C:\Users\alenm\OneDrive\Desktop\EH36M\EV2\cam2_S1_Directions"
    sample = load_sample(folder)
    generate_motion_gif(sample, save_path="C:/Users/alenm/OneDrive/Desktop/human_motion.gif", step=8)
