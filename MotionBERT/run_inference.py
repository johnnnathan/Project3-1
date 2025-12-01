import random
import numpy as np
import torch
import os
from EH36M.train_pose_model import EventPoseTransformer
from motionbert_loader import load_motionbert_model
from MotionBERT.labels.ntu60_labels import NTU60_LABELS
from visualize_pose import animate_pose_sequence

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

pose_model = EventPoseTransformer()
model_path = os.path.join(os.path.dirname(__file__), "../EH36M/event_pose_model_final.pth")
pose_model.load_state_dict(torch.load(model_path, map_location=device))
pose_model.to(device)
pose_model.eval()

action_model = load_motionbert_model()
action_model.to(device)
action_model.eval()


def split_events_into_frames(events, num_frames=30):
    min_t, max_t = events[:, 2].min(), events[:, 2].max()
    frame_duration = (max_t - min_t) / num_frames
    event_frames = []
    for i in range(num_frames):
        start_t = min_t + i * frame_duration
        end_t = start_t + frame_duration
        frame_events = events[(events[:, 2] >= start_t) & (events[:, 2] < end_t)]
        if len(frame_events) > 0:
            event_frames.append(frame_events)
    return event_frames


def extract_pose(event_frames, pose_model, max_events_per_frame=2000):
    pose_sequence = []
    with torch.no_grad():
        for frame in event_frames:
            frame = frame[:max_events_per_frame]
            frame_tensor = frame.float().unsqueeze(0).to(device)
            pose = pose_model(frame_tensor)
            pose_sequence.append(pose.squeeze(0).cpu())
    return torch.stack(pose_sequence)


def pad_or_trim(pose_seq, target_len=243):
    if pose_seq.shape[0] < target_len:
        pad_num = target_len - pose_seq.shape[0]
        pose_seq = torch.cat([pose_seq, pose_seq[-1:].repeat(pad_num, 1, 1)], dim=0)
    else:
        pose_seq = pose_seq[:target_len]
    return pose_seq


def build_events_tensor(sample):
    if isinstance(sample, dict):
        raw_events = sample.get("events_aligned", sample)
    elif isinstance(sample, list):
        raw_events = []
        for s in sample:
            if isinstance(s, dict) and "events_aligned" in s:
                raw_events.extend(s["events_aligned"])
            else:
                raw_events.append(s)
    else:
        raise RuntimeError(f"Unexpected sample type: {type(sample)}")

    if isinstance(raw_events, list) and len(raw_events) == 0:
        return None

    if isinstance(raw_events, list) and isinstance(raw_events[0], dict):
        print("Event keys detected:", raw_events[0].keys())
        tensor_chunks = []
        for batch in raw_events:
            if "x" in batch:
                ts_array = batch.get("ts", batch.get("timestamp", batch.get("t", [0])))
                pol_array = batch.get("pol", batch.get("p", [0]))
                x = torch.as_tensor(batch["x"], dtype=torch.float32)
                y = torch.as_tensor(batch["y"], dtype=torch.float32)
                ts = torch.as_tensor(ts_array, dtype=torch.float32)
                pol = torch.as_tensor(pol_array, dtype=torch.float32)
                flat = torch.stack([x, y, ts, pol], dim=1)
                tensor_chunks.append(flat)
            elif "events" in batch:
                ev = torch.as_tensor(batch["events"], dtype=torch.float32)
                if ev.numel() > 0:
                    tensor_chunks.append(ev)
        if len(tensor_chunks) == 0:
            return None
        events = torch.cat(tensor_chunks, dim=0)
    else:
        events = torch.as_tensor(raw_events, dtype=torch.float32)

    if events.numel() == 0:
        return None

    return events


base_dir = os.path.dirname(__file__)
cache_dir = os.path.join(base_dir, "../EH36M/cache_eh36m")
all_files = os.listdir(cache_dir)
sitting_files = [f for f in all_files if "SittingDown" in f and f.endswith(".pt")]

if not sitting_files:
    raise RuntimeError("No SittingDown samples found in cache_eh36m")

sitting_files.sort()

events = None
chosen_file = None
for fname in sitting_files:
    path = os.path.join(cache_dir, fname)
    print("Trying sample:", path)
    sample = torch.load(path, weights_only=False)
    events = build_events_tensor(sample)
    if events is not None and events.shape[0] > 0:
        chosen_file = path
        break

if events is None or events.shape[0] == 0:
    raise RuntimeError("No SittingDown file with non-empty events found.")

print("Using sample:", chosen_file)
events = events[events[:, 2].argsort()]

print(f"Total flattened events: {events.shape}")

frames = split_events_into_frames(events)
print(f"Event frames: {len(frames)}")

pose_seq = extract_pose(frames, pose_model)
print(f"Pose sequence shape: {pose_seq.shape}")
pose_seq_vis = pose_seq.clone()

pose_seq = pad_or_trim(pose_seq)
pose_seq = pose_seq.unsqueeze(0)

pose_seq = torch.nn.functional.pad(pose_seq, (0, 1, 0, 4))
print(f"MotionBERT input: {pose_seq.shape}")

pose_seq = pose_seq.to(device)

with torch.no_grad():
    logits = action_model(pose_seq)
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_id = torch.argmax(logits, dim=-1).item()

pred_label = NTU60_LABELS[pred_id] if pred_id < len(NTU60_LABELS) else "Unknown"

print(f"Detected Action ID: {pred_id}")
print(f"Action Label: {pred_label}")

animate_pose_sequence(pose_seq_vis, save_path="pose_sequence.gif")
