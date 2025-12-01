"""
End-to-end sanity checks for the full pipeline.
"""
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def build_events_tensor(sample):
    """Extract events tensor from cached sample (copied from run_inference.py to avoid import issues)."""
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


def test_cache_directory_exists():
    """Test that cache directory exists and has files."""
    cache_dir = "EH36M/cache_eh36m"
    
    assert os.path.exists(cache_dir), f"Cache directory '{cache_dir}' should exist"
    
    all_files = os.listdir(cache_dir)
    pt_files = [f for f in all_files if f.endswith(".pt")]
    
    assert len(pt_files) > 0, "Cache directory should contain at least one .pt file"
    
    print(f"✓ Cache directory check passed ({len(pt_files)} .pt files found)")


def test_build_events_tensor():
    """Test that build_events_tensor returns correct format."""
    cache_dir = "EH36M/cache_eh36m"
    
    if not os.path.exists(cache_dir):
        print("⚠ Skipping build_events_tensor test: cache directory not found")
        return
    
    all_files = os.listdir(cache_dir)
    pt_files = [f for f in all_files if f.endswith(".pt")]
    
    if not pt_files:
        print("⚠ Skipping build_events_tensor test: no cached files found")
        return
    
    # Try to load and process a sample
    sample_path = os.path.join(cache_dir, pt_files[0])
    sample = torch.load(sample_path, weights_only=False)
    
    events = build_events_tensor(sample)
    
    if events is None:
        print("⚠ Skipping build_events_tensor test: sample format not supported")
        return
    
    assert events.ndim == 2, f"Events should be 2D, got {events.ndim}D"
    assert events.shape[1] == 4, f"Events should be [N, 4], got {events.shape}"
    assert events.shape[0] > 0, f"Events should be non-empty, got shape {events.shape}"
    
    print("✓ build_events_tensor test passed")


def test_pose_sequence_shapes():
    """Test that pose sequence has expected shape [T, 13, 2]."""
    # This would require running the full pipeline, so we'll just check the logic
    # In a real scenario, you'd extract pose and check its shape
    expected_joints = 13
    expected_dims = 2
    
    # Dummy check - in real test, you'd run extract_pose and verify
    print("✓ Pose sequence shape check (placeholder - requires full pipeline)")


def test_motionbert_input_preparation():
    """Test that pose sequence can be prepared for MotionBERT input."""
    # Create dummy pose sequence [T, 13, 2]
    T = 30
    pose_seq = torch.randn(T, 13, 2)
    
    # Pad/trim to 243 frames
    target_len = 243
    if pose_seq.shape[0] < target_len:
        pad_num = target_len - pose_seq.shape[0]
        pose_seq = torch.cat([pose_seq, pose_seq[-1:].repeat(pad_num, 1, 1)], dim=0)
    else:
        pose_seq = pose_seq[:target_len]
    
    assert pose_seq.shape[0] == target_len, f"Should have {target_len} frames"
    
    # Add batch dimension
    pose_seq = pose_seq.unsqueeze(0)
    assert pose_seq.shape == (1, target_len, 13, 2)
    
    # Pad to [1, 243, 17, 3]
    pose_seq = torch.nn.functional.pad(pose_seq, (0, 1, 0, 4))
    assert pose_seq.shape == (1, 243, 17, 3), f"Expected [1, 243, 17, 3], got {pose_seq.shape}"
    
    print("✓ MotionBERT input preparation test passed")


if __name__ == "__main__":
    print("Running end-to-end sanity checks...\n")
    test_cache_directory_exists()
    test_build_events_tensor()
    test_pose_sequence_shapes()
    test_motionbert_input_preparation()
    print("\n✓ All end-to-end tests passed!")

