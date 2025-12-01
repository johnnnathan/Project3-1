"""
Sanity checks for EH36M pose estimation model and dataset.
"""
import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EH36M.train_pose_model import EventPoseTransformer, mpjpe
from EH36M.loader import EventH36MDataset


def test_model_forward_pass():
    """Test that EventPoseTransformer produces correct output shapes."""
    model = EventPoseTransformer()
    model.eval()
    
    with torch.no_grad():
        dummy_events = torch.randn(2, 50, 4)  # B=2, T=50
        out = model(dummy_events)
        
        assert out.shape == (2, 13, 2), f"Unexpected output shape: {out.shape}"
        assert torch.isfinite(out).all(), "Non-finite values in model output"
    
    print("✓ Model forward pass test passed")


def test_model_loss_computation():
    """Test that MPJPE loss computes correctly."""
    model = EventPoseTransformer()
    model.eval()
    
    with torch.no_grad():
        dummy_events = torch.randn(2, 50, 4)
        out = model(dummy_events)
        dummy_target = torch.zeros_like(out)
        loss = mpjpe(out, dummy_target)
        
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"
    
    print("✓ Model loss computation test passed")


def test_dataset_sample_shapes():
    """Test that EventH36MDataset returns correct shapes."""
    # This requires cached data, so we'll check if it exists
    cache_dir = "EH36M/cache_eh36m"
    if not os.path.exists(cache_dir):
        print("⚠ Skipping dataset test: cache directory not found")
        return
    
    cached_files = [
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir) if f.endswith(".pt")
    ]
    
    if not cached_files:
        print("⚠ Skipping dataset test: no cached files found")
        return
    
    # Load a sample
    sample = torch.load(cached_files[0], weights_only=False)
    if isinstance(sample, list):
        sample = sample[0] if sample else None
    
    if sample is None or not isinstance(sample, dict):
        print("⚠ Skipping dataset test: invalid sample format")
        return
    
    if 'events_aligned' not in sample or 'skeleton' not in sample:
        print("⚠ Skipping dataset test: sample missing required keys")
        return
    
    dataset = EventH36MDataset(sample, max_events=1000)
    
    if len(dataset) == 0:
        print("⚠ Skipping dataset test: empty dataset")
        return
    
    sample_events, sample_pose = dataset[0]
    
    assert sample_events.ndim == 2, f"Events should be 2D, got {sample_events.ndim}D"
    assert sample_events.shape[1] == 4, f"Events should be [N, 4], got {sample_events.shape}"
    assert sample_pose.shape == (13, 2), f"Pose should be [13, 2], got {sample_pose.shape}"
    
    print("✓ Dataset sample shapes test passed")


def test_training_stability():
    """Test that training loop can handle a small batch without errors."""
    model = EventPoseTransformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dummy batch
    dummy_events = [torch.randn(50, 4) for _ in range(2)]
    dummy_poses = torch.randn(2, 13, 2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    events_padded = torch.nn.utils.rnn.pad_sequence(dummy_events, batch_first=True).float().to(device)
    poses = dummy_poses.float().to(device)
    
    optimizer.zero_grad()
    preds = model(events_padded)
    loss = mpjpe(preds, poses)
    
    # Check loss is finite
    assert torch.isfinite(loss), "Loss should be finite"
    
    loss.backward()
    
    # Check gradients
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
    assert torch.isfinite(total_grad_norm), "Gradient norm should be finite"
    assert total_grad_norm > 0, "Gradient norm should be non-zero"
    
    optimizer.step()
    
    print("✓ Training stability test passed")


if __name__ == "__main__":
    print("Running EH36M sanity checks...\n")
    test_model_forward_pass()
    test_model_loss_computation()
    test_dataset_sample_shapes()
    test_training_stability()
    print("\n✓ All EH36M tests passed!")

