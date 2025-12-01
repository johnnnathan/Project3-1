import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os

# Handle imports for both direct execution and module import
try:
    from EH36M.loader import EventH36MDataset, load_sample
except ImportError:
    # Fallback for direct execution from EH36M directory
    from loader import EventH36MDataset, load_sample

# Event-Based Pose Estimation Pipeline:
# 1. Load and cache EH36M samples (event streams + skeleton keypoints) for fast access.
# 2. Build dataset of valid samples: input (x, y, time, polarity) → output 13 (x, y) keypoints.
# 3. Train Transformer model using GPU, mixed precision, and MPJPE loss.
# 4. Save model checkpoints and final weights for pose estimation inference.

torch.backends.cudnn.benchmark = True

# Debug flag: when True, train only on a tiny subset to check if the model can overfit.
DEBUG_OVERFIT = False

ALLOWED_ACTIONS = [
    "Directions", "Discussion", "Greeting", "Phoning",
    "Posing", "Sitting", "SittingDown",
    "Waiting", "Walking", "WalkTogether"
]


def identity_collate(batch):
    return batch


class EventPoseTransformer(nn.Module):
    def __init__(self, d_model=64, num_heads=4, num_layers=3, num_joints=13):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_joints * 2)
        )

    def forward(self, events):
        x = self.input_proj(events)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x).view(-1, 13, 2)


def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=2))


def train_model(model, train_loader, test_loader, epochs=20, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            events, poses = zip(*batch)
            events = torch.nn.utils.rnn.pad_sequence(events, batch_first=True).float().to(device)
            poses = torch.stack(poses).float().to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(events)
                loss = mpjpe(preds, poses)

            # Training loop stability checks
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_idx + 1}")

            scaler.scale(loss).backward()

            # Check gradient norms before optimizer step
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
            if not torch.isfinite(total_grad_norm):
                raise RuntimeError(
                    f"Non-finite grad norm detected at epoch {epoch + 1}, batch {batch_idx + 1} "
                    f"(grad norm={total_grad_norm})"
                )
            if total_grad_norm == 0:
                print(
                    f"Warning: zero gradient norm at epoch {epoch + 1}, batch {batch_idx + 1}. "
                    "Check if the loss or model parameters are correct."
                )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if DEBUG_OVERFIT:
                # In overfit mode, print loss every iteration for closer inspection.
                print(f"[DEBUG_OVERFIT] Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        # validation (skip in overfit debug mode to focus on train loss)
        if not DEBUG_OVERFIT:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    events, poses = zip(*batch)
                    events = torch.nn.utils.rnn.pad_sequence(events, batch_first=True).float().to(device)
                    poses = torch.stack(poses).float().to(device)
                    preds = model(events)
                    test_loss += mpjpe(preds, poses).item()

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MPJPE: {train_loss / len(train_loader):.4f} | "
                f"Test MPJPE: {test_loss / len(test_loader):.4f}"
            )

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"event_pose_epoch_{epoch + 1}.pth")
                print(f"Checkpoint saved at epoch {epoch + 1}")
        else:
            print(
                f"[DEBUG_OVERFIT] Epoch {epoch + 1}/{epochs} | "
                f"Train MPJPE: {train_loss / len(train_loader):.4f}"
            )

    torch.save(model.state_dict(), "event_pose_model_final.pth")
    print("\nTraining completed and final model saved.")
    return model


if __name__ == "__main__":
    root = r"C:\Users\alenm\OneDrive\Desktop\EH36M"

    print("\nLoading cached samples...")

    # Cache directory should be inside EH36M folder
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_eh36m")
    if not os.path.exists(CACHE_DIR):
        raise RuntimeError(
            f"Cache directory '{CACHE_DIR}' does not exist.\n"
            "Please run 'python EH36M/parser.py' first to cache the EH36M data."
        )
    
    cached_files = [
        os.path.join(CACHE_DIR, f)
        for f in os.listdir(CACHE_DIR) if f.endswith(".pt")
    ]

    samples = []
    for f in cached_files:
        try:
            obj = torch.load(f, weights_only=False)
            if isinstance(obj, list):
                samples.extend(obj)
            else:
                samples.append(obj)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    print(f"Total loaded objects: {len(samples)}")

    valid_samples = [
        s for s in samples
        if isinstance(s, dict) and 'events_aligned' in s and 'skeleton' in s
    ]

    print(f"Valid samples usable for training: {len(valid_samples)}")
    if len(valid_samples) == 0:
        raise RuntimeError("No valid samples found, check cached data.")

    dataset = ConcatDataset([
        EventH36MDataset(sample, max_events=1000) for sample in valid_samples
    ])
    print(f"Total dataset size: {len(dataset)} samples")

    # Basic dataset sanity check: single sample shapes
    sample_events, sample_pose = dataset[0]
    print("Single sample shapes:", sample_events.shape, sample_pose.shape)

    if DEBUG_OVERFIT:
        # Use a tiny subset of the dataset to check if the model can overfit.
        overfit_size = min(32, len(dataset))
        train_data = torch.utils.data.Subset(dataset, list(range(overfit_size)))
        test_data = torch.utils.data.Subset(dataset, list(range(overfit_size)))
        print(f"[DEBUG_OVERFIT] Using {overfit_size} samples for both train and test.")
        num_epochs = 30
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        num_epochs = 25

    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=identity_collate
    )
    test_loader = DataLoader(
        test_data, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=identity_collate
    )

    # Basic DataLoader sanity check: batch contents
    first_batch = next(iter(train_loader))
    events, poses = zip(*first_batch)
    print("First batch lens and shapes:", len(events), events[0].shape, poses[0].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventPoseTransformer().to(device)

    if os.path.exists("event_pose_model_final.pth"):
        model.load_state_dict(torch.load("event_pose_model_final.pth"))
        print("Loaded existing model checkpoint.")

    trained_model = train_model(
        model, train_loader, test_loader,
        epochs=num_epochs, lr=1e-4
    )
