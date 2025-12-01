import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from EH36M.loader import EventH36MDataset, load_sample
import os

# Event-Based Pose Estimation Pipeline:
# 1. Load and cache EH36M samples (event streams + skeleton keypoints) for fast access.
# 2. Build dataset of valid samples: input (x, y, time, polarity) → output 13 (x, y) keypoints.
# 3. Train Transformer model using GPU, mixed precision, and MPJPE loss.
# 4. Save model checkpoints and final weights for pose estimation inference.

torch.backends.cudnn.benchmark = True

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

        for batch in train_loader:
            events, poses = zip(*batch)
            events = torch.nn.utils.rnn.pad_sequence(events, batch_first=True).float().to(device)
            poses = torch.stack(poses).float().to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(events)
                loss = mpjpe(preds, poses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # validation
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

    torch.save(model.state_dict(), "event_pose_model_final.pth")
    print("\nTraining completed and final model saved.")
    return model


if __name__ == "__main__":
    root = r"C:\Users\alenm\OneDrive\Desktop\EH36M"

    print("\nLoading cached samples...")

    CACHE_DIR = "cache_eh36m"
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

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=identity_collate
    )
    test_loader = DataLoader(
        test_data, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=identity_collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventPoseTransformer().to(device)

    if os.path.exists("event_pose_model_final.pth"):
        model.load_state_dict(torch.load("event_pose_model_final.pth"))
        print("Loaded existing model checkpoint.")

    trained_model = train_model(
        model, train_loader, test_loader,
        epochs=25, lr=1e-4
    )
