import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_video
from torchvision.models import resnet18
from PIL import Image

#use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("using CPU.")

#set up dataset
class UCFSPORTS(Dataset):
    def __init__(self, root_dir, num_frames=8, resize=(112, 112)):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.resize = resize
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.samples = []

        for label_idx, action in enumerate(self.classes):
            action_path = os.path.join(root_dir, action)
            for video_folder in os.listdir(action_path):
                folder_path = os.path.join(action_path, video_folder)
                if not os.path.isdir(folder_path):
                    continue
                for file in os.listdir(folder_path):
                    if file.endswith(".avi"):
                        video_path = os.path.join(folder_path, file)
                        self.samples.append((video_path, label_idx))

        print(f"Found {len(self.samples)} videos in {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames, _, _ = read_video(video_path, pts_unit="sec")  # frames: [T,H,W,C]

        # Sample or pad frames
        if len(frames) > self.num_frames:
            start = random.randint(0, len(frames) - self.num_frames)
            frames = frames[start:start + self.num_frames]
        elif len(frames) < self.num_frames:
            pad = self.num_frames - len(frames)
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        # Convert to PIL images, resize, and convert to tensor
        processed = []
        for f in frames:
            img = Image.fromarray(f.numpy())
            img = img.resize(self.resize)
            tensor = transforms.ToTensor()(img)
            processed.append(tensor)
        frames = torch.stack(processed)  # [T,C,H,W]

        return frames, label

#set up the parameters
data_root = r"C:\Users\alenm\OneDrive\Desktop\v2e test\ResNet\ucf101_data\ucf_sports_actions\ucf_sports_actions\ucf action"
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
num_frames = 8
train_ratio = 0.8

#load data with train test split
dataset = UCFSPORTS(data_root, num_frames=num_frames, resize=(112, 112))

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print(f"Classes: {dataset.classes}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

#set up model
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in train_loader:
        frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Average frames across time dimension: [B, T, C, H, W] -> [B, C, H, W]
        if frames.dim() == 5:
            frames = frames.mean(dim=1)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    #model evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if frames.dim() == 5:
                frames = frames.mean(dim=1)
            outputs = model(frames)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

#save pth model file
torch.save(model.state_dict(), "ucf_sports_resnet18_gpu.pth")
print("\nTraining complete.")