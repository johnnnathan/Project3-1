import torch
import torch.nn as nn
import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(BASE_DIR, "config", "MB_ft_NTU60_xsub.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

NUM_JOINTS = config['num_joints']
HIDDEN_DIM = config['hidden_dim']
NUM_CLASSES = config['action_classes']
MAXLEN = config['maxlen']


class SimpleMotionBERT(nn.Module):
    def __init__(self, num_joints=17, hidden_dim=2048, num_classes=60, maxlen=243):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_joints * 3 * maxlen, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.reshape(1, -1)
        return self.fc(x)


def load_motionbert_model():
    model = SimpleMotionBERT(
        num_joints=NUM_JOINTS,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        maxlen=MAXLEN
    )

    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "best_epoch.bin")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.eval()
    return model
