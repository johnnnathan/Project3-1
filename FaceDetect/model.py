"""
EventFaceDetector - Event-based Face Detection Model

Converts event streams to voxel grids and detects faces using a CNN-based detector.
Designed to work with neuromorphic camera data for privacy-preserving face detection.
"""

import torch
import torch.nn as nn
import numpy as np


class EventVoxelEncoder(nn.Module):
    """
    Convert raw events to voxel grid representation.

    Events (N, 4) -> Voxel Grid (B, C, H, W)
    where C = num_time_bins * 2 (positive/negative polarity)

    This preserves spatial information that would be lost with global pooling.
    """

    def __init__(self, resolution=(128, 128), num_time_bins=5):
        super().__init__()
        self.resolution = resolution  # (H, W)
        self.num_time_bins = num_time_bins
        self.num_channels = num_time_bins * 2  # pos/neg polarity

    def forward(self, events):
        """
        Convert events to voxel grid.

        Args:
            events: (N, 4) tensor - x, y, t, p (coordinates, time, polarity)

        Returns:
            voxel_grid: (C, H, W) tensor
        """
        h, w = self.resolution
        device = events.device if torch.is_tensor(events) else 'cpu'

        voxel = torch.zeros((self.num_channels, h, w), device=device, dtype=torch.float32)

        if len(events) == 0:
            return voxel

        if not torch.is_tensor(events):
            events = torch.tensor(events, dtype=torch.float32, device=device)

        # Extract components
        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3]

        # Normalize timestamps to [0, 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
        else:
            t_norm = torch.zeros_like(t)

        # Convert to grid coordinates
        x_idx = (x * (w - 1)).long().clamp(0, w - 1)
        y_idx = (y * (h - 1)).long().clamp(0, h - 1)

        # Assign to time bins
        time_bins = (t_norm * self.num_time_bins).long().clamp(0, self.num_time_bins - 1)

        # Separate by polarity
        pos_mask = p > 0
        neg_mask = ~pos_mask

        # Accumulate events into voxel grid
        for i in range(self.num_time_bins):
            bin_mask = time_bins == i

            # Positive polarity
            pos_bin_mask = bin_mask & pos_mask
            if pos_bin_mask.any():
                pos_coords = torch.stack([y_idx[pos_bin_mask], x_idx[pos_bin_mask]], dim=0)
                voxel[i].index_put_(
                    tuple(pos_coords),
                    torch.ones(pos_bin_mask.sum(), device=device),
                    accumulate=True
                )

            # Negative polarity
            neg_bin_mask = bin_mask & neg_mask
            if neg_bin_mask.any():
                neg_coords = torch.stack([y_idx[neg_bin_mask], x_idx[neg_bin_mask]], dim=0)
                voxel[self.num_time_bins + i].index_put_(
                    tuple(neg_coords),
                    torch.ones(neg_bin_mask.sum(), device=device),
                    accumulate=True
                )

        return voxel


class EventFaceDetector(nn.Module):
    """
    Face detector for event camera data.

    Architecture:
    - Input: Voxel grid from EventVoxelEncoder
    - Backbone: Simple CNN for feature extraction
    - Head: YOLO-style detection head

    Output for each grid cell:
    - objectness score (is there a face?)
    - bounding box (x, y, w, h) relative to cell
    """

    def __init__(self, num_time_bins=5, num_anchors=3, grid_size=8, num_classes=1):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        self.num_classes = num_classes

        in_channels = num_time_bins * 2  # pos + neg polarity

        # CNN Backbone
        self.backbone = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Detection Head
        # Output: num_anchors * (5 + num_classes) per cell
        # 5 = objectness + x + y + w + h
        out_channels = num_anchors * (5 + num_classes)

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1),
        )

        # Default anchor boxes (width, height) relative to grid cell
        self.register_buffer('anchors', torch.tensor([
            [0.5, 0.5],   # Small face
            [0.75, 0.75], # Medium face
            [1.0, 1.0],   # Large face
        ]))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_grid):
        """
        Forward pass.

        Args:
            voxel_grid: (B, C, H, W) voxel grid from EventVoxelEncoder

        Returns:
            output: (B, num_anchors, grid_h, grid_w, 5 + num_classes)
                    Contains [obj, x, y, w, h, class_scores...]
        """
        # Feature extraction
        features = self.backbone(voxel_grid)

        # Detection
        output = self.head(features)

        # Reshape: (B, A*(5+C), H, W) -> (B, A, H, W, 5+C)
        batch_size = output.shape[0]
        grid_h, grid_w = output.shape[2], output.shape[3]

        output = output.view(
            batch_size,
            self.num_anchors,
            5 + self.num_classes,
            grid_h,
            grid_w
        )
        output = output.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 5+C)

        return output

    def decode_predictions(self, output, conf_threshold=0.5, nms_threshold=0.4):
        """
        Decode model output to bounding boxes.

        Args:
            output: (B, A, H, W, 5+C) model output
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS

        Returns:
            List of detections per batch item, each detection is [x, y, w, h, conf, class]
        """
        batch_size = output.shape[0]
        grid_h, grid_w = output.shape[2], output.shape[3]

        all_detections = []

        for b in range(batch_size):
            detections = []

            for a in range(self.num_anchors):
                for i in range(grid_h):
                    for j in range(grid_w):
                        pred = output[b, a, i, j]

                        # Objectness score
                        obj_conf = torch.sigmoid(pred[0])

                        if obj_conf < conf_threshold:
                            continue

                        # Bounding box
                        x = (torch.sigmoid(pred[1]) + j) / grid_w
                        y = (torch.sigmoid(pred[2]) + i) / grid_h
                        w = torch.exp(pred[3]) * self.anchors[a, 0] / grid_w
                        h = torch.exp(pred[4]) * self.anchors[a, 1] / grid_h

                        # Class score (for single class, just use objectness)
                        if self.num_classes > 1:
                            class_scores = torch.softmax(pred[5:], dim=0)
                            class_conf, class_idx = class_scores.max(0)
                            conf = obj_conf * class_conf
                        else:
                            conf = obj_conf
                            class_idx = torch.tensor(0)

                        if conf >= conf_threshold:
                            detections.append([
                                x.item(), y.item(),
                                w.item(), h.item(),
                                conf.item(),
                                class_idx.item()
                            ])

            # Apply NMS
            if detections:
                detections = self._nms(detections, nms_threshold)

            all_detections.append(detections)

        return all_detections

    def _nms(self, detections, threshold):
        """Non-maximum suppression."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best[:4], d[:4]) < threshold
            ]

        return keep

    def _iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to corners
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def create_model(num_time_bins=5, pretrained=False, checkpoint_path=None):
    """
    Factory function to create EventFaceDetector.

    Args:
        num_time_bins: Number of temporal bins for voxel grid
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file

    Returns:
        model: EventFaceDetector instance
    """
    model = EventFaceDetector(num_time_bins=num_time_bins)

    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    return model


if __name__ == '__main__':
    # Quick test
    print("Testing EventFaceDetector...")

    # Create model
    model = EventFaceDetector(num_time_bins=5)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dummy voxel grid
    batch_size = 2
    voxel = torch.randn(batch_size, 10, 128, 128)  # 10 channels = 5 bins * 2 polarities

    # Forward pass
    output = model(voxel)
    print(f"Input shape: {voxel.shape}")
    print(f"Output shape: {output.shape}")

    # Decode predictions
    detections = model.decode_predictions(output, conf_threshold=0.3)
    print(f"Detections per batch: {[len(d) for d in detections]}")

    print("Test passed!")
