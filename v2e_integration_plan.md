# V2E Integration Plan for Labeling Model

## What is V2E?

**V2E (Video to Events)** is a Python-based simulator that converts conventional video frames into realistic DVS (Dynamic Vision Sensor) event streams. Developed by SensorsINI, it's the current state-of-the-art tool for generating synthetic event camera data.

**GitHub**: https://github.com/SensorsINI/v2e
**Paper**: CVPR 2021 Workshop on Event-based Vision

## Key Capabilities

- Converts low frame-rate video to high-precision event streams
- Models realistic DVS characteristics:
  - Pixel-level Gaussian event threshold mismatch
  - Finite intensity-dependent bandwidth
  - Intensity-dependent noise
  - Low illumination conditions
- Enables transfer learning from frame-based datasets to event-based models

## Use Cases for This Project

### 1. Data Augmentation
Generate synthetic labeled event data from existing labeled video datasets (ImageNet, COCO, etc.)

### 2. Training Labeling Models
- Use V2E to convert labeled videos → labeled event streams
- Train classification/detection models on synthetic events
- Transfer learning to real DVS data

### 3. Testing Under Various Conditions
- Simulate different lighting conditions (day/night/indoor)
- Test robustness of models before deploying on real hardware

## Integration Architecture

```
┌─────────────────┐
│  Labeled Video  │ (e.g., MNIST videos, object detection datasets)
│   (MP4/AVI)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│      V2E        │ ← Convert frames to events
│   Simulator     │    (with realistic DVS modeling)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Synthetic Event │ (timestamp, x, y, polarity) + labels
│   Data + Labels │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Our Visualizer │ ← Visualize and validate generated events
│  (visualizer.py)│
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Training Model  │ ← CNN/SNN/GNN for event classification/detection
│  (PyTorch/TF)   │
└─────────────────┘
```

## Implementation Steps

### Phase 1: Setup and Validation (Fundamentals)

1. **Install V2E**
   ```bash
   pip install v2e
   # OR clone from source
   git clone https://github.com/SensorsINI/v2e.git
   cd v2e
   pip install -e .
   ```

2. **Test V2E with Sample Video**
   ```bash
   python v2e.py -i input_video.mp4 -o output_events --dvs_h5 events.h5
   ```

3. **Adapter Module**: Create converter from V2E output → our format
   ```python
   # v2e_adapter.py
   import h5py
   import numpy as np

   def convert_v2e_to_txt(h5_file, output_txt):
       """Convert V2E HDF5 output to our events.txt format"""
       with h5py.File(h5_file, 'r') as f:
           events = f['events'][:]  # (t, x, y, p)
       np.savetxt(output_txt, events)
   ```

4. **Validation Pipeline**: Run V2E events through our visualizer
   - Verify event distributions look realistic
   - Compare with real DVS data if available

### Phase 2: Labeled Dataset Generation

5. **Create Labeled Event Dataset**
   - Source: MNIST videos or N-MNIST
   - Process each video with V2E
   - Maintain label mapping

6. **Dataset Structure**
   ```
   labeled_events/
   ├── train/
   │   ├── class_0/
   │   │   ├── sample_001_events.npz
   │   │   ├── sample_002_events.npz
   │   ├── class_1/
   │   ...
   ├── val/
   └── test/
   ```

### Phase 3: Model Training

7. **Choose Model Architecture**
   - **ResNet** (shown to work well with V2E data)
   - **Spiking Neural Network** (native event processing)
   - **Graph Neural Network** (if using V2CE for continuous timestamps)

8. **Training Pipeline**
   ```python
   # train_classifier.py
   # Load V2E-generated events
   # Create time surfaces or event frames
   # Train model
   # Evaluate on real DVS data
   ```

9. **Transfer Learning**
   - Pretrain on V2E synthetic data
   - Fine-tune on small real DVS dataset
   - Expected: 40% improvement (per V2E paper results)

## Technical Considerations

### V2E vs V2CE

- **V2E**: Events share discrete timestamps (limitations for SNNs/GNNs)
- **V2CE** (2024): Generates continuous timestamps, better for temporal models
- Recommendation: Start with V2E (stable), explore V2CE if needed

### Data Format Compatibility

V2E outputs:
- HDF5 (.h5) - recommended for large datasets
- Text files
- NPY/NPZ
- AVI video visualization

Our current format: Text files (timestamp, x, y, polarity)
→ Need adapter layer

### Performance Considerations

V2E processing is compute-intensive:
- Use GPU acceleration if available
- Process videos in parallel
- Consider using pre-generated datasets first (e.g., N-MNIST already exists)

## Required Dependencies

```bash
# Core
pip install torch torchvision opencv-python h5py

# V2E
pip install v2e

# Optional but recommended
pip install numpy matplotlib pandas tqdm

# For SNN models (optional)
pip install snntorch
```

## Next Steps

1. Install V2E and run basic examples
2. Create `v2e_adapter.py` to bridge formats
3. Generate small test dataset (100 samples)
4. Visualize with our tool to validate
5. Design model architecture (start simple: event frame → CNN classifier)
6. Implement training loop
7. Benchmark against baseline

## Resources

- V2E GitHub: https://github.com/SensorsINI/v2e
- V2E Paper: CVPR 2021 EventVision Workshop
- N-MNIST Dataset: Pre-converted MNIST to events (good starting point)
- Event-driven Deep Learning tutorial: https://github.com/uzh-rpg/event-driven_object_detection
