# Event-based Human Motion Analysis with MotionBERT

This repository contains the code for our bachelor project on **human motion analysis using event-based cameras** in the context of the EDMO educational robots.

The pipeline has two main stages:

1. **Event-based 2D pose estimation**  
   We train a Transformer model on the **Event-Human3.6M (EH36M)** dataset to predict human joint locations directly from event streams.

2. **Skeleton-based action recognition with MotionBERT**  
   We use the predicted pose sequences as input to a **pre-trained MotionBERT** model (NTU RGB+D 60, x-sub) to classify high-level actions.

This allows us to explore which actions might correspond to collaborative behaviour between students and robots.

---

## Repository Structure

```text
v2e-model/
├─ EH36M/
│  ├─ train_pose_model.py              # Event-based pose Transformer (training + MPJPE loss)
│  ├─ loader.py                        # EH36M sample loading / dataset utilities
│  ├─ parser.py                        # Parsing raw EH36M data (not all used)
│  ├─ event_pose_model_final.pth       # Trained pose model weights (small enough to commit)
│  └─ cache_eh36m/                     # ⚠️ LOCAL CACHE of EH36M samples (git-ignored)
│
├─ MotionBERT/
│  ├─ motionbert_loader.py             # Lightweight MotionBERT classifier wrapper
│  ├─ config/
│  │  └─ MB_ft_NTU60_xsub.yaml         # Config from the original MotionBERT repo
│  ├─ labels/
│  │  └─ ntu60_labels.py               # NTU RGB+D 60 action labels (A1–A60)
│  ├─ visualize_pose.py                # Helpers to animate/plot predicted skeletons
│  ├─ run_inference.py                 # Full pipeline: events → pose → action label
│  └─ checkpoints/                     # ⚠️ PLACE MotionBERT checkpoint here (git-ignored)
│
└─ .gitignore                          # Excludes caches, checkpoints, venv, etc.
Important notes

EH36M/cache_eh36m/ is not in the repo (too large). It contains cached .pt samples (event streams + skeletons).
MotionBERT/checkpoints/best_epoch.bin is also not committed (~230 MB). You must download it yourself (see MotionBERT model zoo) and place it there.


Environment Setup
Python 3.10+ and PyTorch are required.
Bash# Create and activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib pyyaml tqdm

Stage 1 – Event-based Pose Estimation (EH36M)
3.1 Dataset: Event-Human3.6M (EH36M)
EH36M extends Human3.6M with event-camera recordings.
Each cached sample (EH36M/cache_eh36m/*.pt) contains:

























KeyDescriptionx, yPixel coordinatestsEvent timestamppolPolarity (brightness change sign)joints2dGround-truth 2D joint positions
Cached files are generated offline to avoid expensive parsing at runtime.
3.2 Model: EventPoseTransformer
Defined in EH36M/train_pose_model.py

Input: variable-length event sequence [N, 4] → (x, y, t, pol)
Architecture:
Linear projection (4 → d_model)
nn.TransformerEncoder (multi-head self-attention)
Global average pooling over time
MLP regressor → [B, 13, 2] (13 joints in 2D)

Loss: MPJPE (Mean Per Joint Position Error)
Training: Adam + mixed precision (when CUDA available)

Trained weights: EH36M/event_pose_model_final.pth

Stage 2 – Action Recognition with MotionBERT
Why MotionBERT?
MotionBERT is a state-of-the-art transformer for skeleton-based action recognition. It excels on NTU RGB+D 60/120.
We treat our event-based pose model as a feature extractor → MotionBERT classifies the resulting skeleton sequence.
MotionBERT Setup

























ItemPath / InfoConfigMotionBERT/config/MB_ft_NTU60_xsub.yamlCheckpoint (download yourself)MotionBERT/checkpoints/best_epoch.bin (NTU60 x-sub)WrapperMotionBERT/motionbert_loader.pyAction labels (60 classes)MotionBERT/labels/ntu60_labels.py
Example labels:
Python"drink water", "eat meal", "sit down", "stand up", "clapping",
"hand waving", "walking towards", "walking apart", "shaking hands", ...
Full Inference Pipeline
MotionBERT/run_inference.py glues everything together:

Load both models (pose + MotionBERT)
Load a cached EH36M sample
Flatten & sort events by timestamp
Split into fixed number of frames (default 30)
Run pose estimation per frame → [T, 13, 2]
Pad/trim to 243 frames, add missing joints & z-channel → [1, 243, 17, 3]
MotionBERT forward pass → action label

Bashpython MotionBERT/run_inference.py
Example output:
textUsing device: cuda
Using sample: EH36M/cache_eh36m/cam2_S11_Directions.pt
Pose sequence shape: torch.Size([30, 13, 2])
MotionBERT input: torch.Size([1, 243, 17, 3])
Detected Action ID: 8
Action Label: sit down
A GIF (pose_sequence.gif) is also saved for visual inspection.

Running the Full Pipeline
Prerequisites

Local EH36M cache in EH36M/cache_eh36m/ (.pt files)
`EH36M
InterruptedI MEAN EVERYTHING should me in markdown formMarkdown# Event-based Human Motion Analysis with MotionBERT

This repository contains the code for our bachelor project on **human motion analysis using event-based cameras** in the context of the EDMO educational robots.

The pipeline has two main stages:

1. **Event-based 2D pose estimation**  
   We train a Transformer model on the **Event-Human3.6M (EH36M)** dataset to predict human joint locations directly from event streams.

2. **Skeleton-based action recognition with MotionBERT**  
   We use the predicted pose sequences as input to a **pre-trained MotionBERT** model (NTU RGB+D 60, x-sub) to classify high-level actions.

This lets us explore which actions might correspond to collaborative behaviour between students and robots.

---

## Repository Structure
v2e-model/
├─ EH36M/
│  ├─ train_pose_model.py              # Event-based pose Transformer (training + MPJPE loss)
│  ├─ loader.py                        # EH36M sample loading / dataset utilities
│  ├─ parser.py                        # Parsing raw EH36M data (not all used in this project)
│  ├─ event_pose_model_final.pth       # Trained pose model weights (small enough to commit)
│  └─ cache_eh36m/                     # LOCAL CACHE of EH36M samples (ignored in git)
│
├─ MotionBERT/
│  ├─ motionbert_loader.py             # Lightweight MotionBERT classifier wrapper
│  ├─ config/
│  │  └─ MB_ft_NTU60_xsub.yaml         # Config from the original MotionBERT repo
│  ├─ labels/
│  │  └─ ntu60_labels.py               # List of NTU RGB+D 60 action labels (A1–A60)
│  ├─ visualize_pose.py                # Helpers to animate/plot predicted skeletons
│  ├─ run_inference.py                 # Full pipeline: events → pose → action label
│  └─ checkpoints/                     # PLACEInterruptedthe part in the red square is GOOD, but under its just grokai text
i cant copy paste that
so MAKE EVERYTHING FIT INTO A MARKDOWN SECTIONHere is the complete, ready-to-paste GitHub README.md — everything is properly formatted Markdown, looks great when rendered, and includes every section from your original text:
Markdown# Event-based Human Motion Analysis with MotionBERT

This repository contains the code for our bachelor project on **human motion analysis using event-based cameras** in the context of the EDMO educational robots.

The pipeline has two main stages:

1. **Event-based 2D pose estimation**  
   We train a Transformer model on the **Event-Human3.6M (EH36M)** dataset to predict human joint locations directly from event streams.

2. **Skeleton-based action recognition with MotionBERT**  
   We use the predicted pose sequences as input to a **pre-trained MotionBERT** model (NTU RGB+D 60, x-sub) to classify high-level actions.

This lets us explore which actions might correspond to collaborative behaviour between students and robots.

---

## Repository Structure

```text
v2e-model/
├─ EH36M/
│  ├─ train_pose_model.py              # Event-based pose Transformer (training + MPJPE loss)
│  ├─ loader.py                        # EH36M sample loading / dataset utilities
│  ├─ parser.py                        # Parsing raw EH36M data (not all used in this project)
│  ├─ event_pose_model_final.pth       # Trained pose model weights (small enough to commit)
│  └─ cache_eh36m/                     # LOCAL CACHE of EH36M samples (ignored in git)
│
├─ MotionBERT/
│  ├─ motionbert_loader.py             # Lightweight MotionBERT classifier wrapper
│  ├─ config/
│  │  └─ MB_ft_NTU60_xsub.yaml         # Config from the original MotionBERT repo
│  ├─ labels/
│  │  └─ ntu60_labels.py               # List of NTU RGB+D 60 action labels (A1–A60)
│  ├─ visualize_pose.py                # Helpers to animate/plot predicted skeletons
│  ├─ run_inference.py                 # Full pipeline: events → pose → action label
│  └─ checkpoints/                     # PLACE MotionBERT checkpoint here (ignored in git)
│
└─ .gitignore                          # Excludes EH36M cache, MotionBERT checkpoint, venv, etc.
Important

EH36M/cache_eh36m/ is not in the repo (too large). It contains cached .pt samples (event streams + skeletons).
MotionBERT/checkpoints/best_epoch.bin (~230 MB) is also not committed. Download it from the official MotionBERT model zoo and place it there.


Environment Setup
Bash# Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib pyyaml tqdm

Stage 1 – Event-based 2D Pose Estimation (EH36M)
Dataset: Event-Human3.6M (EH36M)
EH36M extends the classic Human3.6M dataset with event-camera recordings.
Each cached sample (EH36M/cache_eh36m/*.pt) contains synchronized event streams and 2D skeletons.
Event format per sample:

x, y – pixel coordinates
ts – timestamp
pol – polarity (±1)
Ground-truth 2D joints (for supervision)

Model: EventPoseTransformer
Defined in EH36M/train_pose_model.py
Input → variable-length event sequence [N, 4] (x, y, t, pol)
Architecture

Linear projection (4 → d_model)
nn.TransformerEncoder (several layers, multi-head self-attention)
Global average pooling over the time dimension
MLP head → [B, 13, 2] (13 body joints in 2D)

Loss: MPJPE (Mean Per Joint Position Error)
Optimizer: Adam + mixed precision (torch.cuda.amp when available)
Trained weights (included):
EH36M/event_pose_model_final.pth

Stage 2 – Action Recognition with MotionBERT
Why MotionBERT?
MotionBERT is a state-of-the-art transformer model for skeleton-based action recognition, achieving top performance on NTU RGB+D 60/120.
Here we use our event-derived 2D poses as input in a zero-shot transfer setting.
Setup

























ComponentLocation / InfoConfigMotionBERT/config/MB_ft_NTU60_xsub.yamlPre-trained checkpoint (download)MotionBERT/checkpoints/best_epoch.bin (NTU60 x-sub)WrapperMotionBERT/motionbert_loader.pyAction labels (60 classes)MotionBERT/labels/ntu60_labels.py
Selected NTU-60 labels (excerpt):
Python"drink water", "eat meal", "brush teeth", "sit down", "stand up",
"clapping", "hand waving", "walking towards", "walking apart", "shaking hands", ...
Full Inference Pipeline
MotionBERT/run_inference.py implements the complete flow:

Load both models (pose + MotionBERT)
Load a cached EH36M sample
Flatten & sort events by timestamp
Split into fixed number of frames (default: 30)
Per-frame pose estimation → [T, 13, 2]
Pad/trim to 243 frames, add missing joints & z-channel → [1, 243, 17, 3]
MotionBERT inference → predicted action label
(Optional) Save animated GIF of the predicted skeleton

Bashpython MotionBERT/run_inference.py
Example output:
textUsing device: cuda
Using sample: EH36M/cache_eh36m/cam2_S11_Directions.pt
Pose sequence shape: torch.Size([30, 13, 2])
MotionBERT input: torch.Size([1, 243, 17, 3])
Detected Action ID: 8
Action Label: sit down
A file pose_sequence.gif is saved in the MotionBERT/ folder for visual inspection.

Running the Full Pipeline
Prerequisites

Local EH36M cache → EH36M/cache_eh36m/ (.pt files)
Trained pose model → EH36M/event_pose_model_final.pth (included)
MotionBERT checkpoint → MotionBERT/checkpoints/best_epoch.bin (download yourself)

Bash# Activate environment
.\.venv\Scripts\activate   # or: source .venv/bin/activate

# Run inference on a sample
python MotionBERT/run_inference.py

Relation to EDMO & Collaboration
This project is a proof-of-concept for privacy-preserving classroom analysis:

Event cameras only record brightness changes → no recognizable images of children.
The pipeline (events → pose → action) works in real time on resource-constrained hardware.

Future steps for the EDMO ecosystem:

Record event data of children interacting with EDMO robots.
Run the same pipeline.
Map detected actions (e.g., "walking towards", "hand waving", "shaking hands") to indicators of collaboration, engagement, or communication.


Limitations & Future Work

MotionBERT was trained on RGB-derived skeletons, not event-based poses → current use is zero-shot transfer.
We predict only 2D poses and pad missing joints/z-coordinates → a full event-based 3D pose estimator would improve accuracy.
Large files (EH36M cache & MotionBERT checkpoint) are not included due to size.

Despite these limitations, the repository demonstrates a complete, end-to-end event-based human motion analysis pipeline ready for further development in educational robotics.

Enjoy experimenting!
Feel free to open issues or PRs if you extend the project.
