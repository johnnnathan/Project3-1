# Event-based Human Motion Analysis with MotionBERT

This repo contains the code for our bachelor project on **human motion analysis using event-based cameras** in the context of the EDMO educational robots.

The pipeline has two main stages:

1. **Event-based 2D pose estimation**  
   We train a Transformer model on the **Event-Human3.6M (EH36M)** dataset to predict human joint locations directly from event streams.

2. **Skeleton-based action recognition with MotionBERT**  
   We use the predicted pose sequences as input to a **pretrained MotionBERT** model (NTU RGB+D 60, x-sub) to classify high-level actions.  
   This lets us explore which actions might correspond to collaborative behaviour between students and robots.

---

## 1. Repository Structure

```text
v2e-model/
├─ EH36M/
│  ├─ train_pose_model.py      # Event-based pose Transformer (training + MPJPE loss)
│  ├─ loader.py                # EH36M sample loading / dataset utilities
│  ├─ parser.py                # Parsing raw EH36M data (not all used in this project)
│  ├─ event_pose_model_final.pth   # Trained pose model weights (small enough to commit)
│  └─ cache_eh36m/             # ⚠️ LOCAL CACHE of EH36M samples (ignored in git)
│
├─ MotionBERT/
│  ├─ motionbert_loader.py     # Lightweight MotionBERT classifier wrapper
│  ├─ config/
│  │  └─ MB_ft_NTU60_xsub.yaml # Config from the original MotionBERT repo
│  ├─ labels/
│  │  └─ ntu60_labels.py       # List of NTU RGB+D 60 action labels (A1–A60)
│  ├─ visualize_pose.py        # Helpers to animate/plot predicted skeletons
│  ├─ run_inference.py         # Full pipeline: events → pose → action label
│  └─ checkpoints/             # ⚠️ PLACE MotionBERT checkpoint here (ignored in git)
│
└─ .gitignore                  # Excludes EH36M cache, MotionBERT checkpoint, venv, etc.
