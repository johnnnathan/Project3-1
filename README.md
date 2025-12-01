# Event-based Human Motion Analysis with MotionBERT
This repo implements a **two-stage pipeline**:
1. **Event-based pose estimation**
A Transformer model is trained on **Event-Human3.6M (EH36M)** to predict 2D body joints from
event streams.
2. **Skeleton-based action recognition**
The predicted poses are fed into a pretrained **MotionBERT** classifier (NTU RGB+D 60, x-sub) to
obtain a high-level action label (e.g. *sit down*, *walk towards*, *clapping*).
This setup is a step towards **privacy-friendly behaviour analysis** for the EDMO educational
robots.
---
## Project Structure
```text
EH36M/
train_pose_model.py # EventPoseTransformer + training loop
loader.py # EH36M dataset utilities
event_pose_model_final.pth # trained pose model (2D, 13 joints)
cache_eh36m/ # local EH36M cache (ignored by git)
MotionBERT/
motionbert_loader.py # loads MotionBERT config + checkpoint
config/MB_ft_NTU60_xsub.yaml
labels/ntu60_labels.py # NTU RGB+D 60 action names
visualize_pose.py # simple pose plotting / GIF export
run_inference.py # events → pose → action
.venv/ # local virtual env (ignored)
.gitignore
```
> **Not stored in GitHub (too large):**
> - `EH36M/cache_eh36m/*.pt` – cached event + skeleton samples
> - `MotionBERT/checkpoints/best_epoch.bin` – MotionBERT NTU60 checkpoint (https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin)
You must provide these locally. 
---
## 0. Setup & Environment
We do not track the virtual environment (.venv) in git.
Instead, everyone recreates the environment from requirements.txt.

From the repo root:
```text
# 1) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# (on Linux/macOS: source .venv/bin/activate)

# 2) install dependencies
pip install -r requirements.txt
```

Then place the data/checkpoints:
-EH36M cached samples → EH36M/cache_eh36m/
-MotionBERT checkpoint → MotionBERT/checkpoints/best_epoch.bin

After that, the scripts will run the same way on any machine.
---
## 1. Pose Estimation on EH36M
`EH36M/train_pose_model.py` trains `EventPoseTransformer`:
- **Input:** variable-length events `[N, 4]` → (x, y, timestamp, polarity)
- **Output:** `[B, 13, 2]` → 13 joint coordinates in 2D
- **Loss:** MPJPE (Mean Per Joint Position Error)
The final model is saved as:
```text
EH36M/event_pose_model_final.pth
```
---
## 2. Action Recognition with MotionBERT
`MotionBERT/motionbert_loader.py` builds a small MotionBERT classifier using:
- `config/MB_ft_NTU60_xsub.yaml`
- `checkpoints/best_epoch.bin` (downloaded separately)
It outputs logits over **60 NTU RGB+D** actions.
Labels are in `labels/ntu60_labels.py`.
---
## 3. Full Pipeline: Events → Pose → Action
`MotionBERT/run_inference.py` does:
1. Load a cached EH36M sample from `EH36M/cache_eh36m/`.
2. Flatten and sort all events by time.
3. Split the stream into fixed temporal frames.
4. For each frame, run `EventPoseTransformer` → pose sequence `[T, 13, 2]`.
5. Pad/trim to 243 frames and expand to `[1, 243, 17, 3]` for MotionBERT.
6. Run MotionBERT and map the predicted ID to an action string.
Usage:
```bash
# create/activate venv first
python MotionBERT/run_inference.py
```
Example output:
```text
Using device: cuda
Loading sample: EH36M/cache_eh36m/cam2_S11_SittingDown_0.pt
Pose sequence shape: torch.Size([30, 13, 2])
MotionBERT input: torch.Size([1, 243, 17, 3])
Detected Action ID: 8
Action Label: sit down
```
You can optionally generate a GIF of the predicted skeleton with:
```python
from visualize_pose import animate_pose_sequence
animate_pose_sequence(pose_seq, save_path="pose_sequence.gif")
```
---
## 4. Notes & Future Work
- MotionBERT is trained on **RGB-based skeletons**, not event-derived poses; here it is used in a
zero-shot transfer setting.
- For EDMO classrooms, the same pipeline can be applied to **real event-camera recordings** of
students working with robots, and actions can later be mapped to collaboration indicators (e.g.
*walking towards*, *hugging*, *hand waving* → social interaction).
