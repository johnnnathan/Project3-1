# app/algorithms/labeler.py
import numpy as np
import base64
import io
try:
    # optional: if you plan to use a PyTorch/TensorFlow model server-side, import here
    import torch
except Exception:
    torch = None

from algorithms.utils import make_frame_for_interval

def simple_motion_label_from_frame(frame, threshold=10):
    """
    Very small rule-based labeler: compute mean brightness and decide label.
    Returns dict {'label': str, 'score': float}
    """
    # frame expected as numpy 2D or 3D array (H,W) or (H,W,3)
    if frame is None:
        return {"label": "unknown", "score": 0.0}
    arr = np.asarray(frame)
    # convert to grayscale-like
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    score = float(gray.mean())
    label = "motion" if score > threshold else "quiet"
    return {"label": label, "score": score}

# Optional API to run a real model if you add one later
_model = None
def load_model(path=None):
    global _model
    # Add model loading logic here if you want server-side labeling (torch, tf, onnxruntime, etc.)
    # Example for torch: _model = torch.load(path)
    _model = None
    return _model

def predict_with_model(frame):
    """
    If _model is loaded, run it on the frame and return predictions.
    Otherwise fall back to simple_motion_label_from_frame.
    """
    if _model is None:
        return simple_motion_label_from_frame(frame)
    # Otherwise, preprocess and run model -> this depends on model specifics
    # TODO: implement per-model once we have the model spec
    return {"label": "model_not_loaded", "score": 0.0}
