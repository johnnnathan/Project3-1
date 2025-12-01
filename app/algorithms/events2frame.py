# app/algorithms/events2frame.py
import numpy as np

def events_to_frame(events, width=320, height=240, point_size=1, polarity_scale=1.0):
    """
    Convert events (list of dicts or Nx3 array) to an HxW grayscale numpy frame (uint8).
    events: either Nx3 array [x,y,p] or list of {'x':..,'y':..,'p':..}
    """
    frame = np.zeros((height, width), dtype=np.float32)
    if events is None:
        return (frame.astype(np.uint8))
    # normalize input
    if isinstance(events, (list,tuple)):
        # expect dicts
        for e in events:
            try:
                x = int(e.get('x',0))
                y = int(e.get('y',0))
                p = float(e.get('p',1.0))
            except Exception:
                continue
            if 0 <= x < width and 0 <= y < height:
                frame[y, x] += (1.0 if p >= 0 else -1.0) * polarity_scale
    else:
        arr = np.asarray(events)
        if arr.ndim >= 2 and arr.shape[1] >= 3:
            xs = arr[:,0].astype(int)
            ys = arr[:,1].astype(int)
            ps = arr[:,2].astype(float)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            for x,y,p in zip(xs[valid], ys[valid], ps[valid]):
                frame[y,x] += (1.0 if p >= 0 else -1.0) * polarity_scale
    # simple normalization to 0..255
    low, high = frame.min(), frame.max()
    if high - low > 0:
        f = (frame - low) / (high - low)
    else:
        f = np.clip(frame, -1, 1) * 0.5 + 0.5
    img = (f * 255.0).astype(np.uint8)
    return img
