# app/algorithms/facedetect.py
import numpy as np
import base64
import io

try:
    import cv2
    _cv2_available = True
except Exception:
    cv2 = None
    _cv2_available = False

# load classifier only if cv2 available
_face_cascade = None
def _ensure_cascade():
    global _face_cascade
    if not _cv2_available:
        return None
    if _face_cascade is None:
        try:
            # use OpenCV's builtin haarcascade path if available
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            _face_cascade = None
    return _face_cascade

def detect_faces_in_frame(frame, scaleFactor=1.1, minNeighbors=4, minSize=(30,30)):
    """
    Input: numpy frame (H,W) or (H,W,3) (uint8)
    Output: list of bounding boxes [{'x':x,'y':y,'w':w,'h':h,'score':1.0}, ...]
    """
    if not _cv2_available:
        # no cv2 -> cannot detect; return empty list
        return []
    cascade = _ensure_cascade()
    if cascade is None:
        return []
    img = np.asarray(frame)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    rects = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    out = []
    for (x,y,w,h) in rects:
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 1.0})
    return out

def blur_faces_in_frame(frame, faces, blur_radius=15):
    """
    Blurs given face bounding boxes in-place and returns modified frame (numpy array).
    If cv2 is not available, returns frame unchanged.
    """
    if not _cv2_available:
        return frame
    img = frame.copy()
    for f in faces:
        x,y,w,h = f['x'], f['y'], f['w'], f['h']
        x0, y0 = max(0,x), max(0,y)
        x1, y1 = min(img.shape[1], x+w), min(img.shape[0], y+h)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        k = int(max(1, blur_radius//2)*2 + 1)
        roi_blur = cv2.GaussianBlur(roi, (k,k), 0)
        img[y0:y1, x0:x1] = roi_blur
    return img
