import os
import numpy as np
from algorithms.utils import make_frame_for_interval, create_frames, save_animation

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def visualize_uploaded_events(file, form):

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    events = np.loadtxt(filepath)
    timestamps = events[:, 0]

    interval = float(form.get("interval", 50))
    chunk_size = make_frame_for_interval(interval, timestamps)

    frames = create_frames(events, chunk_size)
    gif_name = f"visual_{file.filename}.gif"
    gif_path = os.path.join(PROCESSED_FOLDER, gif_name)

    save_animation(frames, gif_path, chunk_size, title_prefix="Visualization")

    return f"/processed/{gif_name}"
