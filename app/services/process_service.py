import os
from algorithms.event_shuffle import event_shuffle
from algorithms.frame_shuffle import frame_shuffle
from algorithms.frame_blur import frame_blur

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_uploaded_file(file, algorithm, form):

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if algorithm == "event_shuffle":
        interval = float(form.get("interval", 50))
        return event_shuffle(filepath, interval=interval)

    if algorithm == "frame_shuffle":
        return frame_shuffle(filepath)

    if algorithm == "frame_blur":
        sigma = float(form.get("sigma", 1.5))
        return frame_blur(filepath, sigma=sigma)

    raise ValueError("Invalid algorithm")
