import os
from algorithms.event_shuffle import event_shuffle
from algorithms.frame_shuffle import frame_shuffle
from algorithms.frame_blur import frame_blur

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def process_uploaded_file(file, algorithm, form):

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if algorithm == "event_shuffle":
        interval = float(form.get("interval", 50))
        raw_out = event_shuffle(filepath, interval=interval)

        outpath = os.path.join("app", raw_out) if not raw_out.startswith("app") else raw_out
        return outpath


    if algorithm == "frame_shuffle":
        return frame_shuffle(filepath)

    if algorithm == "frame_blur":
        sigma = float(form.get("sigma", 1.5))
        return frame_blur(filepath, sigma=sigma)

    raise ValueError("Invalid algorithm")
