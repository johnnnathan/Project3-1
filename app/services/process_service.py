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

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
    APP_DIR = os.path.dirname(BASE_DIR)                  
    PROCESSED_DIR = os.path.join(APP_DIR, "processed")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if algorithm == "event_shuffle":
        interval = float(form.get("interval", 50))
        raw_out = event_shuffle(filepath, interval=interval)

        filename = os.path.basename(raw_out)
        outpath = os.path.join(PROCESSED_FOLDER, filename)

        data = np.loadtxt(raw_out)
        np.savetxt(outpath, data, delimiter=',')

        return outpath


    if algorithm == "frame_shuffle":
        return frame_shuffle(filepath)

    if algorithm == "frame_blur":
        sigma = float(form.get("sigma", 1.5))
        return frame_blur(filepath, sigma=sigma)

    raise ValueError("Invalid algorithm")

