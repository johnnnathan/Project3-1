from flask import Flask, render_template, request, send_file, redirect, url_for, send_from_directory
import os
from algorithms.event_shuffle import event_shuffle
from algorithms.frame_shuffle import frame_shuffle
from algorithms.frame_blur import frame_blur
from algorithms.utils import make_frame_for_interval, create_frames, save_animation
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process/<algorithm>", methods=["POST"])
def process_file(algorithm):
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if algorithm == "event_shuffle":
        interval = float(request.form.get("interval", 50))  # <-- new line
        outpath = event_shuffle(filepath, interval=interval)

    elif algorithm == "frame_shuffle":
        outpath = frame_shuffle(filepath)

    elif algorithm == "frame_blur":
        sigma = float(request.form.get("sigma", 1.5))
        outpath = frame_blur(filepath, sigma=sigma)

    else:
        return "Invalid algorithm", 400

    return send_file(outpath, as_attachment=True)

@app.route("/visualize", methods=["POST"])
def visualize_file():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load events
    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    interval = float(request.form.get("interval", 50))
    chunk_size = make_frame_for_interval(interval, timestamps)

    # Create frames and save GIF
    frames = create_frames(events, chunk_size)
    gif_name = f"visual_{os.path.basename(file.filename)}.gif"
    gif_path = os.path.join(PROCESSED_FOLDER, gif_name)
    save_animation(frames, gif_path, chunk_size, title_prefix="Visualization")

    # Return GIF path for embedding
    return {"gif_url": f"/processed/{gif_name}"}

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
