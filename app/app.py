from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import os
import numpy as np
import io
import base64
from PIL import Image

# Import algorithms
from algorithms.event_shuffle import event_shuffle
from algorithms.frame_shuffle import frame_shuffle
from algorithms.frame_blur import frame_blur
from algorithms.utils import make_frame_for_interval, create_frames, save_animation

from algorithms.events2frame import events_to_frame
from algorithms.facedetect import detect_faces_in_frame, blur_faces_in_frame
from algorithms.labeler import predict_with_model

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# FRONTEND PAGE
# ----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------
# API ENDPOINTS MATCHING FRONTEND
# ----------------------------------------------------------

@app.route("/api/event_shuffle", methods=["POST"])
def api_event_shuffle():
    file = request.files["file"]
    interval = float(request.form.get("interval", 50))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    outpath = event_shuffle(filepath, interval)
    outpath = os.path.join(BASE_DIR, outpath)
    return send_file(outpath, as_attachment=True)


@app.route("/api/frame_shuffle", methods=["POST"])
def api_frame_shuffle():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    outpath = frame_shuffle(filepath)
    outpath = os.path.join(BASE_DIR, outpath)
    return send_file(outpath, as_attachment=True)


@app.route("/api/frame_blur", methods=["POST"])
def api_frame_blur():
    file = request.files["file"]
    sigma = float(request.form.get("sigma", 1.5))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    outpath = frame_blur(filepath, sigma=sigma)
    outpath = os.path.join(BASE_DIR, outpath)
    return send_file(outpath, as_attachment=True)


@app.route("/api/visualize", methods=["POST"])
def api_visualize():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    
    interval = float(request.form.get("interval", 50))
    chunk_size = make_frame_for_interval(interval, timestamps)

    frames = create_frames(events, chunk_size)

    gif_name = f"visual_{file.filename}.gif"
    gif_path = os.path.join(PROCESSED_FOLDER, gif_name)
    save_animation(frames, gif_path, chunk_size)

    return jsonify({"gif_url": f"/processed/{gif_name}"})

@app.route("/api/v2e", methods=["POST"])
def api_v2e():
    print("Run v2e function")
    return "This feature has not been implemented yet."

@app.route("/api/data-labeling", methods=["POST"])
def api_labeling():
    print("Run labeling function")
    return "This feature has not been implemented yet."


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


# ----------------------------------------------------------
# LIVE JSON API (for streaming applications)
# ----------------------------------------------------------
@app.route('/api/process_events', methods=['POST'])
def api_process_events():
    data = request.get_json(force=True)

    events = data.get('events', [])
    width = int(data.get('width', 320))
    height = int(data.get('height', 240))
    blur_faces = bool(data.get('blur_faces', True))
    blur_radius = int(data.get('blur_radius', 15))

    # Convert events → image
    frame = events_to_frame(events, width=width, height=height)

    # Face detection
    faces = detect_faces_in_frame(frame)

    if blur_faces and faces:
        rgb = np.stack([frame]*3, axis=2)
        rgb = blur_faces_in_frame(rgb, faces, blur_radius)
        out_frame = rgb
    else:
        out_frame = np.stack([frame]*3, axis=2)

    # Encode PNG
    pil_img = Image.fromarray(out_frame.astype('uint8'))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    # Label prediction
    label = predict_with_model(out_frame)

    return jsonify({
        "image_png_b64": "data:image/png;base64," + png_b64,
        "faces": faces,
        "label": label
    })


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
