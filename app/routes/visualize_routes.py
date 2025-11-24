from flask import Blueprint, request
from services.visualize_service import visualize_uploaded_events

visualize_routes = Blueprint("visualize_routes", __name__)

@visualize_routes.route("/visualize", methods=["POST"])
def visualize_file():
    file = request.files.get("file")
    if file is None:
        return "No file uploaded", 400

    gif_url = visualize_uploaded_events(file, request.form)
    return {"gif_url": gif_url}
