from flask import Blueprint, request, send_file
from services.process_service import process_uploaded_file

process_routes = Blueprint("process_routes", __name__)

@process_routes.route("/process/<algorithm>", methods=["POST"])
def process_file(algorithm):
    file = request.files.get("file")
    if file is None:
        return "No file uploaded", 400

    outpath = process_uploaded_file(file, algorithm, request.form)
    return send_file(outpath, as_attachment=True)
