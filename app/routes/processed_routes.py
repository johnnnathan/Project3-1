from flask import Blueprint, send_from_directory
import os

PROCESSED_FOLDER = "processed"

processed_routes = Blueprint("processed_routes", __name__)

@processed_routes.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)
