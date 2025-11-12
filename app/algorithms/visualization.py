
import numpy as np
import os
from .utils import make_frame_for_interval, create_frames, save_animation

def visualize_events(filepath, interval=50, output_folder="processed"):
    """
    Generate a GIF from an event file and save it.

    Args:
        filepath: Path to the event .txt file
        interval: Time window (ms) for creating frames
        output_folder: Where to save the GIF

    Returns:
        Path to the generated GIF
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load events
    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    chunk_size = make_frame_for_interval(interval, timestamps)

    # Create frames
    frames = create_frames(events, chunk_size)

    # Save GIF
    gif_name = f"visual_{os.path.basename(filepath)}.gif"
    gif_path = os.path.join(output_folder, gif_name)
    save_animation(frames, gif_path, chunk_size, title_prefix="Visualization")

    return gif_path
