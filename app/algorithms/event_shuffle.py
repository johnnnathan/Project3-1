import numpy as np
import os
from algorithms.utils import make_frame_for_interval, shuffle_events

def event_shuffle(filepath, interval=50):
    """
    Shuffle events within each chunk to preserve privacy while keeping structure.

    Args:
        filepath: path to input event file
        interval: time window (ms) to define chunk size
    """
    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    chunk_size = make_frame_for_interval(interval, timestamps)

    shuffled = shuffle_events(events, chunk_size)

    outpath = os.path.join("processed", f"event_shuffled_{interval}ms_" + os.path.basename(filepath))
    np.savetxt(outpath, shuffled, fmt=["%.9f", "%d", "%d", "%d"], delimiter=' ')
    return outpath
