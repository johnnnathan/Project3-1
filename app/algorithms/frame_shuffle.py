import numpy as np
import os
from algorithms.utils import make_frame_for_interval, shuffle_frames

def frame_shuffle(filepath, interval=50):
    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    chunk_size = make_frame_for_interval(interval, timestamps)

    shuffled = shuffle_frames(events, chunk_size)

    outpath = os.path.join("processed", "frame_shuffled_" + os.path.basename(filepath))
    np.savetxt(outpath, shuffled, fmt=["%.9f", "%d", "%d", "%d"], delimiter=' ')
    return outpath
