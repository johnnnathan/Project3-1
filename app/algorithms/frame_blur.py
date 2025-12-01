import numpy as np
import os
from algorithms.utils import make_frame_for_interval, blur_events

def frame_blur(filepath, interval=50, sigma=1.5):
    events = np.loadtxt(filepath)
    timestamps = events[:, 0]
    chunk_size = make_frame_for_interval(interval, timestamps)

    blurred = blur_events(events, chunk_size, sigma=sigma)

    outpath = os.path.join("processed", "frame_blurred_" + os.path.basename(filepath))
    np.savetxt(outpath, blurred, fmt=["%.9f", "%d", "%d", "%d"], delimiter=' ')
    return outpath
