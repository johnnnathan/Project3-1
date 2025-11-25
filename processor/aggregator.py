import numpy as np

def aggregate_events_to_frames(events, resolution, dt):
    t = events["t"]
    x = events["x"]
    y = events["y"]
    p = events["p"]

    t_min, t_max = t.min(), t.max()
    frames = []
    window_starts = []

    for t0 in np.arange(t_min, t_max, dt):
        mask = (t >= t0) & (t < t0 + dt)
        frame = np.zeros(resolution)

        frame[y[mask], x[mask]] = p[mask]
        frames.append(frame)
        window_starts.append(t0)

    return np.array(frames), np.array(window_starts)
