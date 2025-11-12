import numpy as np
from scipy.ndimage import gaussian_filter

# ----------- CORE SHARED UTILITIES -----------

def make_frame_for_interval(interval_ms, timestamps):
    """Compute number of events per frame based on average event rate."""
    dt = np.diff(timestamps)
    avg_event_rate = 1 / np.mean(dt)
    events_in_interval = int(avg_event_rate * (0.001 * interval_ms))
    return max(1, events_in_interval)


def create_frames(events, chunk_size, width=240, height=180):
    """Convert event stream into frames (used internally by algorithms)."""
    t, x, y, p = (
        events[:, 0],
        events[:, 1].astype(int),
        events[:, 2].astype(int),
        np.where(events[:, 3] == 1, 1, -1),
    )
    frames = []
    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        frame = np.zeros((height, width))
        frame[y[start:end], x[start:end]] = p[start:end]
        frames.append(frame)
    return frames

def save_animation(frames, filename, chunk_size, title_prefix=""):
    import imageio
    images = [frame.astype(np.uint8) for frame in frames]
    imageio.mimsave(filename, images, duration=chunk_size/1000)


def shuffle_events(events, chunk_size):
    """Shuffle events within each temporal chunk."""
    shuffled = np.zeros_like(events)
    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        chunk = events[start:end]
        np.random.shuffle(chunk)
        shuffled[start:end] = chunk
    return shuffled


def shuffle_frames(events, chunk_size):
    """Reorder entire frame segments."""
    indices = np.arange(0, len(events), chunk_size)
    np.random.shuffle(indices)

    reordered = np.zeros_like(events)
    pos = 0
    for start in indices:
        end = min(start + chunk_size, len(events))
        reordered[pos:pos+(end-start)] = events[start:end]
        pos += end - start

    return reordered


def blur_events(events, chunk_size, sigma=1.5, width=240, height=180):
    """
    Applies spatial Gaussian blur to each frame worth of events.
    The output is flattened back to an event list (approximation).
    """
    from scipy.ndimage import gaussian_filter

    t, x, y, p = (
        events[:, 0],
        events[:, 1].astype(int),
        events[:, 2].astype(int),
        np.where(events[:, 3] == 1, 1, -1),
    )

    blurred_events = []

    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        frame = np.zeros((height, width))
        frame[y[start:end], x[start:end]] = p[start:end]

        blurred_frame = gaussian_filter(frame, sigma=sigma)

        ys, xs = np.where(np.abs(blurred_frame) > 0.1)  # keep meaningful events
        ps = np.sign(blurred_frame[ys, xs])
        ts = np.full(len(xs), np.mean(t[start:end]))  # approximate timestamp

        blurred_events.append(np.column_stack((ts, xs, ys, ps)))

    return np.vstack(blurred_events)
