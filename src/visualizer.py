# ----------------IMPORTS--------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ----------------VARIABLE DECLARATIONS--------------------------
events = np.loadtxt("events.txt")
n = len(events)
print("Number of events:", n)
interval = 50
width, height = 240, 180




t = events[:, 0]
x = events[:, 1].astype(int)
y = events[:, 2].astype(int)
p = np.where(events[:, 3]==1, 1, -1)

# ----------------FUNCTIONS--------------------------

#formula to get 50 ms of frames
#take average delta t (difference between two timestamps)
#1 over mean of average
#multiply by 0.05
#50 ms is a good compromise
def make_frame_for_interval(interval, timestamps):
    dt = np.diff(timestamps)
    avg_event_rate = 1 / np.mean(dt)
    events_in_interval= int(avg_event_rate * (0.001 * interval))
    return max(1, events_in_interval)




#we go up until N, we start at 0,
#create the first 'chunks' event,
#display it
#make a new empty frame
#display the next 'chunks' event
#repeat until reached N

def create_frames():
    frames = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        frame = np.zeros((height, width))
        x_chunk = x[start:end]
        y_chunk = y[start:end]
        p_chunk = p[start:end]
        frame[y_chunk, x_chunk] = p_chunk
        frames.append(frame)
    return frames

def plot(frames):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], cmap='bwr', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(img, ax=ax, label="Polarity")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    def update(i):
        img.set_data(frames[i])
        ax.set_title(f"Events {i*chunk_size} → {(i+1)*chunk_size}")
        return [img]


    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
    ani.save("event_animation.gif", writer='pillow', fps=5)
    plt.show()

#Shuffle events within each chunk to preserve order, while maintaining privacy 
def shuffle_events(events, chunk_size):
    shuffled = np.zeros_like(events)
    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        chunk = events[start:end]
        np.random.shuffle(chunk)
        shuffled[start:end] = chunk
    return shuffled

#Shuffle the order of the frames, after the frames have been formed
def shuffle_frames(frames):
    frames_copy = frames.copy()
    np.random.shuffle(frames_copy)
    return frames_copy


if __name__ == "__main__":
    chunk_size = make_frame_for_interval(interval, t)

    #shuffle the events
    #events = shuffle_events(events, chunk_size)

    frames = create_frames();

    #shuffle the frames
    #frames = shuffle_frames(frames)

    plot(frames)
