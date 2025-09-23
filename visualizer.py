import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

events = np.loadtxt("events.txt")
print("Number of events:", len(events))
t = events[:, 0]
x = events[:, 1].astype(int)
y = events[:, 2].astype(int)
p = np.where(events[:, 3]==1, 1, -1)

#formula to get 50 ms of frames
#take average delta t (difference between two timestamps)
#1 over mean of average
#multiply by 0.05
#50 ms is a good compromise
def events_per_50ms(timestamps):
    dt = np.diff(timestamps)
    avg_event_rate = 1 / np.mean(dt)
    events_in_50ms = int(avg_event_rate * 0.05)
    return max(1, events_in_50ms)

width, height = 240, 180
n = 23126288
chunk_size = events_per_50ms(t)

#we go up until N, we start at 0,
#create the first 'chunks' event,
#display it
#make a new empty frame
#display the next 'chunks' event
#repeat until reached N

frames = []
for start in range(0, n, chunk_size):
    end = min(start + chunk_size, n)
    frame = np.zeros((height, width))
    x_chunk = x[start:end]
    y_chunk = y[start:end]
    p_chunk = p[start:end]
    frame[y_chunk, x_chunk] = p_chunk
    frames.append(frame)

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