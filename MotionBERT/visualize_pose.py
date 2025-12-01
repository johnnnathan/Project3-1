import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

EDGES = [
    (0, 1), (1, 2), (2, 3),
    (1, 4), (4, 5), (5, 6),
    (1, 7), (7, 8), (8, 9),
    (7, 10), (10, 11), (11, 12)
]

def plot_single_frame(pose_seq, frame_idx=0):
    pose_2d = pose_seq[frame_idx]
    x = pose_2d[:, 0]
    y = -pose_2d[:, 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y)

    for i, j in EDGES:
        if i < len(x) and j < len(x):
            plt.plot([x[i], x[j]], [y[i], y[j]])

    plt.title(f"Pose frame {frame_idx}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def animate_pose_sequence(pose_seq, save_path="pose_sequence.gif", interval=200):
    T, J, _ = pose_seq.shape

    fig, ax = plt.subplots(figsize=(4, 4))

    def update(frame_idx):
        ax.clear()
        pose_2d = pose_seq[frame_idx]
        x = pose_2d[:, 0]
        y = -pose_2d[:, 1]

        ax.scatter(x, y)
        for i, j in EDGES:
            if i < len(x) and j < len(x):
                ax.plot([x[i], x[j]], [y[i], y[j]])

        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    ani.save(save_path, writer="pillow")
    print("Saved animation to", save_path)
