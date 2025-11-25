import imageio

def save_gif(frames, output_path, fps=30):
    imgs = [(frame * 255).astype("uint8") for frame in frames]
    imageio.mimsave(output_path, imgs, fps=fps)
