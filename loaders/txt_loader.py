import numpy as np

def load_txt(path):
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    xs = data[:, 1].astype(int)
    ys = data[:, 2].astype(int)
    ps = data[:, 3].astype(int)

    return {
        "t": timestamps,
        "x": xs,
        "y": ys,
        "p": ps,
    }
