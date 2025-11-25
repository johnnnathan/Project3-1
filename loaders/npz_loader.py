import numpy as np

def load_npz(path):
    data = np.load(path)

    required = ["t", "x", "y", "p"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in NPZ file.")

    return {
        "t": data["t"],
        "x": data["x"],
        "y": data["y"],
        "p": data["p"],
    }
