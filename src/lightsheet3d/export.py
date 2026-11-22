"""Export of the measured points in formats readable by MeshLab, CloudCompare and Open3D."""
from pathlib import Path

import numpy as np


def write_ply(path, points, colors=None):
    """Write a point cloud (n, 3) as ASCII PLY; `colors` (n, 3) uint8 is optional.

    ASCII rather than binary: the files are a few tens of kB and can be read with a text editor.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
              "property float x", "property float y", "property float z"]
    rows = points
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape:
            raise ValueError(f"colors {colors.shape} does not match points {points.shape}")
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with Path(path).open("w") as f:
        f.write("\n".join(header) + "\n")
        for i, p in enumerate(rows):
            line = f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
            if colors is not None:
                line += " {} {} {}".format(*(int(c) for c in colors[i]))
            f.write(line + "\n")


def read_ply(path):
    """Read an ASCII PLY written by `write_ply`: (points (n, 3), colors (n, 3) or None)."""
    lines = Path(path).read_text().splitlines()
    end = lines.index("end_header")
    n = next(int(line.split()[-1]) for line in lines[:end] if line.startswith("element vertex"))
    data = np.array([line.split() for line in lines[end + 1:end + 1 + n]], np.float64).reshape(n, -1)
    return data[:, :3], (data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else None)


def distance_colors(distance, vmax=2.0):
    """RGB colors (uint8) from green (0 mm) to red (`vmax` mm or more) for a distance in mm."""
    t = np.clip(np.asarray(distance, np.float64) / vmax, 0, 1)
    return np.round(np.column_stack([255 * t, 255 * (1 - t), np.zeros_like(t)])).astype(np.uint8)
