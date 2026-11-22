"""Salvataggio della calibrazione (camera e piano della lama) in JSON, per riusarla senza ricalibrare."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import CameraModel

VERSION = 1


@dataclass(frozen=True)
class Calibration:
    model: CameraModel
    plane: np.ndarray  # (A, B, C) di z = A x + B y + C


def save_calibration(path, model, plane):
    data = {"version": VERSION, "P": model.P.tolist(), "center": model.center.tolist(), "k": model.k.tolist(),
            "offsets": model.offsets.tolist(), "views": model.views.tolist(), "rms": model.rms,
            "plane": np.asarray(plane, float).tolist()}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    if data.get("version") != VERSION:
        raise ValueError(f"{path}: versione {data.get('version')} non supportata (attesa {VERSION})")
    model = CameraModel(np.array(data["P"]), np.array(data["center"]), np.array(data["k"]),
                        np.array(data["offsets"]), np.array(data["views"]), float(data["rms"]))
    return Calibration(model, np.array(data["plane"]))
