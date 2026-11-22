"""Sensibilità del risultato alla calibrazione: si ricalibra escludendo una vista alla volta (jackknife)."""
from dataclasses import dataclass

import numpy as np

from .calibration import fit_camera
from .measure import level_to_plate, measure_peaks
from .pipeline import PLATE_Z, calibrate_light_plane, scan_points
from .solids import align, distances, nominal_curve


@dataclass(frozen=True)
class Run:
    excluded: int | None   # quota della vista esclusa; None = calibrazione completa
    plane: np.ndarray      # (A, B, C)
    camera_rms: float      # px
    median: float          # mm, distanza mediana dal profilo nominale (punti sul solido)
    heights: np.ndarray    # mm, altezza dei picchi; nan se non se ne trovano quanti ne ha il solido


def _run(model, laser_dir, image, solid, excluded, prominence):
    light = calibrate_light_plane(laser_dir, model, exclude=() if excluded is None else (excluded,))
    X = scan_points(image, model, light.plane)
    y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])
    sign, shift = align(solid, y, h)
    yn, _ = nominal_curve(solid, sign, shift)
    on_solid = (y > yn[1]) & (y < yn[-2])
    dist = distances(solid, sign, shift, y[on_solid], h[on_solid])
    peaks = measure_peaks(y, h, prominence=prominence, min_separation=10)
    heights = np.array([p.height for p in peaks]) if len(peaks) == len(solid.peaks()) else np.full(len(solid.peaks()), np.nan)
    return Run(excluded, light.plane, model.rms, float(np.median(dist)), heights)


def leave_one_view_out(camera, laser_dir, image, solid, prominence=6.0, image_size=(2130, 1420)):
    """Calibrazione completa più una per ogni vista del master (tranne quella di riferimento) esclusa.

    `camera` è il risultato di `calibrate_camera`: i centri già rilevati si riusano, quindi ogni
    ricalibrazione costa il solo fit. La vista esclusa lo è dalla camera e dal piano della lama.
    """
    runs = [_run(camera.model, laser_dir, image, solid, None, prominence)]
    for z in sorted({int(v) for v in camera.view_ids} - {PLATE_Z}):
        keep = (camera.view_ids != z) & camera.inliers
        model = fit_camera(camera.world[keep], camera.image[keep], camera.view_ids[keep], image_size, reference_view=PLATE_Z)
        runs.append(_run(model, laser_dir, image, solid, z, prominence))
    return runs


def jackknife_se(values):
    """Errore standard jackknife di una statistica calcolata lasciando fuori un elemento alla volta."""
    values = np.asarray(values, np.float64)
    n = len(values)
    return float(np.sqrt((n - 1) / n * ((values - values.mean()) ** 2).sum()))
