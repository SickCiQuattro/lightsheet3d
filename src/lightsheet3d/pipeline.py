"""Dalle immagini grezze alle misure: calibrazione camera, piano della lama, scansione."""
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import CameraModel, fit_camera
from .grid import build_object_points, find_first_valid_circles_grid, orient_grid
from .laser import laser_centroids
from .lightplane import fit_plane, ray_plane_point, triangulate

Z_BASE = 50          # mm, altezza della base del supporto
Z_STEP = 10          # mm per unità del numero nel nome file
PATTERN_MAX = (21, 14)  # colonne, righe massime della griglia di cerchi
PATTERN_MIN = 9
PLATE_Z = 350        # mm, piano di appoggio dei solidi (immagini "30...")


def z_from_name(path):
    """z (mm) dal nome file, ad esempio 27l.jpg -> 27 * 10 + 50."""
    return int(Path(path).name[:2]) * Z_STEP + Z_BASE


@dataclass(frozen=True)
class CameraCalibration:
    model: CameraModel
    world: np.ndarray     # (n, 3) punti oggetto di tutte le viste
    image: np.ndarray     # (n, 2) centri misurati, pixel grezzi
    view_ids: np.ndarray  # (n,) z della vista di ogni punto
    skipped: tuple        # file senza pattern o senza origine riconoscibile


def calibrate_camera(target_dir, z_max=PLATE_Z, pitch=20.0, image_size=(2130, 1420)):
    """Modello camera dai master di taratura con z <= z_max (le viste oltre il master principale
    non sono usate: il setup si è mosso fra gli scatti)."""
    world, image, view_ids, skipped = [], [], [], []
    for path in sorted(Path(target_dir).glob("*.jpg")):
        z = z_from_name(path)
        if z > z_max:
            continue
        gray = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        pattern, centers = find_first_valid_circles_grid(gray, *PATTERN_MAX, min_size=PATTERN_MIN)
        try:
            centers, origin = orient_grid(gray, pattern, centers) if pattern else (None, None)
        except ValueError:
            origin = None
        if origin is None:
            skipped.append(path.name)
            continue
        world.append(build_object_points(pattern, origin, pitch, z))
        image.append(centers)
        view_ids += [z] * len(centers)
    world, image, view_ids = np.vstack(world), np.vstack(image), np.array(view_ids)
    model = fit_camera(world, image, view_ids, image_size, reference_view=z_max)
    return CameraCalibration(model, world, image, view_ids, tuple(skipped))


@dataclass(frozen=True)
class LightPlaneCalibration:
    plane: np.ndarray          # (A, B, C) di z = A x + B y + C, fit sui soli punti di calibrazione
    points: np.ndarray         # (n, 3) punti della lama nello spazio oggetto
    is_validation: np.ndarray  # (n,) True per i punti esclusi dal fit
    view_ids: np.ndarray       # (n,) quota nominale (dal nome file) della vista di ogni punto


def calibrate_light_plane(lama_dir, model, z_max=PLATE_Z, validation_fraction=0.1, seed=0):
    """Piano della lama dai punti delle immagini con la lama sul master a z noto."""
    points, view_ids = [], []
    for path in sorted(Path(lama_dir).glob("*.jpg")):
        z = z_from_name(path)
        if z > z_max:
            continue
        uv = model.undistort(laser_centroids(cv.imread(str(path))))
        points.append(ray_plane_point(model.P, uv, z))
        view_ids.append(np.full(len(uv), z))
    points, view_ids = np.vstack(points), np.concatenate(view_ids)
    is_validation = np.zeros(len(points), bool)
    is_validation[np.random.default_rng(seed).permutation(len(points))[:int(validation_fraction * len(points))]] = True
    return LightPlaneCalibration(fit_plane(points[~is_validation]), points, is_validation, view_ids)


def scan_points(image_path, model, plane):
    """Punti oggetto (n, 3) in mm del profilo illuminato dalla lama in un'immagine grezza."""
    uv = model.undistort(laser_centroids(cv.imread(str(image_path))))
    return triangulate(model.P, uv, plane)
