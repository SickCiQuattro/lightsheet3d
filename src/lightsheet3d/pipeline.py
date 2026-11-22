"""From raw images to measurements: camera calibration, sheet plane, scan."""
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import CameraModel, fit_camera_robust
from .grid import build_object_points, find_first_valid_circles_grid, orient_grid
from .laser import laser_centroids
from .lightplane import fit_plane, ray_plane_point, triangulate

Z_BASE = 50          # mm, height of the support base
Z_STEP = 10          # mm per unit of the number in the file name
PATTERN_MAX = (21, 14)  # maximum columns, rows of the circle grid
PATTERN_MIN = 9
PLATE_Z = 350        # mm, plane the solids rest on ("30..." images)


def z_from_name(path):
    """z (mm) from the file name, for example 27l.jpg -> 27 * 10 + 50."""
    return int(Path(path).name[:2]) * Z_STEP + Z_BASE


@dataclass(frozen=True)
class CameraCalibration:
    model: CameraModel
    world: np.ndarray     # (n, 3) object points of all views
    image: np.ndarray     # (n, 2) measured centres, raw pixels
    view_ids: np.ndarray  # (n,) z of the view of each point
    skipped: tuple        # files without a pattern or without a recognisable origin
    inliers: np.ndarray   # (n,) False for the centres discarded as wrong before the final fit


def calibrate_camera(target_dir, z_max=PLATE_Z, pitch=20.0, image_size=(2130, 1420), z_of=z_from_name,
                     pattern_max=PATTERN_MAX, pattern_min=PATTERN_MIN):
    """Camera model from the calibration masters with z <= z_max (views beyond the main master
    are not used: the setup moved between shots)."""
    world, image, view_ids, skipped = [], [], [], []
    for path in sorted(Path(target_dir).glob("*.jpg")):
        z = z_of(path)
        if z > z_max:
            continue
        gray = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        pattern, centers = find_first_valid_circles_grid(gray, *pattern_max, min_size=pattern_min)
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
    model, inliers = fit_camera_robust(world, image, view_ids, image_size, reference_view=z_max)
    return CameraCalibration(model, world, image, view_ids, tuple(skipped), inliers)


@dataclass(frozen=True)
class LightPlaneCalibration:
    plane: np.ndarray          # (A, B, C) of z = A x + B y + C, fitted on the calibration points only
    points: np.ndarray         # (n, 3) points of the sheet in object space
    is_validation: np.ndarray  # (n,) True for the points excluded from the fit
    view_ids: np.ndarray       # (n,) nominal height (from the file name) of the view of each point


def calibrate_light_plane(lama_dir, model, z_max=PLATE_Z, validation_fraction=0.1, seed=0, exclude=(), extract=laser_centroids,
                           z_of=z_from_name):
    """Sheet plane from the points of the images with the sheet on the master at known z (views in `exclude` left out)."""
    points, view_ids = [], []
    for path in sorted(Path(lama_dir).glob("*.jpg")):
        z = z_of(path)
        if z > z_max or z in exclude:
            continue
        uv = model.undistort(extract(cv.imread(str(path))))
        points.append(ray_plane_point(model.P, uv, z))
        view_ids.append(np.full(len(uv), z))
    points, view_ids = np.vstack(points), np.concatenate(view_ids)
    is_validation = np.zeros(len(points), bool)
    is_validation[np.random.default_rng(seed).permutation(len(points))[:int(validation_fraction * len(points))]] = True
    return LightPlaneCalibration(fit_plane(points[~is_validation]), points, is_validation, view_ids)


def scan_points(image_path, model, plane, extract=laser_centroids):
    """Object points (n, 3) in mm of the profile lit by the sheet in a raw image."""
    uv = model.undistort(extract(cv.imread(str(image_path))))
    return triangulate(model.P, uv, plane)
