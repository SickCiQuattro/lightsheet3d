"""Experiment, not adopted: joint estimation of camera and sheet plane.

The pipeline first estimates the camera from the masters and then the plane from the sheet points. Here the two steps are
merged: the circles' reprojection residuals (pixels) and the distance of the sheet points from the
plane (mm, weighted by `WEIGHT` pixel/mm) are minimised together, so the sheet also constrains P and the distortion.

Result (see the README): on noisy synthetic data it is indistinguishable from the two-step estimate; on
real data it improves Corpo 6, worsens Corpo 5 and only partly reduces the sensitivity to a single
view. For this reason it is not in the package.

Usage: uv run python experiments/joint_calibration.py
"""
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

from lightsheet3d.calibration import CameraModel, distort_points, project, undistort_points
from lightsheet3d.laser import laser_centroids
from lightsheet3d.measure import level_to_plate, measure_peaks
from lightsheet3d.pipeline import PLATE_Z, calibrate_camera, calibrate_light_plane, scan_points, z_from_name
from lightsheet3d.solids import CORPO5, CORPO6, align, distances, nominal_curve
from lightsheet3d.synthetic import Scene, write_dataset

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (2130, 1420)
WEIGHT = 6.0  # pixels per mm: roughly the image scale on the master (1955 px focal length / 330 mm)
STRIDE = 4    # one sheet point every 4 rows: enough for the fit and much faster


def sheet_points(laser_dir, exclude=()):
    """Sheet pixels (distorted) and nominal height of the master for each image."""
    uv, z = [], []
    for path in sorted(Path(laser_dir).glob("*.jpg")):
        quota = z_from_name(path)
        if quota > PLATE_Z or quota in exclude:
            continue
        found = laser_centroids(cv.imread(str(path)))[::STRIDE]
        uv.append(found)
        z.append(np.full(len(found), float(quota)))
    return np.vstack(uv), np.concatenate(z)


def ray_plane_xy(P, uv, z):
    """(x, y) where each pixel's ray meets the plane z = constant, with a different z per point."""
    r0, r1 = P[0] - uv[:, :1] * P[2], P[1] - uv[:, 1:2] * P[2]
    a = np.stack([r0[:, :2], r1[:, :2]], axis=1)
    b = -np.stack([r0[:, 2] * z + r0[:, 3], r1[:, 2] * z + r1[:, 3]], axis=1)
    return np.linalg.solve(a, b[..., None])[..., 0]


def fit_joint(world, image, view_ids, sheet_uv, sheet_z, init, plane0, reference_view=PLATE_Z, n_radial=2):
    """(model, plane) minimising the circle and sheet residuals together; starts from the two-step estimate."""
    views, idx = np.unique(view_ids, return_inverse=True)
    free = np.arange(len(views)) != int(np.searchsorted(views, reference_view))
    dz = (views - reference_view).astype(np.float64)
    n_off = 2 * int(free.sum())

    def unpack(x):
        offsets = np.zeros((len(views), 2))
        offsets[free] = x[14 + n_radial:14 + n_radial + n_off].reshape(-1, 2)
        offsets -= dz[:, None] * ((dz[:, None] * offsets).sum(axis=0) / (dz**2).sum())  # same gauge as fit_camera
        return x[:12].reshape(3, 4), x[12:14], x[14:14 + n_radial], offsets, x[14 + n_radial + n_off:]

    def residuals(x):
        P, center, k, offsets, (a, b, c) = unpack(x)
        shifted = world.copy()
        shifted[:, :2] += offsets[idx]
        circles = (distort_points(project(P, shifted), center, k) - image).ravel()
        xy = ray_plane_xy(P, undistort_points(sheet_uv, center, k), sheet_z)
        sheet = WEIGHT * (a * xy[:, 0] + b * xy[:, 1] + c - sheet_z) / np.sqrt(a * a + b * b + 1)
        return np.concatenate([circles, sheet])

    x0 = np.concatenate([init.P.ravel() / np.abs(init.P).max(), init.center, init.k,
                         init.offsets[np.isin(init.views, views)][free].ravel(), plane0])
    P, center, k, offsets, plane = unpack(least_squares(residuals, x0, x_scale="jac", loss="soft_l1", f_scale=3.0).x)
    P = P / np.linalg.norm(P[2, :3])
    shifted = world.copy()
    shifted[:, :2] += offsets[idx]
    rms = float(np.sqrt((np.linalg.norm(distort_points(project(P, shifted), center, k) - image, axis=1) ** 2).mean()))
    return CameraModel(P, center, k, offsets, views, rms), plane


def measure(model, plane, image, solid, prominence=6.0):
    """(median distance from the nominal profile, peak heights) of a scan."""
    X = scan_points(image, model, plane)
    y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])
    sign, shift = align(solid, y, h)
    yn, _ = nominal_curve(solid, sign, shift)
    on_solid = (y > yn[1]) & (y < yn[-2])
    heights = [round(p.height, 1) for p in measure_peaks(y, h, prominence=prominence, min_separation=10)]
    return float(np.median(distances(solid, sign, shift, y[on_solid], h[on_solid]))), heights


def calibrate_both(target_dir, laser_dir, exclude=(), camera=None):
    """((model, plane) two-step, (model, plane) joint); `exclude` removes views from both."""
    camera = camera or calibrate_camera(target_dir)
    keep = camera.inliers & ~np.isin(camera.view_ids, exclude)
    separate_plane = calibrate_light_plane(laser_dir, camera.model, exclude=exclude).plane
    uv, z = sheet_points(laser_dir, exclude)
    joint = fit_joint(camera.world[keep], camera.image[keep], camera.view_ids[keep], uv, z, camera.model, separate_plane)
    return (camera.model, separate_plane), joint, camera


def real_data():
    print("== Real data")
    two_step, joint, camera = calibrate_both(ROOT / "data" / "camera_calibration", ROOT / "data" / "light_plane_calibration")
    for name, (model, plane) in (("two-step", two_step), ("joint", joint)):
        out = [measure(model, plane, ROOT / "data" / "scans" / f"{img}.jpg", solid) for img, solid in (("30o34", CORPO6), ("30o3", CORPO5))]
        print(f"{name:10s} Corpo 6: {out[0][0]:.3f} mm {out[0][1]} | Corpo 5: {out[1][0]:.3f} mm {out[1][1]}")
    print("Corpo 6, one view excluded at a time (peak heights):")
    for excluded in sorted({int(v) for v in camera.view_ids} - {PLATE_Z}):
        a, b, _ = calibrate_both(None, ROOT / "data" / "light_plane_calibration", exclude=(excluded,), camera=camera)
        heights = [measure(*pair, ROOT / "data" / "scans" / "30o34.jpg", CORPO6)[1] for pair in (a, b)]
        print(f"  without {excluded}: two-step {heights[0]} | joint {heights[1]}")


def synthetic_data(noise_levels=(0.0, 1.0, 2.0), seeds=3):
    print("== Synthetic data (Corpo 6)")
    with tempfile.TemporaryDirectory() as tmp:
        for noise in noise_levels:
            errors = {"two-step": [], "joint": []}
            for seed in range(seeds):
                root, scan = write_dataset(Scene(), CORPO6, Path(tmp) / f"{noise}_{seed}", noise_px=noise, seed=seed)
                two_step, joint, _ = calibrate_both(root / "camera_calibration", root / "light_plane_calibration")
                for name, pair in (("two-step", two_step), ("joint", joint)):
                    errors[name].append(measure(*pair, root / "scans" / scan, CORPO6, prominence=8.0)[0])
            print(f"noise {noise} px: " + " | ".join(f"{name} {np.mean(v):.3f} mm" for name, v in errors.items()))


if __name__ == "__main__":
    real_data()
    synthetic_data()
