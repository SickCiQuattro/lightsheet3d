from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def _homogeneous(points):
    return np.hstack([points, np.ones((len(points), 1))])


def project(P, world):
    """Project object points (n, 3) to pixels (n, 2) with the matrix P (3, 4)."""
    h = _homogeneous(np.asarray(world, np.float64)) @ P.T
    return h[:, :2] / h[:, 2:]


def reprojection_errors(P, world, image):
    """Euclidean distance in pixels between measured and reprojected points."""
    return np.linalg.norm(project(P, world) - np.asarray(image, np.float64).reshape(-1, 2), axis=1)


def _normalisation(points):
    """Similarity matrix that brings the points to mean 0 and standard deviation 1."""
    mean, scale = points.mean(axis=0), points.std()
    T = np.eye(points.shape[1] + 1)
    T[:-1, :-1] *= 1 / scale
    T[:-1, -1] = -mean / scale
    return T


def estimate_projection_matrix(world, image):
    """Normalised DLT (Hartley): estimate P (3, 4) from 3D-2D correspondences.

    The points must be normalised: with coordinates in mm and pixels the terms of Q
    differ by 5 orders of magnitude and the system is ill-conditioned.
    P is scaled so that |P[2, :3]| = 1, with the sign chosen so the points lie in front.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    T, U = _normalisation(world), _normalisation(image)
    Xn = _homogeneous(world) @ T.T
    xn = _homogeneous(image) @ U.T

    zeros = np.zeros_like(Xn)
    rows_u = np.hstack([Xn, zeros, -xn[:, :1] * Xn])
    rows_v = np.hstack([zeros, Xn, -xn[:, 1:2] * Xn])
    _, _, Vt = np.linalg.svd(np.vstack([rows_u, rows_v]))

    P = np.linalg.inv(U) @ Vt[-1].reshape(3, 4) @ T
    P /= np.linalg.norm(P[2, :3])
    return P if (_homogeneous(world) @ P[2]).mean() > 0 else -P


_RADIAL_SCALE = 1000.0  # px: makes k1, k2 numbers of order 0.1


def _radial_factor(points, center, k):
    r2 = (((points - center) / _RADIAL_SCALE) ** 2).sum(axis=1, keepdims=True)
    return 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k))


def distort_points(ideal, center, k):
    """Apply radial distortion to the ideal pixels (n, 2)."""
    return center + (ideal - center) * _radial_factor(ideal, center, k)


def undistort_points(distorted, center, k, iterations=20):
    """Invert `distort_points` by fixed-point iteration."""
    ideal = distorted.copy()
    for _ in range(iterations):
        ideal = center + (distorted - center) / _radial_factor(ideal, center, k)
    return ideal


@dataclass(frozen=True)
class CameraModel:
    P: np.ndarray        # (3, 4) ideal projection, |P[2, :3]| = 1
    center: np.ndarray   # (2,) centre of radial distortion, px
    k: np.ndarray        # radial coefficients
    offsets: np.ndarray  # (views, 2) estimated lateral shift of the master, mm
    views: np.ndarray    # (views,) identifier of each row of `offsets`
    rms: float           # px, on the calibration points

    def undistort(self, points):
        return undistort_points(points, self.center, self.k)


def fit_camera(world, image, view_ids, image_size, n_radial=2, reference_view=0, f_scale=3.0):
    """Joint estimate of P and radial distortion from the points of all views.

    The master does not move only along z between views (unstable setup): the
    lateral shift of each view is an estimated nuisance parameter,
    with the reference view fixed at zero defining the origin.
    The view identifiers are the z heights (mm).
    Robust cost function (soft L1) against badly detected centres.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    views, idx = np.unique(view_ids, return_inverse=True)
    ref = int(np.searchsorted(views, reference_view))
    free = np.arange(len(views)) != ref

    dz = (views - views[ref]).astype(np.float64)

    def unpack(x):
        offsets = np.zeros((len(views), 2))
        offsets[free] = x[14 + n_radial:].reshape(-1, 2)
        # A drift of the offsets that is linear in z is indistinguishable from a small tilt of the axis
        # in P (same residuals): it is fixed by removing it, i.e. assuming the master slides along z.
        offsets -= dz[:, None] * ((dz[:, None] * offsets).sum(axis=0) / (dz**2).sum())
        return x[:12].reshape(3, 4), x[12:14], x[14:14 + n_radial], offsets

    def residuals(x):
        P, center, k, offsets = unpack(x)
        shifted = world.copy()
        shifted[:, :2] += offsets[idx]
        return (distort_points(project(P, shifted), center, k) - image).ravel()

    P0 = estimate_projection_matrix(world, image)
    x0 = np.concatenate([P0.ravel() / np.abs(P0).max(), np.asarray(image_size) / 2,
                         np.zeros(n_radial), np.zeros(2 * free.sum())])
    sol = least_squares(residuals, x0, x_scale="jac", loss="soft_l1", f_scale=f_scale)

    P, center, k, offsets = unpack(sol.x)
    P = P / np.linalg.norm(P[2, :3])
    if (_homogeneous(world) @ P[2]).mean() < 0:
        P = -P
    rms = float(np.sqrt((residuals(sol.x) ** 2).reshape(-1, 2).sum(axis=1).mean()))
    return CameraModel(P, center, k, offsets, views, rms)


def model_residuals(model, world, image, view_ids):
    """Distance in pixels between measured points and points predicted by the model, with the view offsets."""
    shifted = np.asarray(world, np.float64).copy()
    shifted[:, :2] += model.offsets[np.searchsorted(model.views, view_ids)]
    predicted = distort_points(project(model.P, shifted), model.center, model.k)
    return np.linalg.norm(predicted - np.asarray(image, np.float64).reshape(-1, 2), axis=1)


def fit_camera_robust(world, image, view_ids, image_size, outlier_factor=10.0, **kwargs):
    """`fit_camera` with a second pass that discards wrong centres; returns (model, inliers).

    A badly detected centre (for example a spurious column at the image edge) can have
    tens of pixels of residual: the robust loss attenuates it but does not remove it. Points
    with residual above `outlier_factor` times the median are discarded and the fit is redone on the rest.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    model = fit_camera(world, image, view_ids, image_size, **kwargs)
    residuals = model_residuals(model, world, image, view_ids)
    inliers = residuals <= outlier_factor * np.median(residuals)
    if not inliers.all():
        model = fit_camera(world[inliers], image[inliers], np.asarray(view_ids)[inliers], image_size, **kwargs)
    return model, inliers
