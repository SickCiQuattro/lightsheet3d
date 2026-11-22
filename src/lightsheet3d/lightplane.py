import numpy as np


def _ray_rows(P, uv):
    """Rows (n, 2, 4) of the system relating the 3D point to the pixel: (P0 - u P2, P1 - v P2)."""
    return np.stack([P[0] - uv[:, :1] * P[2], P[1] - uv[:, 1:2] * P[2]], axis=1)


def ray_plane_point(P, uv, z):
    """Object points (n, 3) where the rays of pixels `uv` meet the plane z = constant."""
    rows = _ray_rows(P, np.asarray(uv, np.float64))
    xy = np.linalg.solve(rows[:, :, :2], -(rows[:, :, 2] * z + rows[:, :, 3])[..., None])[..., 0]
    return np.column_stack([xy, np.full(len(xy), float(z))])


def fit_plane(points):
    """Least squares fit of z = A x + B y + C to points (n, 3); returns (A, B, C)."""
    design = np.column_stack([points[:, :2], np.ones(len(points))])
    return np.linalg.lstsq(design, points[:, 2], rcond=None)[0]


def plane_errors(plane, points):
    """Signed distance in mm of the points from the plane z = A x + B y + C."""
    a, b, c = plane
    return (a * points[:, 0] + b * points[:, 1] + c - points[:, 2]) / np.sqrt(a * a + b * b + 1)


def triangulate(P, uv, plane):
    """Object points (n, 3) at the intersection of each pixel's ray with the sheet plane."""
    a, b, c = plane
    rows = _ray_rows(P, np.asarray(uv, np.float64))
    system = np.concatenate([rows[:, :, :3], np.broadcast_to([a, b, -1.0], (len(rows), 1, 3))], axis=1)
    rhs = np.concatenate([-rows[:, :, 3], np.full((len(rows), 1), -c)], axis=1)
    return np.linalg.solve(system, rhs[..., None])[..., 0]
