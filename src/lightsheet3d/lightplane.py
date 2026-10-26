import numpy as np


def _ray_rows(P, uv):
    """Righe (n, 2, 4) del sistema che lega il punto 3D al pixel: (P0 - u P2, P1 - v P2)."""
    return np.stack([P[0] - uv[:, :1] * P[2], P[1] - uv[:, 1:2] * P[2]], axis=1)


def ray_plane_point(P, uv, z):
    """Punti oggetto (n, 3) dove i raggi dei pixel `uv` incontrano il piano z = costante."""
    rows = _ray_rows(P, np.asarray(uv, np.float64))
    xy = np.linalg.solve(rows[:, :, :2], -(rows[:, :, 2] * z + rows[:, :, 3])[..., None])[..., 0]
    return np.column_stack([xy, np.full(len(xy), float(z))])


def fit_plane(points):
    """Minimi quadrati di z = A x + B y + C su punti (n, 3); ritorna (A, B, C)."""
    design = np.column_stack([points[:, :2], np.ones(len(points))])
    return np.linalg.lstsq(design, points[:, 2], rcond=None)[0]


def plane_errors(plane, points):
    """Distanza con segno in mm dei punti dal piano z = A x + B y + C."""
    a, b, c = plane
    return (a * points[:, 0] + b * points[:, 1] + c - points[:, 2]) / np.sqrt(a * a + b * b + 1)


def triangulate(P, uv, plane):
    """Punti oggetto (n, 3) all'intersezione del raggio di ogni pixel con il piano della lama."""
    a, b, c = plane
    rows = _ray_rows(P, np.asarray(uv, np.float64))
    system = np.concatenate([rows[:, :, :3], np.broadcast_to([a, b, -1.0], (len(rows), 1, 3))], axis=1)
    rhs = np.concatenate([-rows[:, :, 3], np.full((len(rows), 1), -c)], axis=1)
    return np.linalg.solve(system, rhs[..., None])[..., 0]
