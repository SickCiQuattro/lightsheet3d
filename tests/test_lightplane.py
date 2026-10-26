import numpy as np

from lightsheet3d.calibration import project
from lightsheet3d.lightplane import fit_plane, plane_errors, ray_plane_point, triangulate

K = np.array([[1250.0, 0, 590], [0, 1250.0, 410], [0, 0, 1]])
P = K @ np.hstack([np.eye(3), -np.array([[1.0], [168.0], [38.0]])])
# lama quasi parallela all'asse ottico: z = A x + B y + C con A grande
PLANE = np.array([5.0, 0.05, 330.0])


def points_on_plane(zs=range(300, 401, 10), n=150):
    pts = []
    for z in zs:
        y = np.linspace(-200, 200, n)
        x = (z - PLANE[2] - PLANE[1] * y) / PLANE[0]
        pts.append(np.column_stack([x, y, np.full(n, z)]))
    return np.vstack(pts)


def test_ray_plane_point_recovers_xy_at_known_z():
    world = np.column_stack([np.linspace(-100, 100, 20), np.linspace(-80, 80, 20), np.full(20, 330.0)])

    back = ray_plane_point(P, project(P, world), z=330.0)

    assert np.allclose(back, world, atol=1e-6)


def test_fit_plane_recovers_coefficients():
    pts = points_on_plane()

    assert np.allclose(fit_plane(pts), PLANE, atol=1e-6)


def test_plane_errors_are_orthogonal_distances_in_mm():
    pts = points_on_plane()
    normal = np.array([PLANE[0], PLANE[1], -1.0]) / np.linalg.norm([PLANE[0], PLANE[1], -1.0])

    on = plane_errors(PLANE, pts)
    off = plane_errors(PLANE, pts + 2.0 * normal)

    assert np.abs(on).max() < 1e-9
    assert np.allclose(np.abs(off), 2.0)


def test_triangulate_recovers_points_on_the_plane():
    world = points_on_plane(zs=[320, 350, 380])

    back = triangulate(P, project(P, world), PLANE)

    assert np.allclose(back, world, atol=1e-5)


def test_fit_from_back_projected_calibration_lines_then_triangulate_object():
    cal = points_on_plane()
    uv = project(P, cal)
    zs = cal[:, 2]
    xy = np.vstack([ray_plane_point(P, uv[zs == z], z)[:, :2] for z in np.unique(zs)])
    plane = fit_plane(np.column_stack([xy, zs]))

    obj = points_on_plane(zs=[345, 355])
    back = triangulate(P, project(P, obj), plane)

    assert np.allclose(back, obj, atol=1e-4)
