import numpy as np

from lightsheet3d.calibration import (
    distort_points,
    estimate_projection_matrix,
    fit_camera,
    fit_camera_robust,
    model_residuals,
    project,
    reprojection_errors,
    undistort_points,
)

# Scena sintetica con la geometria del setup reale: z è la distanza lungo
# l'asse ottico, i piani stanno a z = 300..350 mm e la telecamera è a z ~ 38 mm,
# quindi l'origine del mondo sta dietro il piano camera (profondità < 0).
K = np.array([[1250.0, 0, 590], [0, 1250.0, 410], [0, 0, 1]])
R = np.eye(3)
CAMERA = np.array([1.0, 168.0, 38.0])
P_TRUE = K @ np.hstack([R, (-R @ CAMERA)[:, None]])


def make_points(seed=0):
    xs, ys = np.meshgrid(np.arange(-200, 201, 20), np.arange(-120, 121, 20))
    world = np.vstack([
        np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])
        for z in range(300, 351, 10)
    ]).astype(np.float32)
    return world, project(P_TRUE, world)


def test_recovers_projection_from_float32_points():
    world, image = make_points()

    P = estimate_projection_matrix(world, image.astype(np.float32))

    assert reprojection_errors(P, world, image).max() < 1e-3


def test_noisy_points_reproject_within_noise():
    # con m12 = 1 (profondità positiva all'origine) l'errore qui sale a ~1.5 px
    world, image = make_points()
    noisy = image + np.random.default_rng(0).normal(0, 2.0, image.shape)

    P = estimate_projection_matrix(world, noisy)

    assert reprojection_errors(P, world, image).mean() < 0.25


def test_projection_matrix_is_normalised_with_points_in_front():
    world, image = make_points()

    P = estimate_projection_matrix(world, image)

    assert np.isclose(np.linalg.norm(P[2, :3]), 1.0)
    assert (np.hstack([world, np.ones((len(world), 1))]) @ P[2] > 0).all()


# --- modello congiunto: P + distorsione radiale + offset laterale per vista ---
CENTER = np.array([850.0, 620.0])
K_RADIAL = np.array([-0.10, 0.008])


def make_views(seed=1, noise=0.3, offset_mm=6.0):
    """Sei piani a z = 300..350; ogni vista ha il master spostato lateralmente."""
    rng = np.random.default_rng(seed)
    world, _ = make_points()
    view_ids = np.repeat(np.arange(6), len(world) // 6)
    offsets = rng.uniform(-offset_mm, offset_mm, (6, 2))
    offsets[0] = 0  # vista di riferimento
    shifted = world.astype(np.float64).copy()
    shifted[:, :2] += offsets[view_ids]
    image = distort_points(project(P_TRUE, shifted), CENTER, K_RADIAL)
    return world.astype(np.float64), image + rng.normal(0, noise, image.shape), view_ids, offsets


def test_undistort_inverts_distort():
    pts = np.array([[100.0, 80.0], [850.0, 620.0], [1650.0, 1100.0]])

    back = undistort_points(distort_points(pts, CENTER, K_RADIAL), CENTER, K_RADIAL)

    assert np.abs(back - pts).max() < 1e-3


def test_joint_fit_reprojects_within_noise_despite_lateral_offsets():
    world, image, view_ids, _ = make_views()

    model = fit_camera(world, image, view_ids, image_size=(1700, 1240))

    assert model.rms < 0.6


def test_joint_fit_recovers_reference_view_geometry():
    world, image, view_ids, _ = make_views()
    model = fit_camera(world, image, view_ids, image_size=(1700, 1240))

    ref = view_ids == 0  # offset nullo: ciò che il modello predice è verificabile
    truth = distort_points(project(P_TRUE, world[ref]), CENTER, K_RADIAL)
    pred = distort_points(project(model.P, world[ref]), model.center, model.k)

    assert np.linalg.norm(pred - truth, axis=1).mean() < 0.5
    assert np.allclose(model.offsets[0], 0)


def test_residuals_on_held_out_points_stay_within_noise():
    world, image, view_ids, _ = make_views(noise=0.3)
    train = np.arange(len(world)) % 10 != 0  # 10% dei punti di ogni vista escluso

    model = fit_camera(world[train], image[train], view_ids[train], image_size=(1700, 1240))
    held_out = model_residuals(model, world[~train], image[~train], view_ids[~train])

    assert held_out.shape == ((~train).sum(),)
    assert held_out.mean() < 0.8


def test_robust_fit_discards_a_spurious_column_of_centres():
    world, image, view_ids, _ = make_views()
    bad = np.zeros(len(world), bool)
    bad[view_ids == 3] = world[view_ids == 3, 0] == world[view_ids == 3, 0].max()  # ultima colonna di una vista
    image = image.copy()
    image[bad] += [60.0, 0.0]  # centri spostati di 60 px, come una colonna rilevata male al bordo

    plain = fit_camera(world, image, view_ids, image_size=(1700, 1240))
    model, inliers = fit_camera_robust(world, image, view_ids, image_size=(1700, 1240))

    assert bad.sum() > 5 and not inliers[bad].any() and inliers[~bad].all()
    assert model.rms < 0.6 < plain.rms
