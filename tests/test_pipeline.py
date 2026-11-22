"""Regressione sui dati reali del repo: il profilo del Corpo 6 deve rispettare il disegno."""
from pathlib import Path

import numpy as np
import pytest

from lightsheet3d.lightplane import plane_errors
from lightsheet3d.measure import level_to_plate, measure_peaks
from lightsheet3d.pipeline import PLATE_Z, calibrate_camera, calibrate_light_plane, scan_points
from lightsheet3d.solids import CORPO5, CORPO6, align, distances, nominal_curve

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def calibration():
    camera = calibrate_camera(DATA / "camera_calibration")
    return camera, calibrate_light_plane(DATA / "light_plane_calibration", camera.model)


def test_camera_uses_only_views_up_to_the_main_target(calibration):
    camera, _ = calibration

    assert sorted(set(camera.view_ids)) == [300, 310, 320, 330, 340, 350]  # 300 era scartata dal blob detector di default
    assert camera.model.rms < 10


def test_light_plane_validation_error_is_below_one_millimetre(calibration):
    _, lightplane = calibration

    errors = plane_errors(lightplane.plane, lightplane.points[lightplane.is_validation])

    assert 0 < lightplane.is_validation.sum() < len(lightplane.points)
    assert np.sqrt((errors**2).mean()) < 1.0


def test_corpo6_peak_heights_match_the_drawing(calibration):
    camera, lightplane = calibration
    X = scan_points(DATA / "scans" / "30o34.jpg", camera.model, lightplane.plane)

    y = X[:, 1]
    peaks = measure_peaks(y, level_to_plate(y, PLATE_Z - X[:, 2]), prominence=6)

    assert np.allclose([p.height for p in peaks], [45, 25, 45], atol=2.0)  # disegno: 45 / 25 / 45 mm


@pytest.mark.parametrize("image, solid", [("30o34", CORPO6), ("30o3", CORPO5)])
def test_measured_profile_follows_the_nominal_one(calibration, image, solid):
    camera, lightplane = calibration
    X = scan_points(DATA / "scans" / f"{image}.jpg", camera.model, lightplane.plane)
    y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])

    sign, shift = align(solid, y, h)
    yn, _ = nominal_curve(solid, sign, shift)
    on_solid = (y > yn[1]) & (y < yn[-2])

    assert np.median(distances(solid, sign, shift, y[on_solid], h[on_solid])) < 1.0
