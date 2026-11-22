import numpy as np
import pytest

from lightsheet3d.pipeline import calibrate_camera
from lightsheet3d.solids import CORPO6
from lightsheet3d.synthetic import Scene, write_dataset
from lightsheet3d.uncertainty import jackknife_se, leave_one_view_out


def test_jackknife_se_matches_the_textbook_value():
    # n = 3, deviations from the mean 1, 0, 1: sqrt((n - 1) / n * 2)
    assert jackknife_se([1.0, 2.0, 3.0]) == pytest.approx(np.sqrt(4 / 3))
    assert jackknife_se([5.0, 5.0, 5.0, 5.0]) == 0.0


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root, scan = write_dataset(Scene(), CORPO6, tmp_path_factory.mktemp("jackknife"), noise_px=0.5)
    camera = calibrate_camera(root / "camera_calibration")
    return leave_one_view_out(camera, root / "light_plane_calibration", root / "scans" / scan, CORPO6)


def test_every_view_but_the_reference_is_left_out_once(runs):
    assert [r.excluded for r in runs] == [None, 310, 320, 330, 340]


def test_synthetic_calibration_does_not_depend_on_a_single_view(runs):
    full, *left_out = runs

    assert full.median < 0.4
    assert all(np.abs(r.heights - full.heights).max() < 0.3 for r in left_out)  # on real data it exceeds 4 mm
