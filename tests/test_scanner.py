import numpy as np
import pytest

from lightsheet3d.dataset import Dataset
from lightsheet3d.export import read_ply
from lightsheet3d.scanner import Calibration, Scanner
from lightsheet3d.solids import CORPO6, SOLIDS
from lightsheet3d.synthetic import Scene, write_dataset


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root, scan = write_dataset(Scene(), CORPO6, tmp_path_factory.mktemp("scanner"))
    dataset = Dataset.load(root / "dataset.toml")
    calibration = Calibration.fit(dataset)
    return dataset, calibration, Scanner(calibration).scan(root / "scans" / scan)


def test_fit_reports_how_the_calibration_went(synthetic):
    _, calibration, _ = synthetic
    report = calibration.report

    assert report["views"] == [310, 320, 330, 340, 350] and report["skipped"] == []
    assert report["camera_rms_px"] < 0.1 and report["plane_validation_rms_mm"] < 0.1


def test_scan_returns_a_levelled_profile(synthetic):
    _, _, profile = synthetic

    assert profile.points.shape[1] == 3 and len(profile.height) == len(profile.points)
    assert [round(p.height) for p in profile.peaks()] == [45, 25, 45]


def test_compare_finds_the_nominal_profile_and_its_peaks(synthetic):
    _, _, profile = synthetic

    comparison = profile.compare(CORPO6)

    assert comparison.sign == 1 and comparison.shift == pytest.approx(-57.0, abs=0.3)
    assert comparison.median < 0.06 and comparison.p90 < 0.4 and comparison.rms < 0.5
    assert np.abs(comparison.height_errors).max() < 0.15 and len(comparison.height_errors) == 3


def test_compare_with_the_wrong_solid_is_clearly_worse(synthetic):
    _, _, profile = synthetic

    assert profile.compare(SOLIDS["corpo3"]).median > 10 * profile.compare(CORPO6).median


def test_to_ply_writes_x_y_height_with_distance_colours(synthetic, tmp_path):
    _, _, profile = synthetic

    profile.to_ply(tmp_path / "a.ply", profile.compare(CORPO6))
    points, colors = read_ply(tmp_path / "a.ply")

    assert np.allclose(points[:, 2], profile.height, atol=1e-3) and colors.shape == points.shape


def test_calibration_keeps_the_plate_height_and_extractor(synthetic, tmp_path):
    _, calibration, _ = synthetic
    calibration.save(tmp_path / "c.json")

    again = Calibration.load(tmp_path / "c.json")

    assert (again.plate_z, again.extractor) == (350.0, "centroid") and np.array_equal(again.plane, calibration.plane)


def test_oblique_comparison_of_a_perpendicular_cut_finds_no_stretch(synthetic):
    _, _, profile = synthetic

    comparison = profile.compare(CORPO6, oblique=True)

    assert comparison.scale == pytest.approx(1.0, abs=0.03) and comparison.tilt_deg < 15
    assert comparison.median < profile.compare(CORPO6).median + 0.03
