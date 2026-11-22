import numpy as np
import pytest

from lightsheet3d.cli import main
from lightsheet3d.export import read_ply
from lightsheet3d.storage import load_calibration, save_calibration


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    assert main(["synthetic", str(root / "data"), "--solid", "corpo6"]) == 0
    assert main(["calibrate", str(root / "data" / "camera_calibration"), str(root / "data" / "light_plane_calibration"),
                 "-o", str(root / "calibration.json")]) == 0
    return root


def test_calibration_round_trips_through_json(workspace, tmp_path):
    calibration = load_calibration(workspace / "calibration.json")
    save_calibration(tmp_path / "again.json", calibration.model, calibration.plane)
    again = load_calibration(tmp_path / "again.json")

    assert np.array_equal(again.model.P, calibration.model.P) and np.array_equal(again.plane, calibration.plane)
    points = np.array([[500.0, 400.0], [1500.0, 900.0]])
    assert np.allclose(again.model.undistort(points), calibration.model.undistort(points))


def test_scan_prints_the_peaks_and_writes_a_ply(workspace, capsys):
    ply = workspace / "out" / "scan.ply"
    capsys.readouterr()

    code = main(["scan", str(workspace / "calibration.json"), str(workspace / "data" / "scans" / "30o1.jpg"),
                 "--solid", "corpo6", "--ply", str(ply)])

    out = capsys.readouterr().out
    points, colors = read_ply(ply)
    assert code == 0 and out.count("peak ") == 3
    assert "height  45.0 mm" in out and "comparison with Corpo 6" in out
    assert len(points) > 1000 and colors.shape == points.shape


def test_missing_calibration_is_reported_without_a_traceback(tmp_path, capsys):
    code = main(["scan", str(tmp_path / "nope.json"), "x.jpg"])

    assert code == 1 and "error" in capsys.readouterr().err


def test_unsupported_calibration_version_is_rejected(tmp_path):
    (tmp_path / "old.json").write_text('{"version": 0}')

    with pytest.raises(ValueError, match="versione"):
        load_calibration(tmp_path / "old.json")


def test_extractor_is_stored_in_the_calibration_and_reused_by_scan(workspace, tmp_path, capsys):
    out = tmp_path / "steger.json"
    data = workspace / "data"

    assert main(["calibrate", str(data / "camera_calibration"), str(data / "light_plane_calibration"),
                 "--extractor", "steger", "-o", str(out)]) == 0
    capsys.readouterr()
    assert main(["scan", str(out), str(data / "scans" / "30o1.jpg")]) == 0

    assert load_calibration(out).extractor == "steger"
    assert load_calibration(workspace / "calibration.json").extractor == "centroid"
    assert capsys.readouterr().out.count("peak ") == 3
