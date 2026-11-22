import numpy as np
import pytest

from lightsheet3d.cli import main
from lightsheet3d.export import read_ply
from lightsheet3d.scanner import Calibration


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    assert main(["synthetic", str(root / "data"), "--solid", "corpo6"]) == 0
    assert main(["calibrate", str(root / "data" / "dataset.toml"), "-o", str(root / "calibration.json")]) == 0
    return root


def test_calibration_round_trips_through_json(workspace, tmp_path):
    calibration = Calibration.load(workspace / "calibration.json")
    calibration.save(tmp_path / "again.json")
    again = Calibration.load(tmp_path / "again.json")

    assert np.array_equal(again.model.P, calibration.model.P) and np.array_equal(again.plane, calibration.plane)
    assert again.plate_z == calibration.plate_z == 350.0
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


def test_evaluate_reports_the_manifest_scans_and_writes_a_csv(workspace, capsys):
    capsys.readouterr()

    code = main(["evaluate", str(workspace / "data" / "dataset.toml"), "--calibration", str(workspace / "calibration.json"),
                 "--csv", str(workspace / "results.csv")])

    out = capsys.readouterr().out
    assert code == 0 and "30o1.jpg" in out and "by role:" in out and "by solid:" in out
    assert (workspace / "results.csv").read_text().splitlines()[0].startswith("file,solid,role")


def test_evaluate_without_a_manifest_is_an_error(tmp_path, capsys):
    (tmp_path / "dataset.toml").write_text("pitch_mm = 20.0\n")

    code = main(["evaluate", str(tmp_path / "dataset.toml"), "--calibration", "x.json"])

    assert code == 1 and "error" in capsys.readouterr().err


def test_missing_calibration_is_reported_without_a_traceback(tmp_path, capsys):
    code = main(["scan", str(tmp_path / "nope.json"), "x.jpg"])

    assert code == 1 and "error" in capsys.readouterr().err


def test_unsupported_calibration_version_is_rejected(tmp_path):
    (tmp_path / "old.json").write_text('{"version": 0}')

    with pytest.raises(ValueError, match="versione"):
        Calibration.load(tmp_path / "old.json")


def test_extractor_is_stored_in_the_calibration_and_reused_by_scan(workspace, tmp_path, capsys):
    out = tmp_path / "steger.json"

    assert main(["calibrate", str(workspace / "data" / "dataset.toml"), "--extractor", "steger", "-o", str(out)]) == 0
    capsys.readouterr()
    assert main(["scan", str(out), str(workspace / "data" / "scans" / "30o1.jpg")]) == 0

    assert Calibration.load(out).extractor == "steger"
    assert Calibration.load(workspace / "calibration.json").extractor == "centroid"
    assert capsys.readouterr().out.count("peak ") == 3
