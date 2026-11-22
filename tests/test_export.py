import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightsheet3d.export import distance_colors, read_ply, write_ply
from lightsheet3d.report import scan_3d
from lightsheet3d.solids import CORPO6, nominal_curve


def test_ply_round_trip_with_colors(tmp_path):
    points = np.random.default_rng(0).normal(size=(50, 3)) * 30
    colors = np.random.default_rng(1).integers(0, 256, (50, 3), dtype=np.uint8)

    write_ply(tmp_path / "a.ply", points, colors)
    back, back_colors = read_ply(tmp_path / "a.ply")

    assert np.allclose(back, points, atol=1e-4) and np.array_equal(back_colors, colors)
    assert (tmp_path / "a.ply").read_text().startswith("ply\nformat ascii 1.0\nelement vertex 50\n")


def test_ply_without_colors_has_no_color_properties(tmp_path):
    write_ply(tmp_path / "a.ply", np.zeros((3, 3)))

    assert "red" not in (tmp_path / "a.ply").read_text()
    assert read_ply(tmp_path / "a.ply")[1] is None


def test_ply_rejects_colors_of_the_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        write_ply(tmp_path / "a.ply", np.zeros((3, 3)), np.zeros((2, 3), np.uint8))


def test_distance_colors_go_from_green_to_red():
    colors = distance_colors([0.0, 1.0, 5.0], vmax=2.0)

    assert colors.tolist() == [[0, 255, 0], [128, 128, 0], [255, 0, 0]]


def test_scan_3d_returns_zero_distance_for_points_on_the_nominal_profile():
    yn, hn = nominal_curve(CORPO6, 1, -57.0)
    y = np.linspace(yn[1], yn[-2], 200)
    h = np.interp(y, yn, hn)
    X = np.column_stack([np.full_like(y, 40.0), y, 350 - h])
    ax = plt.figure().add_subplot(projection="3d")

    dist = scan_3d(ax, CORPO6, X, h, 1, -57.0)
    plt.close("all")

    assert np.median(dist) < 0.1
