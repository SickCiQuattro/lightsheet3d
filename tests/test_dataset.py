from pathlib import Path

import pytest

from lightsheet3d.dataset import Dataset

REPO = Path(__file__).parents[1]


def test_the_shipped_dataset_describes_the_lab_images():
    ds = Dataset.load(REPO / "data" / "dataset.toml")

    assert ds.image_size == (2130, 1420) and ds.pitch_mm == 20.0 and ds.plate_z == ds.reference_z == 350.0
    assert ds.target_path == REPO / "data" / "camera_calibration"
    assert len(ds.entries()) == 14 and {e.role for e in ds.entries()} == {"dev", "test"}
    assert all((ds.scans_path / e.file).exists() for e in ds.entries())


@pytest.mark.parametrize("name, z", [("26.jpg", 310), ("26l.jpg", 310), ("30_main.jpg", 350), ("data/x/35.jpg", 400)])
def test_z_comes_from_the_first_two_digits_of_the_name(name, z):
    assert Dataset(root=Path(".")).z_of(name) == z


def test_a_name_without_digits_is_rejected():
    with pytest.raises(ValueError, match="two digits"):
        Dataset(root=Path(".")).z_of("target.jpg")


def test_other_conventions_change_z(tmp_path):
    (tmp_path / "dataset.toml").write_text("z_scale = 5\nz_offset = 100\n")

    assert Dataset.load(tmp_path / "dataset.toml").z_of("30.jpg") == 250


def test_unknown_keys_are_rejected(tmp_path):
    (tmp_path / "dataset.toml").write_text("pich_mm = 20\n")

    with pytest.raises(ValueError, match="pich_mm"):
        Dataset.load(tmp_path / "dataset.toml")


def test_write_and_load_round_trip(tmp_path):
    original = Dataset(root=tmp_path, image_size=(1818, 1212), pitch_mm=10.0, pattern_max=(30, 20), reference_z=300.0,
                       plate_z=300.0, manifest="scans.csv")

    original.write()

    assert Dataset.load(tmp_path / "dataset.toml") == original
