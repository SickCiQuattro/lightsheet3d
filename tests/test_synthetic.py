"""Pipeline completa su immagini sintetiche: la geometria vera è nota, quindi gli errori sono quelli dell'algoritmo."""
import numpy as np
import pytest

from lightsheet3d.measure import measure_peaks
from lightsheet3d.solids import CORPO6
from lightsheet3d.synthetic import evaluate

TRUE_SHIFT = -57.0


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return evaluate(CORPO6, tmp_path_factory.mktemp("clean"))


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    return evaluate(CORPO6, tmp_path_factory.mktemp("noisy"), noise_px=0.5)


def test_clean_calibration_uses_every_view_and_finds_no_drift(clean):
    camera = clean["camera"]

    assert camera.skipped == () and sorted(set(camera.view_ids)) == [310, 320, 330, 340, 350]
    assert camera.model.rms < 0.1
    assert np.abs(camera.model.offsets).max() < 0.05  # il master vero non si sposta: niente deriva inventata


def test_clean_light_plane_matches_the_true_one(clean):
    assert clean["light"].plane == pytest.approx(clean["scene"].plane, rel=0.01, abs=0.01)


def test_clean_profile_matches_the_ground_truth(clean):
    assert clean["sign"] == 1 and clean["shift"] == pytest.approx(TRUE_SHIFT, abs=0.5)
    assert np.median(clean["dist"]) < 0.06
    heights = [p.height for p in measure_peaks(clean["y"], clean["h"], prominence=6, min_separation=10)]
    assert np.allclose(heights, [45, 25.05, 45], atol=0.15)  # senza stima dell'apice erano sottostimati di ~0.5 mm


def test_half_pixel_noise_keeps_the_error_below_half_a_millimetre(noisy):
    assert noisy["sign"] == 1 and noisy["shift"] == pytest.approx(TRUE_SHIFT, abs=1.0)
    assert np.median(noisy["dist"]) < 0.5


def test_two_pixel_noise_does_not_flip_the_profile(tmp_path):
    # con questo seed lo scarto verticale era quasi pari fra l'allineamento giusto e quello speculare
    result = evaluate(CORPO6, tmp_path, noise_px=2.0, seed=2)

    assert result["sign"] == 1 and result["shift"] == pytest.approx(TRUE_SHIFT, abs=1.5)
