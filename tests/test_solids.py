from itertools import pairwise
from math import radians, tan

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lightsheet3d.solids import CORPO1, CORPO2, CORPO3, CORPO4, CORPO5, CORPO6, SOLIDS, align, nominal_curve


def _rise(h, deg):
    return h / tan(radians(deg))


def _walk(steps):
    return np.cumsum([(0.0, 0.0), *steps], axis=0)


# Profili costruiti a mano dalle quote dei disegni (corsa del fianco = altezza / tan(inclinazione)): sono un
# controllo indipendente dei profili estratti dai PDF in profiles.json.
HAND_CORPO6 = _walk([
    (0, 10), (5, 0), (0, 5), (5, 0),
    (_rise(30, 60), 30), (_rise(30, 75), -30), (10, 0),
    (_rise(10.05, 70), 10.05), (_rise(10.05, 65), -10.05), (10, 0),
    (_rise(30, 80), 30), (_rise(30, 55), -30),
    (5, 0), (0, -10), (5, 0), (0, -5),
])
_CORPO5_FLANKS = _walk([
    (0, 35), (8, 0), (6.85, -6.85 * tan(radians(75))), (11.87, 11.87 * tan(radians(60))), (8, 0),
    (11.38, -11.38 * tan(radians(65))), (6.95, 6.95 * tan(radians(80))), (8, 0),
    (13.95, -13.95 * tan(radians(70))),
])
HAND_CORPO5 = np.vstack([_CORPO5_FLANKS, [_CORPO5_FLANKS[-1, 0], 0.0]])  # più il lato destro fino al piano


def _distance_between(a, b, step=0.05):
    def dense(outline):
        return np.vstack([np.linspace(p, q, max(2, int(np.hypot(*(q - p)) / step))) for p, q in pairwise(outline)])
    A, B = dense(a), dense(b)
    return max(cKDTree(B).query(A)[0].max(), cKDTree(A).query(B)[0].max())


def test_corpo6_extracted_from_the_drawing_matches_the_hand_built_profile():
    assert _distance_between(CORPO6.outline, HAND_CORPO6) < 0.05


def test_corpo5_extracted_from_the_drawing_matches_the_hand_built_profile():
    assert _distance_between(CORPO5.outline, HAND_CORPO5) < 0.05


@pytest.mark.parametrize("solid, width, height", [(CORPO1, 75, 45), (CORPO2, 75, 30), (CORPO3, 100, 25), (CORPO4, 75, 24),
                                                  (CORPO5, 75, 45), (CORPO6, 100, 45)])
def test_every_profile_has_the_quoted_width_and_height(solid, width, height):
    assert solid.width == pytest.approx(width, abs=0.01)
    assert solid.outline[:, 1].max() == pytest.approx(height, abs=0.02)
    assert solid.depth == pytest.approx(50.0, abs=0.1)
    assert solid.outline[0].tolist() == [0.0, 0.0] and solid.outline[-1, 1] == 0.0


def test_corpo1_matches_the_staircase_and_the_slope_read_from_the_drawing():
    expected = [(0, 0), (0, 15), (15, 15), (15, 45), (30, 45), (30, 35), (45, 35), (45, 25), (75, 10), (75, 0)]

    assert np.allclose(CORPO1.outline, expected, atol=0.03)  # pendenza 1:2, angolo quotato 116.57 gradi


def test_solid_registry_lists_the_six_profiles():
    assert sorted(SOLIDS) == [f"corpo{n}" for n in range(1, 7)]


def test_corpo6_matches_the_quoted_dimensions():
    assert CORPO6.width == pytest.approx(100, abs=0.05)  # le 12 quote in pianta sommano 100.01
    assert [h for _, h in CORPO6.peaks()] == pytest.approx([45, 25.05, 45], abs=0.06)


def test_corpo5_matches_the_quoted_dimensions():
    assert CORPO5.width == pytest.approx(75, abs=0.01)  # le quote in pianta sommano esattamente 75
    assert [h for _, h in CORPO5.peaks()] == pytest.approx([35, 30, 45], abs=0.1)


@pytest.mark.parametrize("solid", [CORPO5, CORPO6])
@pytest.mark.parametrize("sign, shift", [(1, -40.0), (-1, 25.0)])
def test_align_recovers_offset_and_direction(solid, sign, shift):
    yn, hn = nominal_curve(solid, sign, shift)
    y = np.linspace(yn[0], yn[-1], 600)
    h = np.interp(y, yn, hn) + np.random.default_rng(0).normal(0, 0.3, y.size)

    found_sign, found_shift = align(solid, y, h)

    assert found_sign == sign and found_shift == pytest.approx(shift, abs=0.3)


def test_nominal_curve_is_increasing_in_y_in_both_directions():
    for sign in (1, -1):
        y, h = nominal_curve(CORPO6, sign, 0.0)

        assert np.all(np.diff(y) >= 0) and h.max() == pytest.approx(45.0, abs=0.01)
