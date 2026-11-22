import numpy as np
import pytest

from lightsheet3d.solids import CORPO5, CORPO6, align, nominal_curve


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
