import cv2 as cv
import numpy as np
import pytest

from lightsheet3d.grid import (
    build_object_points,
    find_first_valid_circles_grid,
    find_origin_index,
)


def synthetic_grid(cols, rows, step=60, radius=12, margin=80, big=()):
    """Griglia di cerchi; i (col, riga) in `big` hanno raggio maggiore."""
    img = np.full((rows * step + 2 * margin, cols * step + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            rad = 19 if (c, r) in big else radius
            cv.circle(img, (margin + c * step, margin + r * step), rad, 0, -1)
    return img


def test_finds_largest_grid_and_returns_flat_centers():
    img = synthetic_grid(cols=12, rows=10)

    pattern, centers = find_first_valid_circles_grid(img, max_cols=14, max_rows=12, min_size=8)

    assert pattern == (12, 10)
    assert centers.shape == (120, 2)
    assert centers.dtype == np.float64


def test_centers_follow_row_major_order():
    img = synthetic_grid(cols=12, rows=10)

    _, centers = find_first_valid_circles_grid(img, max_cols=12, max_rows=10, min_size=8)

    assert np.allclose(centers[1] - centers[0], [60, 0], atol=1.0)
    assert np.allclose(centers[12] - centers[0], [0, 60], atol=1.0)


def test_returns_none_when_no_grid():
    blank = np.full((400, 400), 255, np.uint8)

    assert find_first_valid_circles_grid(blank, max_cols=12, max_rows=10, min_size=8) == (None, None)


def test_origin_is_corner_of_the_three_large_circles():
    big = [(3, 2), (7, 2), (3, 5)]  # angolo in (3, 2)
    img = synthetic_grid(cols=12, rows=10, big=big)
    _, centers = find_first_valid_circles_grid(img, max_cols=12, max_rows=10, min_size=8)

    assert find_origin_index(img, centers) == 2 * 12 + 3


def test_origin_corner_is_found_wherever_the_triplet_sits():
    big = [(8, 6), (4, 6), (8, 3)]  # angolo in (8, 6)
    img = synthetic_grid(cols=12, rows=10, big=big)
    _, centers = find_first_valid_circles_grid(img, max_cols=12, max_rows=10, min_size=8)

    assert find_origin_index(img, centers) == 6 * 12 + 8


def test_origin_requires_three_large_circles():
    img = synthetic_grid(cols=12, rows=10)
    _, centers = find_first_valid_circles_grid(img, max_cols=12, max_rows=10, min_size=8)
    # griglia senza cerchi grandi: le aree sono tutte uguali, nessun angolo ben definito
    with pytest.raises(ValueError):
        find_origin_index(img, centers)


def test_object_points_are_centred_on_origin_with_known_pitch_and_height():
    obj = build_object_points((12, 10), origin_index=27, step=20.0, z=300.0)

    assert obj.shape == (120, 3)
    assert np.allclose(obj[27], [0, 0, 300])
    assert np.allclose(obj[28] - obj[27], [20, 0, 0])
    assert np.allclose(obj[39] - obj[27], [0, 20, 0])
