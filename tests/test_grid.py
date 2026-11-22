import cv2 as cv
import numpy as np
import pytest

from lightsheet3d.grid import (
    build_object_points,
    find_first_valid_circles_grid,
    find_origin_index,
    orient_grid,
)


def synthetic_grid(cols, rows, step=60, radius=12, margin=80, big=(), big_radius=19):
    """Griglia di cerchi; i (col, riga) in `big` hanno raggio maggiore."""
    img = np.full((rows * step + 2 * margin, cols * step + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            rad = big_radius if (c, r) in big else radius
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


def test_finds_the_grid_when_the_large_circles_exceed_the_default_blob_area():
    # il master vicino alla camera: i tre cerchi grandi superano i 5000 px² del rilevatore di OpenCV
    img = synthetic_grid(cols=8, rows=6, step=160, radius=30, margin=150, big={(2, 2), (5, 2), (2, 4)}, big_radius=45)

    pattern, _ = find_first_valid_circles_grid(img, max_cols=9, max_rows=7, min_size=6)

    assert pattern == (8, 6)


@pytest.mark.parametrize("rotate", [False, True])
def test_orient_grid_puts_the_l_arms_toward_increasing_columns_and_rows(rotate):
    # "L" con origine in (col 3, riga 2): bracci lunghi 3 colonne e 2 righe
    big = {(3, 2), (6, 2), (3, 4)}
    img = synthetic_grid(cols=10, rows=8, step=70, radius=13, margin=90, big=big)
    if rotate:
        img = cv.rotate(img, cv.ROTATE_180)  # findCirclesGrid restituisce i centri in un altro ordine

    pattern, centers = find_first_valid_circles_grid(img, max_cols=10, max_rows=8, min_size=8)
    centers, origin = orient_grid(img, pattern, centers)

    # i tre cerchi grandi cadono nei punti dell'immagine in cui sono disegnati, qualunque sia l'ordine dei centri
    def pixel(col, row):
        xy = np.array([90 + col * 70, 90 + row * 70], float)
        return np.array(img.shape[::-1], float) - xy if rotate else xy

    cols, _ = pattern
    assert np.allclose(centers[[origin, origin + 3, origin + 2 * cols]], [pixel(3, 2), pixel(6, 2), pixel(3, 4)], atol=1.5)
