import cv2 as cv
import numpy as np

from lightsheet3d.laser import laser_centroids, laser_steger


def synthetic_frame(height=400, width=300, half_width=7, gap=None, speck=True):
    """Lama inclinata e satura su fondo grigio; `gap` = (riga0, riga1) senza lama."""
    rng = np.random.default_rng(0)
    img = np.full((height, width, 3), 170, np.uint8)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    rows = np.arange(height)
    centers = 100.0 + 0.1 * rows
    for r in rows:
        if gap and gap[0] <= r < gap[1]:
            continue
        lo, hi = int(np.ceil(centers[r] - half_width)), int(np.floor(centers[r] + half_width))
        img[r, lo:hi + 1] = 255
    if speck:
        img[50:53, 250:253] = 255  # riflesso isolato
    return img, centers


def test_centroids_match_true_line_center_per_row():
    img, centers = synthetic_frame()

    uv = laser_centroids(img)

    assert len(uv) == 400
    assert np.abs(uv[:, 0] - centers[uv[:, 1].astype(int)]).max() < 0.6


def test_rows_without_laser_are_dropped_not_imputed():
    img, _ = synthetic_frame(gap=(150, 200))

    uv = laser_centroids(img)

    rows = uv[:, 1].astype(int)
    assert len(uv) == 350
    assert not np.isin(np.arange(150, 200), rows).any()
    assert not np.isnan(uv).any()


def test_isolated_speck_is_ignored():
    img, _ = synthetic_frame(speck=True)

    uv = laser_centroids(img)

    assert uv[:, 0].max() < 200


def test_returns_empty_array_when_no_laser():
    img = np.full((50, 50, 3), 120, np.uint8)

    assert laser_centroids(img).shape == (0, 2)


def test_centroid_is_subpixel_on_a_blurred_stripe():
    cols = np.arange(300)
    profile = np.clip(80 + 300 * np.exp(-((cols - 100.3) ** 2) / (2 * 2.5**2)), 0, 255)  # nucleo saturo, bordi sfumati
    img = np.repeat(np.tile(profile, (400, 1))[..., None], 3, axis=2).astype(np.uint8)

    uv = laser_centroids(img)

    assert len(uv) == 400
    assert np.abs(uv[:, 0] - 100.3).max() < 0.1  # la media del solo nucleo saturo darebbe 100.0


def blurred_frame(sigma=2.5, grain=12, seed=0, height=400, width=300, half_width=3):
    """Lama inclinata sfocata con rumore sulle intensità; il centro vero è 100 + 0.1 riga."""
    rng = np.random.default_rng(seed)
    cols = np.arange(width)
    rows = np.arange(height)
    centers = 100.0 + 0.1 * rows
    profile = 255 * (np.abs(cols[None, :] - centers[:, None]) <= half_width).astype(np.float64)
    img = np.stack([profile] * 3, axis=2) * 0.6 + 100
    img = cv.GaussianBlur(img, (0, 0), sigma) if sigma else img
    return np.clip(img + rng.normal(0, grain, img.shape), 0, 255).astype(np.uint8), centers


def test_steger_finds_the_same_rows_as_the_centroid_and_ignores_specks():
    img, centers = synthetic_frame(gap=(100, 140))

    a, b = laser_centroids(img), laser_steger(img)

    assert len(a) == len(b) == 360
    assert np.array_equal(a[:, 1], np.round(b[:, 1]))
    error = np.abs(b[:, 0] - centers[np.round(b[:, 1]).astype(int)])
    assert np.median(error) < 0.15 and error.max() < 1.5  # alle estremità della lama la normale ruota: si ripiega sul baricentro


def test_steger_is_more_accurate_than_the_centroid_on_a_blurred_noisy_stripe():
    errors = {}
    for name, extract in (("centroid", laser_centroids), ("steger", laser_steger)):
        img, centers = blurred_frame(sigma=1.2, grain=15, half_width=5)
        uv = extract(img, threshold=200)
        errors[name] = np.sqrt(np.mean((uv[:, 0] - centers[uv[:, 1].astype(int)]) ** 2))

    assert errors["steger"] < errors["centroid"]
