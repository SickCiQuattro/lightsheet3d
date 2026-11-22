import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter


def _stripe_runs(img_bgr, threshold, min_area):
    """(gray, [(row, start, end)]): the longest saturated run of the sheet in each row.

    The core of the sheet is saturated on all channels, whereas light paper has high R but lower G and B:
    min(B, G, R) is used. Small connected components (reflections) are discarded.
    """
    gray = img_bgr.min(axis=2)
    _, labels, stats, _ = cv.connectedComponentsWithStats((gray >= threshold).astype(np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[:, cv.CC_STAT_AREA] >= min_area)
    mask = np.isin(labels, keep[keep != 0])
    runs = []
    for row in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[row])
        run = max(np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1), key=len)
        runs.append((row, run[0], run[-1]))
    return gray, runs


def _weighted_centroid(gray, row, start, end, margin):
    """Column of the intensity-weighted centroid over the saturated run [start, end] plus `margin` pixels on each side."""
    window = np.arange(max(start - margin, 0), min(end + margin + 1, gray.shape[1]))
    weight = gray[row, window].astype(np.float64)
    weight -= weight.min()
    return (weight * window).sum() / weight.sum() if weight.sum() else (start + end) / 2


def laser_centroids(img_bgr, threshold=235, min_area=200, margin=2):
    """Centroid of the sheet for each row, in pixels (column, row), without interpolating.

    The centroid is weighted by intensity over the saturated core plus `margin` pixels on each side, so the
    blurred edges give the sub-pixel position. Rows without the sheet produce no points.
    """
    gray, runs = _stripe_runs(img_bgr, threshold, min_area)
    points = [(_weighted_centroid(gray, row, start, end, margin), row) for row, start, end in runs]
    return np.array(points, np.float64).reshape(-1, 2)


def laser_steger(img_bgr, sigma=None, threshold=235, min_area=200, margin=3, max_shift=1.0):
    """Centre of the sheet for each row with Steger's method, in pixels (column, row).

    The rows and the search region come from the saturated core (as in `laser_centroids`, hence with the
    same robustness to reflections); the position is the ridge of the min(B, G, R) brightness found with
    Gaussian derivatives of scale `sigma`: at each pixel the eigenvector of the Hessian with the most
    negative curvature gives the normal to the ridge, and the sub-pixel maximum is searched along it. It finds the
    centre along the normal rather than along the row, and exploits the edge profile instead of the saturated part alone.
    `sigma` must be at least width/(2√3), otherwise at the centre of a wide sheet the curvature
    vanishes: if missing it is derived from the median width of the saturated core, with a minimum of 1.5 px.
    If a row has no valid ridge, or the ridge is more than `max_shift` pixels from the centroid
    (at the end of a broken sheet the normal rotates and the position shifts by several pixels), it
    falls back to the weighted centroid.
    """
    gray, runs = _stripe_runs(img_bgr, threshold, min_area)
    if not runs:
        return np.empty((0, 2))
    if sigma is None:
        sigma = max(1.5, 0.3 * float(np.median([end - start + 1 for _, start, end in runs])))
    gray = gray.astype(np.float64)

    def d(order):  # Gaussian derivative; order = (rows, columns)
        return gaussian_filter(gray, sigma, order=order)

    gx, gy, gxx, gxy, gyy = d((0, 1)), d((1, 0)), d((0, 2)), d((1, 1)), d((2, 0))

    # most negative eigenvalue of the Hessian [[gxx, gxy], [gxy, gyy]] and its eigenvector: the normal to the ridge
    lam = (gxx + gyy) / 2 - np.hypot((gxx - gyy) / 2, gxy)
    nx, ny = gxy, lam - gxx
    norm = np.hypot(nx, ny)
    along_x = (gxx <= gyy).astype(np.float64)  # gxy = 0: the normal is the axis of least curvature
    nx = np.where(norm > 1e-9, nx / np.maximum(norm, 1e-9), along_x)
    ny = np.where(norm > 1e-9, ny / np.maximum(norm, 1e-9), 1 - along_x)
    curvature = nx**2 * gxx + 2 * nx * ny * gxy + ny**2 * gyy
    t = -(nx * gx + ny * gy) / np.where(np.abs(curvature) < 1e-12, -1e-12, curvature)
    ridge = (lam < 0) & (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5)

    points = []
    for row, start, end in runs:
        cols = np.arange(max(start - margin, 0), min(end + margin + 1, gray.shape[1]))
        cols = cols[ridge[row, cols]]
        centroid = _weighted_centroid(gray, row, start, end, margin)
        if len(cols) == 0:
            points.append((centroid, row))
            continue
        col = cols[np.argmin(lam[row, cols])]
        u, v = col + t[row, col] * nx[row, col], row + t[row, col] * ny[row, col]
        points.append((u, v) if abs(u - centroid) <= max_shift else (centroid, row))
    return np.array(points, np.float64).reshape(-1, 2)


EXTRACTORS = {"centroid": laser_centroids, "steger": laser_steger}
