import cv2 as cv
import numpy as np


def _blob_detector(max_area=20000):
    """Blob detector with a high maximum area: the default one (5000 px) discards the large circles
    of the "L" when the master is close to the camera."""
    params = cv.SimpleBlobDetector_Params()
    params.maxArea = max_area
    return cv.SimpleBlobDetector_create(params)


def find_first_valid_circles_grid(img_gray, max_cols, max_rows, min_size=10):
    """Look for the largest circle pattern, from large patterns to small ones.

    Returns ((columns, rows), centres) with float64 centres of shape (n, 2) in
    row-major order, or (None, None) if no pattern is valid.
    """
    detector = _blob_detector()
    for cols in range(max_cols, min_size - 1, -1):
        for rows in range(max_rows, min_size - 1, -1):
            found, centers = cv.findCirclesGrid(
                img_gray, (cols, rows), flags=cv.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector
            )
            if found and centers is not None:
                return (cols, rows), centers.reshape(-1, 2).astype(np.float64)
    return None, None


def _large_circles(img_gray, centers, min_area_ratio):
    """Indices of the three largest circles; ValueError if they are not clearly larger than the fourth."""
    _, binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, labels, stats, _ = cv.connectedComponentsWithStats(binary)
    h, w = labels.shape
    ij = np.rint(centers).astype(int)
    label = labels[ij[:, 1].clip(0, h - 1), ij[:, 0].clip(0, w - 1)]
    area = np.where(label > 0, stats[label, cv.CC_STAT_AREA], 0)

    order = np.argsort(area)[::-1]
    if area[order[2]] < min_area_ratio * area[order[3]]:
        raise ValueError("there are not three circles clearly larger than the others")
    return order[:3]


def find_origin_index(img_gray, centers, min_area_ratio=1.3):
    """Index of the centre that serves as origin: the corner of the "L" of three large circles.

    The three large circles are the three largest blobs; the corner is the vertex where
    the two sides are closest to 90 degrees. Raises ValueError if the three blobs
    are not clearly larger than the fourth.
    """
    large = _large_circles(img_gray, centers, min_area_ratio)

    def abs_cos(corner):
        a, b = (centers[i] - centers[corner] for i in large if i != corner)
        return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return int(min(large, key=abs_cos))


def orient_grid(img_gray, pattern, centers, min_area_ratio=1.3):
    """Reorder the centres so that the "L" points towards increasing columns and rows; returns (centres, origin).

    `findCirclesGrid` cannot tell a symmetric grid rotated by 180° apart: the order of the centres, and
    hence the direction of the axes, depends on the image. The "L" has its vertex at the origin and its two arms
    along a row and a column: if they point towards decreasing indices the columns or rows are reversed.
    Raises ValueError if the arms are not aligned with the grid.
    """
    cols, rows = pattern
    origin = find_origin_index(img_gray, centers, min_area_ratio)
    r0, c0 = divmod(origin, cols)
    arms = [divmod(i, cols) for i in _large_circles(img_gray, centers, min_area_ratio) if i != origin]
    along_row = [c - c0 for r, c in arms if r == r0]
    along_col = [r - r0 for r, c in arms if c == c0]
    if len(along_row) != 1 or len(along_col) != 1:
        raise ValueError("the arms of the \"L\" are not aligned with the rows and columns of the grid")
    grid = centers.reshape(rows, cols, 2)
    if along_row[0] < 0:
        grid, c0 = grid[:, ::-1], cols - 1 - c0
    if along_col[0] < 0:
        grid, r0 = grid[::-1], rows - 1 - r0
    return np.ascontiguousarray(grid).reshape(-1, 2), r0 * cols + c0


def build_object_points(pattern, origin_index, step, z):
    """Object points (n, 3) in mm: origin at `origin_index`, x along the columns, y along the rows."""
    cols, rows = pattern
    cx, cy = np.meshgrid(np.arange(cols), np.arange(rows))
    xy = np.column_stack([cx.ravel(), cy.ravel()]).astype(np.float64)
    xy = (xy - xy[origin_index]) * step
    return np.column_stack([xy, np.full(len(xy), z)])
