from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Peak:
    y: float            # mm, position along the profile
    height: float       # mm above the resting plane
    left_angle: float   # degrees from the horizontal, left flank
    right_angle: float  # degrees from the horizontal, right flank


def level_to_plate(y, h, tolerances=(5.0, 2.0, 1.5)):
    """Subtract the line of the resting plane, estimated iteratively on the lowest points."""
    line = np.array([0.0, np.median(h[h <= np.percentile(h, 20)])])
    for tol in tolerances:
        on_plate = np.abs(h - np.polyval(line, y)) < tol
        line = np.polyfit(y[on_plate], h[on_plate], 1)
    return h - np.polyval(line, y)


def _flank_line(y, h, y_range, h_range):
    """Total-least-squares line through the flank points in the window: (midpoint, direction) or None."""
    inside = (y >= y_range[0]) & (y <= y_range[1]) & (h >= h_range[0]) & (h <= h_range[1])
    if inside.sum() < 5:
        return None
    pts = np.column_stack([y[inside], h[inside]])
    return pts.mean(axis=0), np.linalg.svd(pts - pts.mean(axis=0))[2][0]


def _angle(line):
    """Inclination (degrees) from the horizontal of a line (point, direction); nan if missing."""
    return float("nan") if line is None else float(np.degrees(np.arctan2(abs(line[1][1]), abs(line[1][0]))))


def _apex_height(y, h, y_peak, h_peak, left, right, plateau=3.0, tip=0.6, reach=3.0, lift=4.0):
    """Height of a peak from the raw profile (y, h), which is always truncated at the apex.

    The sheet has thickness and the per-row centroid cuts the top: a sharp peak is estimated from
    the intersection of the lines of the two flanks, a wide plateau (more than `plateau` mm within `tip` mm
    of the top) from the median of its points. If a flank cannot be measured, or the intersection falls
    far from the peak, it falls back to the maximum of the nearby points.
    """
    near = np.abs(y - y_peak) <= 6.0
    top = near & (h >= h[near].max() - tip)
    if np.ptp(y[top]) >= plateau:
        return float(np.median(h[top]))
    fallback = float(h[np.abs(y - y_peak) <= 1.0].max())
    if left is None or right is None:
        return fallback
    (p, d), (q, e) = left, right
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(cross) < 1e-6:
        return fallback
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / cross
    apex = p + t * d
    if abs(apex[0] - y_peak) > reach or not h_peak - 1.0 <= apex[1] <= h_peak + lift:
        return fallback
    return float(apex[1])


def measure_peaks(y, h, prominence=8.0, min_separation=5.0, band=(0.25, 0.75), reach=60.0):
    """Peaks of an already levelled profile (plane at h = 0): height and flank inclination.

    The angle of each flank is measured on the central band of the rise (from `band[0]` to
    `band[1]` of the height between the peak and the adjacent base), to avoid corners and fillets.
    """
    order = np.argsort(y)
    y, h = np.asarray(y)[order], np.asarray(h)[order]
    smooth = median_filter(h, size=5)
    idx, _ = find_peaks(smooth, prominence=prominence)
    # staggered points on a vertical edge must not give double peaks: keep the highest
    kept = []
    for i in sorted(idx, key=lambda i: -smooth[i]):
        if all(abs(y[i] - y[j]) >= min_separation for j in kept):
            kept.append(i)
    idx = np.array(sorted(kept), dtype=int)
    if len(idx) == 0:
        return []

    valleys = [idx[k] + int(np.argmin(smooth[idx[k]:idx[k + 1] + 1])) for k in range(len(idx) - 1)]
    floor = min((smooth[v] for v in valleys), default=0.0)

    peaks = []
    for k, i in enumerate(idx):
        left_base, right_base = (smooth[valleys[k - 1]] if k else floor), (smooth[valleys[k]] if k < len(valleys) else floor)
        left_limit = y[valleys[k - 1]] if k else y[i] - reach
        right_limit = y[valleys[k]] if k < len(valleys) else y[i] + reach
        lines = []
        for base, limits in ((left_base, (left_limit, y[i])), (right_base, (y[i], right_limit))):
            rise = smooth[i] - base
            lines.append(_flank_line(y, h, limits, (smooth[i] - band[1] * rise, smooth[i] - band[0] * rise)))
        height = _apex_height(y, h, y[i], smooth[i], *lines)
        peaks.append(Peak(float(y[i]), height, _angle(lines[0]), _angle(lines[1])))
    return peaks
