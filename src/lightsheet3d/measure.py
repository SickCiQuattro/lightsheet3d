from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Peak:
    y: float            # mm, posizione lungo il profilo
    height: float       # mm sopra il piano di appoggio
    left_angle: float   # gradi rispetto all'orizzontale, fianco a sinistra
    right_angle: float  # gradi rispetto all'orizzontale, fianco a destra


def level_to_plate(y, h, tolerances=(5.0, 2.0, 1.5)):
    """Sottrae la retta del piano di appoggio, stimata iterativamente sui punti più bassi."""
    line = np.array([0.0, np.median(h[h <= np.percentile(h, 20)])])
    for tol in tolerances:
        on_plate = np.abs(h - np.polyval(line, y)) < tol
        line = np.polyfit(y[on_plate], h[on_plate], 1)
    return h - np.polyval(line, y)


def _flank_line(y, h, y_range, h_range):
    """Retta ai minimi quadrati totali sui punti del fianco nella finestra: (punto medio, direzione) o None."""
    inside = (y >= y_range[0]) & (y <= y_range[1]) & (h >= h_range[0]) & (h <= h_range[1])
    if inside.sum() < 5:
        return None
    pts = np.column_stack([y[inside], h[inside]])
    return pts.mean(axis=0), np.linalg.svd(pts - pts.mean(axis=0))[2][0]


def _angle(line):
    """Inclinazione (gradi) rispetto all'orizzontale di una retta (punto, direzione); nan se manca."""
    return float("nan") if line is None else float(np.degrees(np.arctan2(abs(line[1][1]), abs(line[1][0]))))


def _apex_height(y, h, y_peak, h_peak, left, right, plateau=3.0, tip=0.6, reach=3.0, lift=4.0):
    """Altezza di un picco dal profilo grezzo (y, h), che nell'apice è sempre troncato.

    La lama ha spessore e il baricentro per riga taglia la cima: un picco acuto si stima con
    l'intersezione delle rette dei due fianchi, un piano largo (più di `plateau` mm entro `tip` mm
    dalla cima) con la mediana dei suoi punti. Se un fianco non è misurabile, o l'intersezione cade
    lontano dal picco, si ripiega sul massimo dei punti vicini.
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
    """Picchi di un profilo già livellato (piano a h = 0): altezza e inclinazione dei fianchi.

    L'angolo di ogni fianco si misura sulla fascia centrale della salita (da `band[0]` a
    `band[1]` dell'altezza fra picco e base adiacente), per evitare spigoli e raccordi.
    """
    order = np.argsort(y)
    y, h = np.asarray(y)[order], np.asarray(h)[order]
    smooth = median_filter(h, size=5)
    idx, _ = find_peaks(smooth, prominence=prominence)
    # punti sfalsati su un bordo verticale non devono dare picchi doppi: si tiene il più alto
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
