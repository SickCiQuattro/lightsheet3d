import cv2 as cv
import numpy as np


def laser_centroids(img_bgr, threshold=235, min_area=200):
    """Baricentro della lama per ogni riga, in pixel (colonna, riga), senza interpolare.

    Il nucleo della lama è saturo su tutti i canali, mentre la carta chiara ha
    R alto ma G e B più bassi: si usa min(B, G, R). Le componenti connesse
    piccole (riflessi) sono scartate. Le righe senza lama non producono punti.
    """
    mask = img_bgr.min(axis=2) >= threshold
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[:, cv.CC_STAT_AREA] >= min_area)
    mask = np.isin(labels, keep[keep != 0])

    points = []
    for row in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[row])
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        points.append((max(runs, key=len).mean(), row))
    return np.array(points, np.float64).reshape(-1, 2)
