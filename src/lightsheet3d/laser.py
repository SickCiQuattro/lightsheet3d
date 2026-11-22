import cv2 as cv
import numpy as np


def laser_centroids(img_bgr, threshold=235, min_area=200, margin=2):
    """Baricentro della lama per ogni riga, in pixel (colonna, riga), senza interpolare.

    Il nucleo della lama è saturo su tutti i canali, mentre la carta chiara ha
    R alto ma G e B più bassi: si usa min(B, G, R). Le componenti connesse
    piccole (riflessi) sono scartate. Il baricentro è pesato con l'intensità sul nucleo
    e `margin` pixel per lato, così i bordi sfumati danno la posizione sotto il pixel.
    Le righe senza lama non producono punti.
    """
    gray = img_bgr.min(axis=2)
    mask = gray >= threshold
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[:, cv.CC_STAT_AREA] >= min_area)
    mask = np.isin(labels, keep[keep != 0])

    points = []
    for row in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[row])
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        run = max(runs, key=len)
        window = np.arange(max(run[0] - margin, 0), min(run[-1] + margin + 1, gray.shape[1]))
        weight = gray[row, window].astype(np.float64)
        weight -= weight.min()
        points.append(((weight * window).sum() / weight.sum() if weight.sum() else run.mean(), row))
    return np.array(points, np.float64).reshape(-1, 2)
