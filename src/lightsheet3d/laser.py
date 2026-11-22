import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter


def _stripe_runs(img_bgr, threshold, min_area):
    """(gray, [(riga, inizio, fine)]): il tratto saturo più lungo della lama in ogni riga.

    Il nucleo della lama è saturo su tutti i canali, mentre la carta chiara ha R alto ma G e B più bassi:
    si usa min(B, G, R). Le componenti connesse piccole (riflessi) sono scartate.
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
    """Colonna del baricentro pesato con l'intensità sul tratto saturo [start, end] e `margin` pixel per lato."""
    window = np.arange(max(start - margin, 0), min(end + margin + 1, gray.shape[1]))
    weight = gray[row, window].astype(np.float64)
    weight -= weight.min()
    return (weight * window).sum() / weight.sum() if weight.sum() else (start + end) / 2


def laser_centroids(img_bgr, threshold=235, min_area=200, margin=2):
    """Baricentro della lama per ogni riga, in pixel (colonna, riga), senza interpolare.

    Il baricentro è pesato con l'intensità sul nucleo saturo e `margin` pixel per lato, così i bordi
    sfumati danno la posizione sotto il pixel. Le righe senza lama non producono punti.
    """
    gray, runs = _stripe_runs(img_bgr, threshold, min_area)
    points = [(_weighted_centroid(gray, row, start, end, margin), row) for row, start, end in runs]
    return np.array(points, np.float64).reshape(-1, 2)


def laser_steger(img_bgr, sigma=None, threshold=235, min_area=200, margin=3, max_shift=1.0):
    """Centro della lama per ogni riga con il metodo di Steger, in pixel (colonna, riga).

    Le righe e la regione di ricerca vengono dal nucleo saturo (come in `laser_centroids`, quindi con la
    stessa robustezza ai riflessi); la posizione è la cresta della luminosità min(B, G, R) trovata con le
    derivate gaussiane di scala `sigma`: in ogni pixel l'autovettore della Hessiana con la curvatura più
    negativa dà la normale alla cresta e il massimo sotto il pixel si cerca lungo di essa. Ricava il
    centro sulla normale e non sulla riga, e sfrutta il profilo dei bordi invece della sola parte satura.
    `sigma` deve essere almeno larghezza/(2√3), altrimenti al centro di una lama larga la curvatura si
    annulla: se manca si ricava dalla larghezza mediana del nucleo saturo, con un minimo di 1.5 px.
    Se in una riga non c'è una cresta valida, o la cresta dista più di `max_shift` pixel dal baricentro
    (all'estremità di una lama interrotta la normale ruota e la posizione si sposta di diversi pixel), si
    ripiega sul baricentro pesato.
    """
    gray, runs = _stripe_runs(img_bgr, threshold, min_area)
    if not runs:
        return np.empty((0, 2))
    if sigma is None:
        sigma = max(1.5, 0.3 * float(np.median([end - start + 1 for _, start, end in runs])))
    gray = gray.astype(np.float64)

    def d(order):  # derivata gaussiana; order = (righe, colonne)
        return gaussian_filter(gray, sigma, order=order)

    gx, gy, gxx, gxy, gyy = d((0, 1)), d((1, 0)), d((0, 2)), d((1, 1)), d((2, 0))

    # autovalore più negativo della Hessiana [[gxx, gxy], [gxy, gyy]] e relativo autovettore: la normale alla cresta
    lam = (gxx + gyy) / 2 - np.hypot((gxx - gyy) / 2, gxy)
    nx, ny = gxy, lam - gxx
    norm = np.hypot(nx, ny)
    along_x = (gxx <= gyy).astype(np.float64)  # gxy = 0: la normale è l'asse di curvatura minore
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
