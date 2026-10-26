import cv2 as cv
import numpy as np


def find_first_valid_circles_grid(img_gray, max_cols, max_rows, min_size=10):
    """Cerca il pattern di cerchi più grande, dai pattern grandi ai piccoli.

    Ritorna ((colonne, righe), centri) con centri float64 di shape (n, 2) in
    ordine row-major, oppure (None, None) se nessun pattern è valido.
    """
    for cols in range(max_cols, min_size - 1, -1):
        for rows in range(max_rows, min_size - 1, -1):
            found, centers = cv.findCirclesGrid(
                img_gray, (cols, rows), flags=cv.CALIB_CB_SYMMETRIC_GRID
            )
            if found and centers is not None:
                return (cols, rows), centers.reshape(-1, 2).astype(np.float64)
    return None, None


def find_origin_index(img_gray, centers, min_area_ratio=1.3):
    """Indice del centro che fa da origine: l'angolo della "L" di tre cerchi grandi.

    I tre cerchi grandi sono i tre blob più estesi; l'angolo è il vertice dove
    i due lati sono più vicini a 90 gradi. Solleva ValueError se i tre blob
    non sono nettamente più grandi del quarto.
    """
    _, binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, labels, stats, _ = cv.connectedComponentsWithStats(binary)
    h, w = labels.shape
    ij = np.rint(centers).astype(int)
    label = labels[ij[:, 1].clip(0, h - 1), ij[:, 0].clip(0, w - 1)]
    area = np.where(label > 0, stats[label, cv.CC_STAT_AREA], 0)

    order = np.argsort(area)[::-1]
    if area[order[2]] < min_area_ratio * area[order[3]]:
        raise ValueError("non ci sono tre cerchi nettamente più grandi degli altri")

    def abs_cos(corner):
        a, b = (centers[i] - centers[corner] for i in order[:3] if i != corner)
        return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return int(min(order[:3], key=abs_cos))


def build_object_points(pattern, origin_index, step, z):
    """Punti oggetto (n, 3) in mm: origine in `origin_index`, x lungo le colonne, y lungo le righe."""
    cols, rows = pattern
    cx, cy = np.meshgrid(np.arange(cols), np.arange(rows))
    xy = np.column_stack([cx.ravel(), cy.ravel()]).astype(np.float64)
    xy = (xy - xy[origin_index]) * step
    return np.column_stack([xy, np.full(len(xy), z)])
