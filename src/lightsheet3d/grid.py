import cv2 as cv
import numpy as np


def _blob_detector(max_area=20000):
    """Rilevatore di blob con area massima alta: quello di default (5000 px) scarta i cerchi grandi
    dell'"L" quando il master è vicino alla camera."""
    params = cv.SimpleBlobDetector_Params()
    params.maxArea = max_area
    return cv.SimpleBlobDetector_create(params)


def find_first_valid_circles_grid(img_gray, max_cols, max_rows, min_size=10):
    """Cerca il pattern di cerchi più grande, dai pattern grandi ai piccoli.

    Ritorna ((colonne, righe), centri) con centri float64 di shape (n, 2) in
    ordine row-major, oppure (None, None) se nessun pattern è valido.
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
    """Indici dei tre cerchi più estesi; ValueError se non sono nettamente più grandi del quarto."""
    _, binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, labels, stats, _ = cv.connectedComponentsWithStats(binary)
    h, w = labels.shape
    ij = np.rint(centers).astype(int)
    label = labels[ij[:, 1].clip(0, h - 1), ij[:, 0].clip(0, w - 1)]
    area = np.where(label > 0, stats[label, cv.CC_STAT_AREA], 0)

    order = np.argsort(area)[::-1]
    if area[order[2]] < min_area_ratio * area[order[3]]:
        raise ValueError("non ci sono tre cerchi nettamente più grandi degli altri")
    return order[:3]


def find_origin_index(img_gray, centers, min_area_ratio=1.3):
    """Indice del centro che fa da origine: l'angolo della "L" di tre cerchi grandi.

    I tre cerchi grandi sono i tre blob più estesi; l'angolo è il vertice dove
    i due lati sono più vicini a 90 gradi. Solleva ValueError se i tre blob
    non sono nettamente più grandi del quarto.
    """
    large = _large_circles(img_gray, centers, min_area_ratio)

    def abs_cos(corner):
        a, b = (centers[i] - centers[corner] for i in large if i != corner)
        return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return int(min(large, key=abs_cos))


def orient_grid(img_gray, pattern, centers, min_area_ratio=1.3):
    """Riordina i centri in modo che l'"L" vada verso colonne e righe crescenti; ritorna (centri, origine).

    `findCirclesGrid` non distingue una griglia simmetrica ruotata di 180°: l'ordine dei centri, e
    quindi il verso degli assi, dipende dall'immagine. L'"L" ha il vertice nell'origine e i due bracci
    lungo una riga e una colonna: se vanno verso indici decrescenti si invertono colonne o righe.
    Solleva ValueError se i bracci non sono allineati alla griglia.
    """
    cols, rows = pattern
    origin = find_origin_index(img_gray, centers, min_area_ratio)
    r0, c0 = divmod(origin, cols)
    arms = [divmod(i, cols) for i in _large_circles(img_gray, centers, min_area_ratio) if i != origin]
    along_row = [c - c0 for r, c in arms if r == r0]
    along_col = [r - r0 for r, c in arms if c == c0]
    if len(along_row) != 1 or len(along_col) != 1:
        raise ValueError("i bracci della \"L\" non sono allineati alle righe e alle colonne della griglia")
    grid = centers.reshape(rows, cols, 2)
    if along_row[0] < 0:
        grid, c0 = grid[:, ::-1], cols - 1 - c0
    if along_col[0] < 0:
        grid, r0 = grid[::-1], rows - 1 - r0
    return np.ascontiguousarray(grid).reshape(-1, 2), r0 * cols + c0


def build_object_points(pattern, origin_index, step, z):
    """Punti oggetto (n, 3) in mm: origine in `origin_index`, x lungo le colonne, y lungo le righe."""
    cols, rows = pattern
    cx, cy = np.meshgrid(np.arange(cols), np.arange(rows))
    xy = np.column_stack([cx.ravel(), cy.ravel()]).astype(np.float64)
    xy = (xy - xy[origin_index]) * step
    return np.column_stack([xy, np.full(len(xy), z)])
