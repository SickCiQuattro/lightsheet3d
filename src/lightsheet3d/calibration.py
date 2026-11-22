from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def _homogeneous(points):
    return np.hstack([points, np.ones((len(points), 1))])


def project(P, world):
    """Proietta punti oggetto (n, 3) in pixel (n, 2) con la matrice P (3, 4)."""
    h = _homogeneous(np.asarray(world, np.float64)) @ P.T
    return h[:, :2] / h[:, 2:]


def reprojection_errors(P, world, image):
    """Distanza euclidea in pixel tra punti misurati e punti ri-proiettati."""
    return np.linalg.norm(project(P, world) - np.asarray(image, np.float64).reshape(-1, 2), axis=1)


def _normalisation(points):
    """Matrice di similarità che porta i punti a media 0 e deviazione standard 1."""
    mean, scale = points.mean(axis=0), points.std()
    T = np.eye(points.shape[1] + 1)
    T[:-1, :-1] *= 1 / scale
    T[:-1, -1] = -mean / scale
    return T


def estimate_projection_matrix(world, image):
    """DLT normalizzata (Hartley): stima P (3, 4) da corrispondenze 3D-2D.

    I punti vanno normalizzati: con coordinate in mm e pixel i termini di Q
    differiscono di 5 ordini di grandezza e il sistema è mal condizionato.
    P è scalata con |P[2, :3]| = 1 e segno tale che i punti stiano davanti.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    T, U = _normalisation(world), _normalisation(image)
    Xn = _homogeneous(world) @ T.T
    xn = _homogeneous(image) @ U.T

    zeros = np.zeros_like(Xn)
    rows_u = np.hstack([Xn, zeros, -xn[:, :1] * Xn])
    rows_v = np.hstack([zeros, Xn, -xn[:, 1:2] * Xn])
    _, _, Vt = np.linalg.svd(np.vstack([rows_u, rows_v]))

    P = np.linalg.inv(U) @ Vt[-1].reshape(3, 4) @ T
    P /= np.linalg.norm(P[2, :3])
    return P if (_homogeneous(world) @ P[2]).mean() > 0 else -P


_RADIAL_SCALE = 1000.0  # px: rende k1, k2 numeri di ordine 0.1


def _radial_factor(points, center, k):
    r2 = (((points - center) / _RADIAL_SCALE) ** 2).sum(axis=1, keepdims=True)
    return 1 + sum(ki * r2 ** (i + 1) for i, ki in enumerate(k))


def distort_points(ideal, center, k):
    """Applica la distorsione radiale ai pixel ideali (n, 2)."""
    return center + (ideal - center) * _radial_factor(ideal, center, k)


def undistort_points(distorted, center, k, iterations=20):
    """Inverte `distort_points` per punto fisso."""
    ideal = distorted.copy()
    for _ in range(iterations):
        ideal = center + (distorted - center) / _radial_factor(ideal, center, k)
    return ideal


@dataclass(frozen=True)
class CameraModel:
    P: np.ndarray        # (3, 4) proiezione ideale, |P[2, :3]| = 1
    center: np.ndarray   # (2,) centro della distorsione radiale, px
    k: np.ndarray        # coefficienti radiali
    offsets: np.ndarray  # (viste, 2) spostamento laterale stimato del master, mm
    views: np.ndarray    # (viste,) identificativo di ogni riga di `offsets`
    rms: float           # px, sui punti di calibrazione

    def undistort(self, points):
        return undistort_points(points, self.center, self.k)


def fit_camera(world, image, view_ids, image_size, n_radial=2, reference_view=0, f_scale=3.0):
    """Stima congiunta di P e distorsione radiale dai punti di tutte le viste.

    Il master non si muove solo lungo z fra le viste (setup instabile): lo
    spostamento laterale di ogni vista è un parametro di disturbo stimato,
    con la vista di riferimento fissata a zero che definisce l'origine.
    Gli identificativi delle viste sono le quote z (mm).
    Funzione di costo robusta (soft L1) contro i centri mal rilevati.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    views, idx = np.unique(view_ids, return_inverse=True)
    ref = int(np.searchsorted(views, reference_view))
    free = np.arange(len(views)) != ref

    dz = (views - views[ref]).astype(np.float64)

    def unpack(x):
        offsets = np.zeros((len(views), 2))
        offsets[free] = x[14 + n_radial:].reshape(-1, 2)
        # Una deriva degli offset lineare in z è indistinguibile da una piccola inclinazione dell'asse
        # in P (stessi residui): si fissa togliendola, cioè assumendo che il master scorra lungo z.
        offsets -= dz[:, None] * ((dz[:, None] * offsets).sum(axis=0) / (dz**2).sum())
        return x[:12].reshape(3, 4), x[12:14], x[14:14 + n_radial], offsets

    def residuals(x):
        P, center, k, offsets = unpack(x)
        shifted = world.copy()
        shifted[:, :2] += offsets[idx]
        return (distort_points(project(P, shifted), center, k) - image).ravel()

    P0 = estimate_projection_matrix(world, image)
    x0 = np.concatenate([P0.ravel() / np.abs(P0).max(), np.asarray(image_size) / 2,
                         np.zeros(n_radial), np.zeros(2 * free.sum())])
    sol = least_squares(residuals, x0, x_scale="jac", loss="soft_l1", f_scale=f_scale)

    P, center, k, offsets = unpack(sol.x)
    P = P / np.linalg.norm(P[2, :3])
    if (_homogeneous(world) @ P[2]).mean() < 0:
        P = -P
    rms = float(np.sqrt((residuals(sol.x) ** 2).reshape(-1, 2).sum(axis=1).mean()))
    return CameraModel(P, center, k, offsets, views, rms)


def model_residuals(model, world, image, view_ids):
    """Distanza in pixel fra punti misurati e punti predetti dal modello, con gli offset delle viste."""
    shifted = np.asarray(world, np.float64).copy()
    shifted[:, :2] += model.offsets[np.searchsorted(model.views, view_ids)]
    predicted = distort_points(project(model.P, shifted), model.center, model.k)
    return np.linalg.norm(predicted - np.asarray(image, np.float64).reshape(-1, 2), axis=1)


def fit_camera_robust(world, image, view_ids, image_size, outlier_factor=10.0, **kwargs):
    """`fit_camera` con un secondo passo che scarta i centri sbagliati; ritorna (modello, inliers).

    Un centro rilevato male (per esempio una colonna spuria al bordo dell'immagine) può avere
    decine di pixel di residuo: la perdita robusta lo attenua ma non lo annulla. Si scartano i punti
    con residuo oltre `outlier_factor` volte la mediana e si rifà il fit sugli altri.
    """
    world = np.asarray(world, np.float64)
    image = np.asarray(image, np.float64).reshape(-1, 2)
    model = fit_camera(world, image, view_ids, image_size, **kwargs)
    residuals = model_residuals(model, world, image, view_ids)
    inliers = residuals <= outlier_factor * np.median(residuals)
    if not inliers.all():
        model = fit_camera(world[inliers], image[inliers], np.asarray(view_ids)[inliers], image_size, **kwargs)
    return model, inliers
