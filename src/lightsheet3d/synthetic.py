"""Immagini sintetiche di uno scanner a lama di luce con geometria nota (verità a terra).

La scena di default ha i parametri stimati sul setup reale (camera a circa 330 mm dal master,
focale 1955 px, lama inclinata nel piano x-z). Le immagini hanno gli stessi nomi e lo stesso
formato di quelle reali, quindi la pipeline le legge senza modifiche.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import distort_points, project
from .measure import level_to_plate
from .pipeline import PLATE_Z, Z_BASE, Z_STEP, calibrate_camera, calibrate_light_plane, scan_points
from .solids import align, distances, nominal_curve

FOCAL = 1955.0                   # px, per la dimensione apparente dei cerchi
LARGE = {(0.0, 0.0), (60.0, 0.0), (0.0, 40.0)}  # "L" di cerchi grandi: origine, +x, +y (mm)
MARGIN = 90                      # px dal bordo entro cui un cerchio è considerato visibile
VIEWS = range(310, 351, 10)      # z (mm) delle viste del master, come nel setup reale


@dataclass(frozen=True)
class Scene:
    P: np.ndarray = field(default_factory=lambda: np.array([[1948.493, -105.702, 747.635, 45495.232],
                                                            [119.224, 1960.919, 338.122, 93671.343],
                                                            [-0.00323, 0.01342, 0.9999, -17.508]]))
    center: np.ndarray = field(default_factory=lambda: np.array([875.559, 662.222]))  # centro distorsione radiale, px
    k: np.ndarray = field(default_factory=lambda: np.array([-0.10885, 0.01054]))     # coefficienti radiali
    plane: np.ndarray = field(default_factory=lambda: np.array([2.89308, 0.0081, 216.731]))  # lama: z = A x + B y + C
    image_size: tuple = (2130, 1420)
    pitch: float = 20.0                                # mm fra i cerchi del master

    def to_pixels(self, world):
        return distort_points(project(self.P, np.asarray(world, np.float64)), self.center, self.k)

    def sheet_x(self, y, z):
        """x (mm) in cui la lama passa per (y, z)."""
        a, b, c = self.plane
        return (z - b * y - c) / a

    def visible_grid(self):
        """Coordinate (mm) di righe e colonne di un blocco di master interamente visibile a ogni z.

        Parte dalla griglia completa e toglie di volta in volta la riga o la colonna con più
        cerchi fuori campo (o troppo vicini al bordo).
        """
        w, h = self.image_size
        xs, ys = np.arange(-140, 241, self.pitch), np.arange(-100, 121, self.pitch)
        while True:
            outside = np.zeros((len(ys), len(xs)), bool)
            for z in (VIEWS[0], VIEWS[-1]):
                uv = self.to_pixels([(x, y, z) for y in ys for x in xs]).reshape(len(ys), len(xs), 2)
                outside |= ~((uv[..., 0] > MARGIN) & (uv[..., 0] < w - MARGIN) & (uv[..., 1] > MARGIN) & (uv[..., 1] < h - MARGIN))
            if not outside.any():
                return xs, ys
            by_col, by_row = outside.sum(axis=0), outside.sum(axis=1)
            if by_col.max() >= by_row.max():
                xs = np.delete(xs, by_col.argmax())
            else:
                ys = np.delete(ys, by_row.argmax())


def _blank(scene):
    w, h = scene.image_size
    return np.full((h, w, 3), 200, np.uint8)


def _sub(v):
    """Coordinata subpixel (4 bit frazionari) per le primitive di OpenCV."""
    return round(v * 16)


def render_master(scene, z, noise_px=0.0, rng=None):
    """Master di taratura a quota `z`: cerchi neri su carta, con la "L" di cerchi grandi."""
    rng = rng or np.random.default_rng(0)
    image = _blank(scene)
    xs, ys = scene.visible_grid()
    for x in xs:
        for y in ys:
            u, v = scene.to_pixels([(x, y, z)])[0] + rng.normal(0, noise_px, 2)
            r = (5.0 if (x, y) in LARGE else 3.0) * FOCAL / (scene.P[2] @ [x, y, z, 1])
            cv.circle(image, (_sub(u), _sub(v)), _sub(r), (40, 40, 40), -1, cv.LINE_AA, shift=4)
    return image


def _draw_sheet(scene, world, noise_px, rng):
    """Lama (saturata su tutti i canali, come nelle foto reali) lungo la curva 3D `world`."""
    image = _blank(scene)
    uv = scene.to_pixels(world) + rng.normal(0, noise_px, (len(world), 2))
    cv.polylines(image, [np.array([[_sub(u), _sub(v)] for u, v in uv])], False, (255, 255, 255), 5, cv.LINE_AA, shift=4)
    return image


def render_sheet_on_master(scene, z, noise_px=0.0, rng=None):
    """Lama sul master piano a quota `z`."""
    rng = rng or np.random.default_rng(0)
    y = np.arange(-130, 151, 1.0)
    return _draw_sheet(scene, np.column_stack([scene.sheet_x(y, z), y, np.full(len(y), float(z))]), noise_px, rng)


def render_scan(scene, solid, sign=1, shift=0.0, noise_px=0.0, rng=None):
    """Lama su un solido appoggiato al piano z = PLATE_Z; il profilo nominale è la verità a terra."""
    rng = rng or np.random.default_rng(0)
    yn, hn = nominal_curve(solid, sign, shift)
    y = np.concatenate([[yn[0] - 60], yn, [yn[-1] + 60]])
    z = PLATE_Z - np.concatenate([[0.0], hn, [0.0]])
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.hypot(*(b - a)))))
                            for a, b in zip(np.column_stack([y, z])[:-1], np.column_stack([y, z])[1:])])
    y, z = dense[:, 0], dense[:, 1]
    return _draw_sheet(scene, np.column_stack([scene.sheet_x(y, z), y, z]), noise_px, rng)


def write_dataset(scene, solid, out_dir, sign=1, shift=-57.0, noise_px=0.0, seed=0):
    """Scrive un dataset con la struttura di `data/`; restituisce (cartella, nome della scansione)."""
    out, rng = Path(out_dir), np.random.default_rng(seed)
    names = {"camera_calibration": "{n}.jpg", "light_plane_calibration": "{n}l.jpg"}
    for sub in (*names, "scans"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for z in VIEWS:
        n = (z - Z_BASE) // Z_STEP
        for sub, render in (("camera_calibration", render_master), ("light_plane_calibration", render_sheet_on_master)):
            cv.imwrite(str(out / sub / names[sub].format(n=n)), render(scene, z, noise_px, rng), [cv.IMWRITE_JPEG_QUALITY, 95])
    scan = "30o1.jpg"
    cv.imwrite(str(out / "scans" / scan), render_scan(scene, solid, sign, shift, noise_px, rng), [cv.IMWRITE_JPEG_QUALITY, 95])
    return out, scan


def evaluate(solid, work_dir, noise_px=0.0, shift=-57.0, seed=0, scene=None):
    """Genera un dataset, esegue la pipeline e confronta il profilo con la verità a terra.

    Restituisce un dizionario con `camera`, `light`, il profilo (`y`, `h`), il verso e la traslazione
    trovati da `align` e la distanza (mm) di ogni punto dal profilo nominale (`dist`).
    """
    scene = scene or Scene()
    root, scan = write_dataset(scene, solid, work_dir, shift=shift, noise_px=noise_px, seed=seed)
    camera = calibrate_camera(root / "camera_calibration")
    light = calibrate_light_plane(root / "light_plane_calibration", camera.model)
    X = scan_points(root / "scans" / scan, camera.model, light.plane)
    y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])
    sign, found = align(solid, y, h)
    yn, _ = nominal_curve(solid, sign, found)
    on_solid = (y > yn[1]) & (y < yn[-2])
    return {"scene": scene, "camera": camera, "light": light, "y": y, "h": h, "sign": sign, "shift": found,
            "dist": distances(solid, sign, found, y[on_solid], h[on_solid])}
