"""Synthetic images of a light-sheet scanner with known geometry (ground truth).

The default scene has parameters estimated on the real setup (camera about 330 mm from the master,
focal length 1955 px, sheet tilted in the x-z plane). The images have the same names and the same
format as the real ones, so the pipeline reads them unchanged.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import distort_points, project
from .dataset import Dataset
from .laser import laser_centroids
from .measure import level_to_plate
from .pipeline import PLATE_Z, Z_BASE, Z_STEP, calibrate_camera, calibrate_light_plane, scan_points
from .solids import SOLIDS, align, distances, nominal_curve

FOCAL = 1955.0                   # px, for the apparent size of the circles
LARGE = {(0.0, 0.0), (60.0, 0.0), (0.0, 40.0)}  # "L" of large circles: origin, +x, +y (mm)
MARGIN = 90                      # px from the edge within which a circle is considered visible
VIEWS = range(310, 351, 10)      # z (mm) of the master views, as in the real setup


@dataclass(frozen=True)
class Scene:
    P: np.ndarray = field(default_factory=lambda: np.array([[1948.493, -105.702, 747.635, 45495.232],
                                                            [119.224, 1960.919, 338.122, 93671.343],
                                                            [-0.00323, 0.01342, 0.9999, -17.508]]))
    center: np.ndarray = field(default_factory=lambda: np.array([875.559, 662.222]))  # radial distortion centre, px
    k: np.ndarray = field(default_factory=lambda: np.array([-0.10885, 0.01054]))     # radial coefficients
    plane: np.ndarray = field(default_factory=lambda: np.array([2.89308, 0.0081, 216.731]))  # sheet: z = A x + B y + C
    image_size: tuple = (2130, 1420)
    pitch: float = 20.0                                # mm between the master's circles

    def to_pixels(self, world):
        return distort_points(project(self.P, np.asarray(world, np.float64)), self.center, self.k)

    def sheet_x(self, y, z):
        """x (mm) at which the sheet passes through (y, z)."""
        a, b, c = self.plane
        return (z - b * y - c) / a

    def visible_grid(self):
        """Coordinates (mm) of rows and columns of a master block fully visible at every z.

        Starts from the full grid and each time removes the row or column with the most
        circles out of the field (or too close to the edge).
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
    """Subpixel coordinate (4 fractional bits) for the OpenCV primitives."""
    return round(v * 16)


def render_master(scene, z, noise_px=0.0, rng=None):
    """Calibration master at height `z`: black circles on paper, with the "L" of large circles."""
    rng = rng or np.random.default_rng(0)
    image = _blank(scene)
    xs, ys = scene.visible_grid()
    for x in xs:
        for y in ys:
            u, v = scene.to_pixels([(x, y, z)])[0] + rng.normal(0, noise_px, 2)
            r = (5.0 if (x, y) in LARGE else 3.0) * FOCAL / (scene.P[2] @ [x, y, z, 1])
            cv.circle(image, (_sub(u), _sub(v)), _sub(r), (40, 40, 40), -1, cv.LINE_AA, shift=4)
    return image


def _draw_sheet(scene, world, noise_px, rng, blur=0.0, grain=0.0):
    """Sheet (saturated on all channels, as in the real photos) along the 3D curve `world`.

    `blur` (px, standard deviation) blurs the sheet like real optics; `grain` (grey levels)
    adds Gaussian noise to the intensities. Both leave the geometry intact.
    """
    image = _blank(scene)
    uv = scene.to_pixels(world) + rng.normal(0, noise_px, (len(world), 2))
    cv.polylines(image, [np.array([[_sub(u), _sub(v)] for u, v in uv])], False, (255, 255, 255), 5, cv.LINE_AA, shift=4)
    if blur:
        image = cv.GaussianBlur(image, (0, 0), blur)
    if grain:
        image = np.clip(image + rng.normal(0, grain, image.shape), 0, 255).astype(np.uint8)
    return image


def render_sheet_on_master(scene, z, noise_px=0.0, rng=None, blur=0.0, grain=0.0):
    """Sheet on the flat master at height `z`."""
    rng = rng or np.random.default_rng(0)
    y = np.arange(-130, 151, 1.0)
    return _draw_sheet(scene, np.column_stack([scene.sheet_x(y, z), y, np.full(len(y), float(z))]), noise_px, rng, blur, grain)


def render_scan(scene, solid, sign=1, shift=0.0, noise_px=0.0, rng=None, blur=0.0, grain=0.0):
    """Sheet on a solid resting on the plane z = PLATE_Z; the nominal profile is the ground truth."""
    rng = rng or np.random.default_rng(0)
    yn, hn = nominal_curve(solid, sign, shift)
    y = np.concatenate([[yn[0] - 60], yn, [yn[-1] + 60]])
    z = PLATE_Z - np.concatenate([[0.0], hn, [0.0]])
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.hypot(*(b - a)))))
                            for a, b in zip(np.column_stack([y, z])[:-1], np.column_stack([y, z])[1:])])
    y, z = dense[:, 0], dense[:, 1]
    return _draw_sheet(scene, np.column_stack([scene.sheet_x(y, z), y, z]), noise_px, rng, blur, grain)


def write_dataset(scene, solid, out_dir, sign=1, shift=-57.0, noise_px=0.0, seed=0, blur=0.0, grain=0.0):
    """Write a dataset with the structure of `data/`; returns (folder, scan name)."""
    out, rng = Path(out_dir), np.random.default_rng(seed)
    names = {"camera_calibration": "{n}.jpg", "light_plane_calibration": "{n}l.jpg"}
    for sub in (*names, "scans"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for z in VIEWS:
        n = (z - Z_BASE) // Z_STEP
        for sub, image in (("camera_calibration", render_master(scene, z, noise_px, rng)),
                           ("light_plane_calibration", render_sheet_on_master(scene, z, noise_px, rng, blur, grain))):
            cv.imwrite(str(out / sub / names[sub].format(n=n)), image, [cv.IMWRITE_JPEG_QUALITY, 95])
    scan = "30o1.jpg"
    image = render_scan(scene, solid, sign, shift, noise_px, rng, blur, grain)
    cv.imwrite(str(out / "scans" / scan), image, [cv.IMWRITE_JPEG_QUALITY, 95])
    key = next((k for k, v in SOLIDS.items() if v is solid), None)
    if key:  # the manifest with the truth: the scan is of this solid
        (out / "scans.csv").write_text(f"file,solid,role\n{scan},{key},test\n")
    Dataset(root=out, image_size=scene.image_size, pitch_mm=scene.pitch, manifest="scans.csv" if key else None).write()
    return out, scan


def evaluate(solid, work_dir, noise_px=0.0, shift=-57.0, seed=0, scene=None, blur=0.0, grain=0.0, extract=laser_centroids):
    """Generate a dataset, run the pipeline and compare the profile with the ground truth.

    Returns a dictionary with `camera`, `light`, the profile (`y`, `h`), the direction and shift
    found by `align` and the distance (mm) of each point from the nominal profile (`dist`). `extract` is the
    function that finds the sheet in an image (`laser_centroids` or `laser_steger`).
    """
    scene = scene or Scene()
    root, scan = write_dataset(scene, solid, work_dir, shift=shift, noise_px=noise_px, seed=seed, blur=blur, grain=grain)
    camera = calibrate_camera(root / "camera_calibration")
    light = calibrate_light_plane(root / "light_plane_calibration", camera.model, extract=extract)
    X = scan_points(root / "scans" / scan, camera.model, light.plane, extract=extract)
    y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])
    sign, found = align(solid, y, h)
    yn, _ = nominal_curve(solid, sign, found)
    on_solid = (y > yn[1]) & (y < yn[-2])
    return {"scene": scene, "camera": camera, "light": light, "y": y, "h": h, "sign": sign, "shift": found,
            "dist": distances(solid, sign, found, y[on_solid], h[on_solid])}
