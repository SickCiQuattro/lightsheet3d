"""Livello di uso: calibrare un dataset, misurare una scansione e confrontarla con un solido noto.

    dataset = Dataset.load("data/dataset.toml")
    calibration = Calibration.fit(dataset)
    profile = Scanner(calibration).scan("data/scans/30o34.jpg")
    profile.peaks()
    comparison = profile.compare(SOLIDS["corpo6"])
    profile.to_ply("corpo6.ply", comparison)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import CameraModel
from .export import distance_colors, write_ply
from .laser import EXTRACTORS
from .lightplane import plane_errors
from .measure import Peak, level_to_plate, measure_peaks
from .pipeline import PLATE_Z, calibrate_camera, calibrate_light_plane, scan_points
from .solids import align, align_oblique, distances, nominal_curve

VERSION = 1


@dataclass(frozen=True)
class Calibration:
    """Camera e piano della lama, più il metodo con cui si trova la lama e la quota del piano d'appoggio."""
    model: CameraModel
    plane: np.ndarray                 # (A, B, C) di z = A x + B y + C
    extractor: str = "centroid"       # chiave di `laser.EXTRACTORS`
    plate_z: float = PLATE_Z
    report: dict = field(default_factory=dict, compare=False)  # diagnostica del fit, non salvata

    @classmethod
    def fit(cls, dataset, extractor="centroid"):
        """Calibra camera e piano della lama con le immagini e le convenzioni di `dataset`."""
        camera = calibrate_camera(dataset.target_path, z_max=dataset.reference_z, pitch=dataset.pitch_mm,
                                  image_size=dataset.image_size, z_of=dataset.z_of,
                                  pattern_max=dataset.pattern_max, pattern_min=dataset.pattern_min)
        light = calibrate_light_plane(dataset.sheet_path, camera.model, z_max=dataset.reference_z,
                                      extract=EXTRACTORS[extractor], z_of=dataset.z_of)
        errors = plane_errors(light.plane, light.points[light.is_validation])
        report = {"views": sorted(int(z) for z in set(camera.view_ids)), "skipped": list(camera.skipped),
                  "rejected_centres": int((~camera.inliers).sum()), "camera_rms_px": camera.model.rms,
                  "plane_validation_rms_mm": float(np.sqrt((errors**2).mean()))}
        return cls(camera.model, light.plane, extractor, dataset.plate_z, report)

    def save(self, path):
        m = self.model
        data = {"version": VERSION, "P": m.P.tolist(), "center": m.center.tolist(), "k": m.k.tolist(),
                "offsets": m.offsets.tolist(), "views": m.views.tolist(), "rms": m.rms,
                "plane": np.asarray(self.plane, float).tolist(), "extractor": self.extractor, "plate_z": self.plate_z}
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("version") != VERSION:
            raise ValueError(f"{path}: versione {data.get('version')} non supportata (attesa {VERSION})")
        model = CameraModel(np.array(data["P"]), np.array(data["center"]), np.array(data["k"]),
                            np.array(data["offsets"]), np.array(data["views"]), float(data["rms"]))
        return cls(model, np.array(data["plane"]), data.get("extractor", "centroid"), data.get("plate_z", PLATE_Z))


@dataclass(frozen=True)
class Comparison:
    """Confronto di un profilo misurato con quello nominale di un solido."""
    solid: object                     # solids.Solid
    sign: int                         # verso dell'asse Y della scansione rispetto al solido
    shift: float                      # mm
    scale: float                      # allungamento di Y del profilo nominale (1 = taglio perpendicolare)
    distance: np.ndarray              # (n,) distanza di ogni punto dal profilo nominale, mm
    on_solid: np.ndarray              # (n,) bool: punti sopra il solido, non sul piano d'appoggio
    peaks: list                       # [(Peak misurato, altezza nominale)] se i picchi sono quanti quelli nominali

    @property
    def median(self):
        return float(np.median(self.distance[self.on_solid]))

    @property
    def p90(self):
        return float(np.percentile(self.distance[self.on_solid], 90))

    @property
    def rms(self):
        return float(np.sqrt((self.distance[self.on_solid] ** 2).mean()))

    @property
    def tilt_deg(self):
        """Angolo (gradi) fra la lama e la normale all'asse del solido, dedotto da `scale`: 0 se il taglio è perpendicolare."""
        return float(np.degrees(np.arccos(min(1.0, 1.0 / self.scale))))

    @property
    def height_errors(self):
        """Altezza misurata meno nominale di ogni picco, mm (vuoto se i picchi non corrispondono)."""
        return np.array([peak.height - nominal for peak, nominal in self.peaks])


@dataclass(frozen=True)
class Profile:
    """Profilo di una scansione: punti 3D, e altezza dal piano d'appoggio lungo Y."""
    image: str
    points: np.ndarray                # (n, 3) mm nel riferimento del master
    height: np.ndarray                # (n,) mm sopra il piano d'appoggio (livellato)

    @property
    def y(self):
        return self.points[:, 1]

    def peaks(self, prominence=6.0, min_separation=10.0):
        return measure_peaks(self.y, self.height, prominence=prominence, min_separation=min_separation)

    def compare(self, solid, prominence=6.0, oblique=False):
        """Allinea il profilo nominale di `solid` alla scansione e misura la distanza di ogni punto.

        Con `oblique=True` il profilo nominale può anche allungarsi lungo Y, come succede se il solido è
        posato di traverso rispetto alla lama; `Comparison.scale` e `tilt_deg` dicono di quanto.
        """
        if oblique:
            sign, shift, scale = align_oblique(solid, self.y, self.height)
        else:
            (sign, shift), scale = align(solid, self.y, self.height), 1.0
        yn, _ = nominal_curve(solid, sign, shift, scale=scale)
        distance = distances(solid, sign, shift, self.y, self.height, pad=(np.ptp(self.y) + 2 * solid.width) * scale, scale=scale)
        on_solid = (self.y > yn[1]) & (self.y < yn[-2])
        found = sorted(self.peaks(prominence), key=lambda p: p.y)
        nominal = [h for _, h in sorted((sign * x + shift, h) for x, h in solid.peaks())]
        pairs = list(zip(found, nominal)) if len(found) == len(nominal) else []
        return Comparison(solid, sign, shift, scale, distance, on_solid, pairs)

    def to_ply(self, path, comparison=None):
        """Punti (x, y, altezza) in PLY, colorati con la distanza dal profilo nominale se c'è `comparison`."""
        colors = None if comparison is None else distance_colors(comparison.distance)
        write_ply(path, np.column_stack([self.points[:, 0], self.y, self.height]), colors)


class Scanner:
    def __init__(self, calibration):
        self.calibration = calibration

    def scan(self, image_path):
        c = self.calibration
        X = scan_points(image_path, c.model, c.plane, extract=EXTRACTORS[c.extractor])
        return Profile(str(image_path), X, level_to_plate(X[:, 1], c.plate_z - X[:, 2]))


__all__ = ["Calibration", "Comparison", "Peak", "Profile", "Scanner"]
