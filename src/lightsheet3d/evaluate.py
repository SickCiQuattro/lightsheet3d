"""Valutazione su tutte le scansioni con solido noto di un dataset (il manifest `scans.csv`)."""
import csv
from dataclasses import dataclass

import numpy as np

from .scanner import Scanner
from .solids import SOLIDS


@dataclass(frozen=True)
class ScanResult:
    file: str
    solid: str
    role: str                  # "dev" o "test": vedi `dataset.ManifestEntry`
    points: int                # punti sul solido
    median: float              # mm, distanza mediana dal profilo nominale
    p90: float
    rms: float
    height_errors: tuple       # mm, per picco; vuoto se i picchi misurati non sono quanti i nominali
    scale: float = 1.0         # allungamento di Y trovato (1 senza `oblique`)


def evaluate(calibration, dataset, solids=SOLIDS, prominence=6.0, oblique=False):
    scanner = Scanner(calibration)
    results = []
    for entry in dataset.entries():
        comparison = scanner.scan(dataset.scans_path / entry.file).compare(solids[entry.solid], prominence, oblique)
        results.append(ScanResult(entry.file, entry.solid, entry.role, int(comparison.on_solid.sum()), comparison.median,
                                  comparison.p90, comparison.rms, tuple(float(e) for e in comparison.height_errors),
                                  comparison.scale))
    return results


def summarize(results, by="role"):
    """{gruppo: {scans, median (media delle mediane), worst, height_mean, height_rms}} per `role` o `solid`."""
    groups = {}
    for r in results:
        groups.setdefault(getattr(r, by), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        errors = np.array([e for r in rs for e in r.height_errors])
        out[key] = {"scans": len(rs), "median": float(np.mean([r.median for r in rs])), "worst": float(max(r.median for r in rs)),
                    "height_mean": float(errors.mean()) if errors.size else float("nan"),
                    "height_rms": float(np.sqrt((errors**2).mean())) if errors.size else float("nan")}
    return out


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "solid", "role", "points", "median_mm", "p90_mm", "rms_mm", "height_errors_mm", "scale"])
        for r in results:
            writer.writerow([r.file, r.solid, r.role, r.points, f"{r.median:.3f}", f"{r.p90:.3f}", f"{r.rms:.3f}",
                             " ".join(f"{e:+.2f}" for e in r.height_errors), f"{r.scale:.3f}"])
