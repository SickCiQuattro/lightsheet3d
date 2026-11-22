"""Evaluation on all the scans of a dataset with a known solid (the `scans.csv` manifest)."""
import csv
from dataclasses import dataclass

import numpy as np

from .scanner import Scanner
from .solids import SOLIDS


@dataclass(frozen=True)
class ScanResult:
    file: str
    solid: str
    role: str                  # "dev" or "test": see `dataset.ManifestEntry`
    points: int                # points on the solid
    median: float              # mm, median distance from the nominal profile
    p90: float
    rms: float
    height_errors: tuple       # mm, per peak; empty if the measured peaks are not as many as the nominal ones
    scale: float = 1.0         # Y stretch found (1 without `oblique`)


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
    """{group: {scans, median (mean of the medians), worst, height_mean, height_rms}} per `role` or `solid`."""
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
