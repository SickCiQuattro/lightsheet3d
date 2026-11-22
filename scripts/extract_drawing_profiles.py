"""Extract the nominal profiles of the solids from the technical drawings in PDF and write them to `profiles.json`.

The drawings are course material and are not in the repository: this script regenerates the profiles
(which are just numbers, the dimensions of the solids) from a folder with SDV3D_Corpo1.pdf ... SDV3D_Corpo6.pdf.

The PDF contains the front profile as a vector path filled with light grey: `pdftocairo -svg`
converts it and its exact vertices are read, curves included. The fill is narrower than the outline by
half a line width per side (about 0.04 mm in total), so it is scaled isotropically to the nominal
width; the isotropic scale also corrects Corpo 4, drawn at 2:1. The depth is the height of the
top view, at the same scale.

Usage: python scripts/extract_drawing_profiles.py PDF_FOLDER src/lightsheet3d/profiles.json
Requires pdftocairo (poppler).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

MM_PER_POINT = 25.4 / 72
FILL = "rgb(94.508362%, 95.292664%, 96.076965%)"  # the grey (241, 243, 245) of the profile in the drawings
WIDTH = {1: 75.0, 2: 75.0, 3: 100.0, 4: 75.0, 5: 75.0, 6: 100.0}  # dimensioned nominal width, mm
SIMPLIFY = 0.01  # mm: tolerance with which arcs are reduced to polylines


def parse_path(d):
    """Subpaths of a cairo path (absolute M, L, C, Z only) as polylines in PDF points."""
    tokens = re.findall(r"[MLCZ]|-?\d+\.?\d*(?:e-?\d+)?", d)
    subpaths, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t == "M":
            pos = start = (float(tokens[i + 1]), float(tokens[i + 2]))
            points = [pos]
            subpaths.append(points)
            i += 3
        elif t == "L":
            pos = (float(tokens[i + 1]), float(tokens[i + 2]))
            points.append(pos)
            i += 3
        elif t == "C":
            p0, p1, p2, p3 = (np.array(q) for q in (pos, *[(float(tokens[i + 1 + 2 * k]), float(tokens[i + 2 + 2 * k])) for k in range(3)]))
            for u in np.linspace(0, 1, max(3, int(np.linalg.norm(p3 - p0) / 0.2) + 2))[1:]:
                points.append(tuple((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u**2 * p2 + u**3 * p3))
            pos = tuple(p3)
            i += 7
        elif t == "Z":
            points.append(start)
            pos = start
            i += 1
        else:
            i += 1
    return [np.array(p) * MM_PER_POINT for p in subpaths if len(p) > 3]


def filled_polygons(svg):
    text = Path(svg).read_text()
    polygons = []
    for match in re.finditer(r"<path ([^>]*)/>", text):
        attrs = match.group(1)
        if FILL in attrs:
            polygons += parse_path(re.search(r' d="([^"]*)"', attrs).group(1))
    return polygons


def extract(svg, width):
    """(outline (n, 2) in mm from (0, 0) to (width, 0) along the top side, depth in mm)."""
    polygons = filled_polygons(svg)
    widest = max(np.ptp(p[:, 0]) for p in polygons)
    wide = [p for p in polygons if np.ptp(p[:, 0]) > widest - 0.3]  # front view and top view
    front, top = max(wide, key=lambda p: p[:, 1].min()), min(wide, key=lambda p: p[:, 1].min())
    scale = width / np.ptp(front[:, 0])
    points = np.column_stack([(front[:, 0] - front[:, 0].min()) * scale, (front[:, 1].max() - front[:, 1]) * scale])
    points = points[np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > 1e-6, axis=1)]]
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    points = np.roll(points, -int(np.argmin(np.hypot(*points.T))), axis=0)  # starts at the corner (0, 0)
    if points[1, 1] < 1e-6:
        points = np.vstack([points[0], points[1:][::-1]])  # and climbs up the left side
    end = int(np.argmin(np.hypot(points[:, 0] - width, points[:, 1])))
    outline = cv.approxPolyDP(points[:end + 1].astype(np.float32).reshape(-1, 1, 2), SIMPLIFY, False).reshape(-1, 2)
    return outline.astype(float), float(np.ptp(top[:, 1]) * scale)


def main(pdf_dir, output):
    profiles = {}
    with tempfile.TemporaryDirectory() as tmp:
        for number, width in WIDTH.items():
            svg = Path(tmp) / f"corpo{number}.svg"
            subprocess.run(["pdftocairo", "-svg", str(Path(pdf_dir) / f"SDV3D_Corpo{number}.pdf"), str(svg)], check=True)
            outline, depth = extract(svg, width)
            profiles[f"corpo{number}"] = {"name": f"Corpo {number}", "depth": round(depth, 2),
                                          "outline": np.round(outline, 3).tolist()}
            print(f"Corpo {number}: {len(outline)} vertices, height {outline[:, 1].max():.2f} mm, depth {depth:.2f} mm")
    Path(output).write_text(json.dumps(profiles, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
