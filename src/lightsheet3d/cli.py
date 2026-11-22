"""Command line: calibrate, scan and generate synthetic data."""
import argparse
import sys
from pathlib import Path

import numpy as np

from .export import distance_colors, write_ply
from .lightplane import plane_errors
from .measure import level_to_plate, measure_peaks
from .pipeline import PLATE_Z, calibrate_camera, calibrate_light_plane, scan_points
from .solids import SOLIDS, align, distances, nominal_curve
from .storage import load_calibration, save_calibration
from .synthetic import Scene, write_dataset


def calibrate(args):
    camera = calibrate_camera(args.target_dir)
    light = calibrate_light_plane(args.laser_dir, camera.model)
    save_calibration(args.output, camera.model, light.plane)
    errors = plane_errors(light.plane, light.points[light.is_validation])
    print(f"camera: views {sorted(int(z) for z in set(camera.view_ids))}, skipped {list(camera.skipped)}, "
          f"rms {camera.model.rms:.2f} px")
    print(f"plane: z = {light.plane[0]:.3f} x + {light.plane[1]:.4f} y + {light.plane[2]:.1f}, "
          f"validation rms {np.sqrt((errors**2).mean()):.3f} mm")
    print(f"calibration written to {args.output}")


def scan(args):
    calibration = load_calibration(args.calibration)
    solid = SOLIDS[args.solid] if args.solid else None
    for image in args.images:
        X = scan_points(image, calibration.model, calibration.plane)
        y, h = X[:, 1], level_to_plate(X[:, 1], PLATE_Z - X[:, 2])
        print(f"{image}: {len(X)} points")
        for i, p in enumerate(measure_peaks(y, h, prominence=args.prominence, min_separation=10), 1):
            print(f"  peak {i}: y = {p.y:6.1f} mm  height {p.height:5.1f} mm  flanks l/r {p.left_angle:5.1f}° / {p.right_angle:5.1f}°")
        colors = None
        if solid:
            sign, shift = align(solid, y, h)
            dist = distances(solid, sign, shift, y, h, pad=np.ptp(y) + 2 * solid.width)
            yn, _ = nominal_curve(solid, sign, shift)
            on_solid = (y > yn[1]) & (y < yn[-2])  # as in the notebooks: only the points on the solid, not on the support plane
            print(f"  comparison with {solid.name}: median distance {np.median(dist[on_solid]):.2f} mm, "
                  f"90th percentile {np.percentile(dist[on_solid], 90):.2f} mm ({on_solid.sum()} points on the solid)")
            colors = distance_colors(dist)
        if args.ply:
            out = Path(args.ply) if len(args.images) == 1 else Path(args.ply) / f"{Path(image).stem}.ply"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_ply(out, np.column_stack([X[:, 0], y, h]), colors)
            print(f"  point cloud written to {out}")


def synthetic(args):
    root, scan_name = write_dataset(Scene(), SOLIDS[args.solid], args.output_dir, noise_px=args.noise, seed=args.seed)
    print(f"synthetic dataset in {root} (scan: {scan_name}, noise {args.noise} px)")


def parser():
    p = argparse.ArgumentParser(prog="lightsheet3d", description="Light-sheet 3D scanner.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="calibrate the camera and the light plane and save the result as JSON")
    c.add_argument("target_dir", help="folder with the master images (e.g. data/camera_calibration)")
    c.add_argument("laser_dir", help="folder with the sheet on the master (e.g. data/light_plane_calibration)")
    c.add_argument("-o", "--output", default="calibration.json")
    c.set_defaults(run=calibrate)

    s = sub.add_parser("scan", help="measure the peaks of one or more images of the sheet on a solid")
    s.add_argument("calibration", help="file written by `calibrate`")
    s.add_argument("images", nargs="+")
    s.add_argument("--solid", choices=sorted(SOLIDS), help="compare with the nominal profile of this solid")
    s.add_argument("--prominence", type=float, default=6.0, help="minimum peak prominence, mm (default 6)")
    s.add_argument("--ply", help="write the points as PLY (one file, or a folder if there are several images)")
    s.set_defaults(run=scan)

    g = sub.add_parser("synthetic", help="generate a synthetic dataset with known geometry")
    g.add_argument("output_dir")
    g.add_argument("--solid", choices=sorted(SOLIDS), default="corpo6")
    g.add_argument("--noise", type=float, default=0.0, help="noise on the centres, pixels (default 0)")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(run=synthetic)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
