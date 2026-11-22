"""Command line: calibrate, scan, evaluate and generate synthetic data."""
import argparse
import sys
from pathlib import Path

from .dataset import Dataset
from .evaluate import evaluate, summarize, write_csv
from .laser import EXTRACTORS
from .scanner import Calibration, Scanner
from .solids import SOLIDS
from .synthetic import Scene, write_dataset


def calibrate(args):
    calibration = Calibration.fit(Dataset.load(args.dataset), args.extractor)
    calibration.save(args.output)
    r = calibration.report
    print(f"camera: views {r['views']}, skipped {r['skipped']}, {r['rejected_centres']} wrong centres discarded, "
          f"rms {r['camera_rms_px']:.2f} px")
    a, b, c = calibration.plane
    print(f"plane: z = {a:.3f} x + {b:.4f} y + {c:.1f}, validation rms {r['plane_validation_rms_mm']:.3f} mm")
    print(f"calibration written to {args.output}")


def scan(args):
    scanner = Scanner(Calibration.load(args.calibration))
    solid = SOLIDS[args.solid] if args.solid else None
    for image in args.images:
        profile = scanner.scan(image)
        print(f"{image}: {len(profile.points)} points")
        for i, p in enumerate(profile.peaks(args.prominence), 1):
            print(f"  peak {i}: y = {p.y:6.1f} mm  height {p.height:5.1f} mm  flanks l/r {p.left_angle:5.1f}° / {p.right_angle:5.1f}°")
        comparison = profile.compare(solid, args.prominence, args.oblique) if solid else None
        if comparison:
            print(f"  comparison with {solid.name}: median distance {comparison.median:.2f} mm, "
                  f"90th percentile {comparison.p90:.2f} mm ({int(comparison.on_solid.sum())} points on the solid)")
            if args.oblique:
                print(f"  oblique cut: Y stretch {comparison.scale:.2f} (tilt {comparison.tilt_deg:.0f} deg)")
        if args.ply:
            out = Path(args.ply) if len(args.images) == 1 else Path(args.ply) / f"{Path(image).stem}.ply"
            out.parent.mkdir(parents=True, exist_ok=True)
            profile.to_ply(out, comparison)
            print(f"  point cloud written to {out}")


def evaluate_command(args):
    dataset = Dataset.load(args.dataset)
    calibration = Calibration.load(args.calibration) if args.calibration else Calibration.fit(dataset, args.extractor)
    results = evaluate(calibration, dataset, oblique=args.oblique)
    if not results:
        raise ValueError(f"{args.dataset}: no manifest with known solids")
    print(f"{'scan':12s} {'solid':7s} {'role':5s} {'points':>6s} {'median':>7s} {'p90':>6s} {'stretch':>7s}  height errors (distances and errors in mm)")
    for r in results:
        print(f"{r.file:12s} {r.solid:7s} {r.role:5s} {r.points:6d} {r.median:7.2f} {r.p90:6.2f} {r.scale:7.2f}  "
              f"{' '.join(f'{e:+.1f}' for e in r.height_errors)}")
    for by in ("role", "solid"):
        print(f"\nby {by}:")
        for key, s in summarize(results, by).items():
            print(f"  {key:8s} {s['scans']:2d} scans  mean median distance {s['median']:.2f} mm (worst {s['worst']:.2f})  "
                  f"height error {s['height_mean']:+.2f} mm, rms {s['height_rms']:.2f} mm")
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nresults written to {args.csv}")


def synthetic(args):
    root, scan_name = write_dataset(Scene(), SOLIDS[args.solid], args.output_dir, noise_px=args.noise, seed=args.seed)
    print(f"synthetic dataset in {root} (scan: {scan_name}, noise {args.noise} px); calibrate it with {root / 'dataset.toml'}")


def parser():
    p = argparse.ArgumentParser(prog="lightsheet3d", description="Light-sheet 3D scanner.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="calibrate the camera and the light plane of a dataset and save the result as JSON")
    c.add_argument("dataset", help="dataset.toml (e.g. data/dataset.toml)")
    c.add_argument("-o", "--output", default="calibration.json")
    c.add_argument("--extractor", choices=sorted(EXTRACTORS), default="centroid", help="how the sheet is found in the images (default centroid)")
    c.set_defaults(run=calibrate)

    s = sub.add_parser("scan", help="measure the peaks of one or more images of the sheet on a solid")
    s.add_argument("calibration", help="file written by `calibrate`")
    s.add_argument("images", nargs="+")
    s.add_argument("--solid", choices=sorted(SOLIDS), help="compare with the nominal profile of this solid")
    s.add_argument("--prominence", type=float, default=6.0, help="minimum peak prominence, mm (default 6)")
    s.add_argument("--oblique", action="store_true", help="let the nominal profile stretch along Y (a solid placed across the sheet)")
    s.add_argument("--ply", help="write the points as PLY (one file, or a folder if there are several images)")
    s.set_defaults(run=scan)

    e = sub.add_parser("evaluate", help="measure every scan listed in the dataset manifest and report the error against the drawings")
    e.add_argument("dataset", help="dataset.toml with a manifest")
    e.add_argument("--calibration", help="use this calibration instead of fitting one")
    e.add_argument("--extractor", choices=sorted(EXTRACTORS), default="centroid")
    e.add_argument("--oblique", action="store_true", help="let the nominal profile stretch along Y (a solid placed across the sheet)")
    e.add_argument("--csv", help="also write the per-scan results as CSV")
    e.set_defaults(run=evaluate_command)

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
