"""Description of a dataset: where the images are and which naming and geometry conventions they follow."""
import csv
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    file: str    # file name in the scans folder
    solid: str   # key in `solids.SOLIDS`
    role: str    # "dev" if the scans guided the algorithm choices, "test" otherwise


@dataclass(frozen=True)
class Dataset:
    """Folders and conventions of a dataset; read from a `dataset.toml`.

    The z height (mm) of a calibration image comes from the first two digits of the file name:
    z = digits * `z_scale` + `z_offset` (26.jpg, 26l.jpg -> 310 with the default values).
    """
    root: Path
    image_size: tuple = (2130, 1420)
    pitch_mm: float = 20.0               # pitch of the master's circles
    pattern_max: tuple = (21, 14)        # maximum columns and rows of the grid searched for
    pattern_min: int = 9
    z_scale: float = 10.0
    z_offset: float = 50.0
    reference_z: float = 350.0           # height of the reference master
    plate_z: float = 350.0               # height of the plane the solids rest on
    target_dir: str = "camera_calibration"
    sheet_dir: str = "light_plane_calibration"
    scans_dir: str = "scans"
    manifest: str | None = None          # CSV file,solid,role with the known solids of the scans

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = tomllib.loads(path.read_text())
        unknown = set(data) - set(cls.__dataclass_fields__) - {"root"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        for key in ("image_size", "pattern_max"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(root=path.parent, **data)

    def z_of(self, image_path):
        """Height (mm) of the master in a calibration image, from the file name."""
        match = re.match(r"\d{2}", Path(image_path).name)
        if not match:
            raise ValueError(f"{image_path}: the name does not start with two digits")
        return int(match.group()) * self.z_scale + self.z_offset

    @property
    def target_path(self):
        return self.root / self.target_dir

    @property
    def sheet_path(self):
        return self.root / self.sheet_dir

    @property
    def scans_path(self):
        return self.root / self.scans_dir

    def entries(self):
        """The scans with a known solid, from the manifest (empty list if there is none)."""
        if not self.manifest:
            return []
        with (self.root / self.manifest).open(newline="") as f:
            return [ManifestEntry(r["file"], r["solid"], r.get("role", "test")) for r in csv.DictReader(f)]

    def write(self, path=None):
        """Write the `dataset.toml` (in the root if `path` is missing)."""
        lines = [f"image_size = [{self.image_size[0]}, {self.image_size[1]}]",
                 f"pitch_mm = {self.pitch_mm}",
                 f"pattern_max = [{self.pattern_max[0]}, {self.pattern_max[1]}]",
                 f"pattern_min = {self.pattern_min}",
                 f"z_scale = {self.z_scale}", f"z_offset = {self.z_offset}",
                 f"reference_z = {self.reference_z}", f"plate_z = {self.plate_z}",
                 f'target_dir = "{self.target_dir}"', f'sheet_dir = "{self.sheet_dir}"', f'scans_dir = "{self.scans_dir}"']
        if self.manifest:
            lines.append(f'manifest = "{self.manifest}"')
        Path(path or self.root / "dataset.toml").write_text("\n".join(lines) + "\n")
