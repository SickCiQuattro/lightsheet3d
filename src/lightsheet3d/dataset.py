"""Descrizione di un dataset: dove stanno le immagini e quali convenzioni di nome e geometria seguono."""
import csv
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    file: str    # nome del file nella cartella delle scansioni
    solid: str   # chiave in `solids.SOLIDS`
    role: str    # "dev" se le scansioni hanno guidato le scelte di algoritmo, "test" altrimenti


@dataclass(frozen=True)
class Dataset:
    """Cartelle e convenzioni di un dataset; si legge da un `dataset.toml`.

    La quota z (mm) di un'immagine di taratura viene dalle prime due cifre del nome del file:
    z = cifre * `z_scale` + `z_offset` (26.jpg, 26l.jpg -> 310 con i valori di default).
    """
    root: Path
    image_size: tuple = (2130, 1420)
    pitch_mm: float = 20.0               # passo dei cerchi del master
    pattern_max: tuple = (21, 14)        # colonne e righe massime della griglia cercata
    pattern_min: int = 9
    z_scale: float = 10.0
    z_offset: float = 50.0
    reference_z: float = 350.0           # quota del master di riferimento
    plate_z: float = 350.0               # quota del piano su cui poggiano i solidi
    target_dir: str = "camera_calibration"
    sheet_dir: str = "light_plane_calibration"
    scans_dir: str = "scans"
    manifest: str | None = None          # CSV file,solid,role con i solidi noti delle scansioni

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = tomllib.loads(path.read_text())
        unknown = set(data) - set(cls.__dataclass_fields__) - {"root"}
        if unknown:
            raise ValueError(f"{path}: chiavi sconosciute {sorted(unknown)}")
        for key in ("image_size", "pattern_max"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(root=path.parent, **data)

    def z_of(self, image_path):
        """Quota (mm) del master in un'immagine di taratura, dal nome del file."""
        match = re.match(r"\d{2}", Path(image_path).name)
        if not match:
            raise ValueError(f"{image_path}: il nome non inizia con due cifre")
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
        """Le scansioni con solido noto, dal manifest (lista vuota se non c'è)."""
        if not self.manifest:
            return []
        with (self.root / self.manifest).open(newline="") as f:
            return [ManifestEntry(r["file"], r["solid"], r.get("role", "test")) for r in csv.DictReader(f)]

    def write(self, path=None):
        """Scrive il `dataset.toml` (nella radice se `path` manca)."""
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
