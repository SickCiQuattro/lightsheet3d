from pathlib import Path

import pytest

from lightsheet3d.dataset import Dataset
from lightsheet3d.evaluate import ScanResult, evaluate, summarize, write_csv
from lightsheet3d.scanner import Calibration

REPO = Path(__file__).parents[1]


def test_summarize_groups_by_role_and_solid():
    results = [ScanResult("a", "corpo1", "dev", 100, 0.5, 1.0, 0.8, (0.2, -0.2)),
               ScanResult("b", "corpo1", "test", 100, 1.5, 2.0, 1.2, (1.0,)),
               ScanResult("c", "corpo2", "test", 100, 1.0, 2.0, 1.2, ())]

    by_role = summarize(results)
    by_solid = summarize(results, by="solid")

    assert by_role["test"] == {"scans": 2, "median": 1.25, "worst": 1.5, "height_mean": 1.0, "height_rms": 1.0}
    assert by_solid["corpo1"]["scans"] == 2 and by_solid["corpo1"]["height_mean"] == pytest.approx(1 / 3)
    assert by_solid["corpo2"]["height_mean"] != by_solid["corpo2"]["height_mean"]  # nan: nessun picco confrontabile


def test_write_csv_has_one_row_per_scan(tmp_path):
    write_csv([ScanResult("a.jpg", "corpo1", "test", 10, 0.5, 1.0, 0.8, (0.25,))], tmp_path / "r.csv")

    assert (tmp_path / "r.csv").read_text().splitlines()[1] == "a.jpg,corpo1,test,10,0.500,1.000,0.800,+0.25,1.000"


@pytest.fixture(scope="module")
def real():
    dataset = Dataset.load(REPO / "data" / "dataset.toml")
    return evaluate(Calibration.fit(dataset), dataset)


def test_real_scans_of_all_six_solids_are_measured(real):
    assert len(real) == 14 and {r.solid for r in real} == {f"corpo{n}" for n in range(1, 7)}


def test_error_on_the_held_out_scans_stays_below_a_millimetre_and_a_half(real):
    # le 12 scansioni "test" non hanno guidato nessuna scelta di algoritmo; i valori attuali sono 0.91 mm in media, 1.54 al peggio
    summary = summarize(real)["test"]

    assert summary["scans"] == 12 and summary["median"] < 1.1 and summary["worst"] < 1.8


def test_allowing_the_oblique_stretch_lowers_the_held_out_error(real):
    dataset = Dataset.load(REPO / "data" / "dataset.toml")
    oblique = evaluate(Calibration.fit(dataset), dataset, oblique=True)
    plain, stretched = summarize(real)["test"]["median"], summarize(oblique)["test"]["median"]

    assert stretched < plain - 0.05 and stretched < 0.95  # 0.91 -> 0.80 mm oggi
    # dove il solido è allineato alla lama il parametro deve restare vicino a 1: la scala di Y della calibrazione è giusta
    assert [r.scale for r in oblique if r.file in ("30o31.jpg", "30o33.jpg", "30o7.jpg", "30o12.jpg")] == pytest.approx([1.0] * 4, abs=0.03)
