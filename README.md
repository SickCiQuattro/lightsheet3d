# Light Sheet 3D Scanner

**English** · [Italiano](#italiano)

A triangulation 3D scanner built from a camera and a laser light sheet: camera calibration, light-plane calibration and millimetre measurement of the profile of 3D-printed solids. Final project of the *3D Vision Systems* course (University of Brescia, 2024/25), rewritten as a tested Python package.

![Measured profiles of Corpo 6 and Corpo 5 over their nominal profiles](docs/figures/03_profili.png)

*Red: measured points. Dashed: nominal profile built from the dimensions of the course drawings, aligned to the scan automatically.*

## Results

Two solids with known geometry, measured from one laser image each. The images were taken in a lab with a setup that was not perfectly stable: the errors below are those of that setup.

| Quantity | Result |
|---|---|
| Distance of the measured points from the nominal profile (median / rms) | Corpo 6: 0.53 / 1.04 mm · Corpo 5: 0.69 / 1.14 mm |
| Peak heights, Corpo 6 (nominal 45 / 25 / 45 mm) | 44.6 / 24.2 / 45.2 mm, mean error −0.35 ± 0.42 mm |
| Peak heights, Corpo 5 (nominal 35 / 30 / 45 mm) | 35.4 / 30.3 / 46.1 mm, mean error +0.63 ± 0.36 mm |
| Flank inclination, Corpo 6 | within 5° of the drawing (measured 62.6°, 70.7°, 60.6°, 83.6°, 54.1° vs 60°, 70°, 65°, 80°, 55°) |
| Light plane, error on held-out points | 0.50 mm rms (10% of the points held out) |
| Camera, reprojection residual | median 1.6 px (about 0.4 mm on the master) |

> [!NOTE]
> Two limits are visible in the plot. The right flank of the first peak of Corpo 6 cannot be measured: it is too steep to be lit by the sheet and seen by the camera. On Corpo 5 the heights agree, but the central plateau sits about 3–4 mm to the left of the drawing; I have not found out whether that comes from the printed part or from the scan, and the test only requires a median distance below 1 mm. Both are discussed in [notebooks/03_misure.ipynb](notebooks/03_misure.ipynb).

## How it works

1. **Camera** ([notebook 1](notebooks/01_calibrazione_camera.ipynb)). The master is a grid of circles with three larger ones in an "L" whose corner is the origin. A single model (projection matrix, radial distortion and the master's lateral shift for each view) is fitted by least squares on all views. It works on points, not images, so distortion is never corrected twice.
2. **Light sheet** ([notebook 2](notebooks/02_lama_di_luce.ipynb)). The centroid of the sheet on each row is brought to object space by intersecting the ray with the plane at known `z`; the plane `z = Ax + By + C` is fitted by least squares and validated on held-out points and views.
3. **Measurement** ([notebook 3](notebooks/03_misure.ipynb)). Every sheet pixel on a solid is triangulated with the camera and the plane; the profile is levelled on the support plane, peaks, heights and flank angles are extracted, and the profile is compared with the nominal one.

## Nominal profiles

`lightsheet3d.solids` holds the profile of each solid, built step by step from the dimensions of the drawings (flank run = height / tan(inclination)). The dimensions in the top view of Corpo 6 sum to 100.01 mm and those of Corpo 5 to 75.00 mm, which the tests check. The scan frame is arbitrary, so `align` finds the shift and direction that best put the nominal profile on the measured points, and `distances` measures each point's distance to it in the (y, h) plane, which does not inflate errors on steep flanks.

The drawings themselves belong to the course authors and are not in this repository.

## Layout

```
src/lightsheet3d/
  grid.py         circle grid and origin (corner of the "L")
  calibration.py  normalised DLT, radial distortion, joint model
  laser.py        sheet centroid per row
  lightplane.py   light plane and triangulation
  measure.py      profile levelling, heights and angles
  solids.py       nominal profiles, alignment and distance to the measured profile
  report.py       measured vs nominal plot
  pipeline.py     from raw images to measurements
tests/            synthetic unit tests and regressions on the real data
notebooks/        three notebooks with saved outputs
images/           raw images: target, sheet, object
docs/figures/     figures produced by the notebooks
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or later (developed on 3.15).

```bash
uv sync
uv run pytest        # about 45 s: includes the regressions on the real data
```

Open the notebooks with the project kernel (`ipykernel` is a dev dependency), for example from VS Code. JupyterLab is not included: `pyyaml` has no wheel for Python 3.15 yet.

## Data and licence

Code: [MIT](LICENSE). The images were acquired in the lab. The assignment text and the drawings of the solids are course material (S. Dello Iacono, D. Solinas) and remain the property of their authors.

---

## Italiano

Scanner 3D a triangolazione con telecamera e lama di luce: calibrazione della camera, calibrazione del piano laser e misura in millimetri del profilo di solidi stampati in 3D. Progetto finale del corso *Sistemi di Visione 3D* (Università degli Studi di Brescia, A.A. 2024/25), riscritto come pacchetto Python testato.

### Risultati

Due solidi di geometria nota, misurati ciascuno da una sola immagine della lama. Le immagini sono state scattate in laboratorio con un setup non perfettamente stabile: gli errori sono quelli di quel setup.

| Grandezza | Risultato |
|---|---|
| Distanza dei punti misurati dal profilo nominale (mediana / rms) | Corpo 6: 0,53 / 1,04 mm · Corpo 5: 0,69 / 1,14 mm |
| Altezza dei picchi, Corpo 6 (nominale 45 / 25 / 45 mm) | 44,6 / 24,2 / 45,2 mm, errore medio −0,35 ± 0,42 mm |
| Altezza dei picchi, Corpo 5 (nominale 35 / 30 / 45 mm) | 35,4 / 30,3 / 46,1 mm, errore medio +0,63 ± 0,36 mm |
| Inclinazione dei fianchi, Corpo 6 | entro 5° dal disegno |
| Piano della lama, errore sui punti esclusi | 0,50 mm rms |
| Camera, residuo di ri-proiezione | mediana 1,6 px (circa 0,4 mm sul master) |

Due limiti si vedono nel grafico. Il fianco destro del primo picco del Corpo 6 non è misurabile, perché è troppo ripido per essere illuminato dalla lama e visto dalla camera. Nel Corpo 5 le altezze concordano, ma il piano centrale sta circa 3–4 mm a sinistra del disegno: non ho verificato se dipenda dal pezzo stampato o dalla scansione, e il test chiede solo una distanza mediana sotto 1 mm.

### Profili nominali

`lightsheet3d.solids` contiene il profilo di ogni solido, costruito passo per passo dalle quote dei disegni (corsa del fianco = altezza / tan(inclinazione)). Le quote in pianta del Corpo 6 sommano 100,01 mm e quelle del Corpo 5 75,00 mm, e i test lo verificano. Il riferimento Y della scansione è arbitrario: `align` trova traslazione e verso che portano meglio il profilo nominale sui punti misurati, e `distances` misura la distanza di ogni punto nel piano (y, h), senza amplificare gli errori sui fianchi ripidi.

I disegni appartengono agli autori del corso e non sono in questo repository.

### Setup

Richiede [uv](https://docs.astral.sh/uv/) e Python 3.12 o successivo (sviluppato su 3.15).

```bash
uv sync
uv run pytest        # circa 45 s: include le regressioni sui dati reali
```

### Dati e licenza

Codice: [MIT](LICENSE). Le immagini sono state acquisite in laboratorio. La traccia e i disegni dei solidi sono materiale del corso (S. Dello Iacono, D. Solinas) e restano di proprietà dei rispettivi autori.
