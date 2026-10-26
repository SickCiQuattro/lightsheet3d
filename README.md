# Light Sheet 3D Scanner

Scanner 3D a triangolazione con telecamera e lama di luce: calibrazione della camera, calibrazione del piano laser e misura in millimetri del profilo di solidi stampati in 3D. Progetto finale del corso *Sistemi di Visione 3D* (Università degli Studi di Brescia, A.A. 2024/25), rivisto e riscritto come pacchetto Python testato.

![Profili misurati dei Corpi 6 e 5](docs/figures/03_profili.png)

## Risultati

Misurati sui due solidi con disegno di riferimento ([docs/disegni](docs/disegni)). Le immagini sono state scattate in laboratorio con un setup non perfettamente stabile: gli errori sono quelli di quel setup.

| Grandezza | Risultato |
|---|---|
| Altezza dei picchi, Corpo 6 (45 / 25 / 45 mm) | 44.6 / 24.2 / 45.2 mm, errore medio −0.35 ± 0.42 mm |
| Altezza dei picchi, Corpo 5 (35 / 30 / 45 mm) | 35.4 / 30.3 / 46.1 mm, errore medio +0.63 ± 0.36 mm |
| Inclinazione dei fianchi, Corpo 6 | entro 5° dal disegno (misurati 62.6°, 70.7°, 60.6°, 83.6°, 54.1° contro 60°, 70°, 65°, 80°, 55°) |
| Piano della lama, scarto sui punti di validazione | 0.50 mm rms (10% dei punti esclusi) |
| Camera, residuo di ri-proiezione | mediana 1.6 px (circa 0.4 mm sul master) |

> [!NOTE]
> Il fianco destro del primo picco del Corpo 6 non è misurabile: è troppo ripido per essere illuminato dalla lama e visto dalla camera. Il limite è del setup e non dell'algoritmo; è discusso in [notebooks/03_misure.ipynb](notebooks/03_misure.ipynb).

## Come funziona

1. **Camera** ([notebook 1](notebooks/01_calibrazione_camera.ipynb)). Il master è una griglia di cerchi con tre cerchi più grandi a "L", il cui angolo è l'origine. Un unico modello (matrice di proiezione, distorsione radiale e spostamento laterale del master per ogni vista) si stima con i minimi quadrati su tutte le viste. Si lavora sui punti, non sulle immagini, quindi non c'è doppia correzione della distorsione.
2. **Lama di luce** ([notebook 2](notebooks/02_lama_di_luce.ipynb)). Il baricentro della lama su ogni riga si porta nello spazio oggetto intersecando il raggio con il piano a `z` noto; il piano `z = Ax + By + C` si stima ai minimi quadrati e si valida su punti esclusi e su viste escluse.
3. **Misura** ([notebook 3](notebooks/03_misure.ipynb)). Ogni pixel della lama su un solido si triangola con la camera e il piano; il profilo si livella sul piano di appoggio e se ne ricavano altezze e inclinazioni.

## Struttura

```
src/lightsheet3d/
  grid.py         griglia di cerchi e origine (angolo della "L")
  calibration.py  DLT normalizzata, distorsione radiale, modello congiunto
  laser.py        baricentro della lama per riga
  lightplane.py   piano della lama e triangolazione
  measure.py      livellamento del profilo, altezze e angoli
  pipeline.py     dalle immagini grezze alle misure
tests/            test unitari sintetici e una regressione sui dati reali
notebooks/        tre notebook con output salvati
images/           immagini grezze: target, lama, oggetto
docs/             traccia, disegni dei solidi, figure
```

## Setup

Richiede [uv](https://docs.astral.sh/uv/) e Python 3.14 o successivo (sviluppato su 3.15).

```bash
uv sync
uv run pytest        # circa 45 s: include la regressione sui dati reali
```

I notebook si aprono con il kernel del progetto (`ipykernel` è nelle dipendenze di sviluppo), ad esempio da VS Code. JupyterLab non è incluso: `pyyaml` non ha ancora wheel per Python 3.15.

## Dati e disegni

Le immagini sono state acquisite in laboratorio. La traccia e i disegni dei solidi sono materiale del corso (S. Dello Iacono, D. Solinas) e restano di proprietà dei rispettivi autori.
