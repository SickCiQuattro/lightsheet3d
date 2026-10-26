# Light Sheet 3D Scanner

Progetto finale del corso **Sistemi di Visione 3D** (Università degli Studi di Brescia, A.A. 2024/25).

Sistema di misura 3D basato su telecamera e lama di luce: calibrazione della camera, calibrazione del piano di luce e ricostruzione del profilo di solidi 3D per triangolazione. La traccia è in [docs/traccia_progetto.pdf](docs/traccia_progetto.pdf).

## Pipeline

I notebook vanno eseguiti in ordine.

| Notebook | Contenuto |
|---|---|
| [01_sdv3d.ipynb](01_sdv3d.ipynb) | Ricerca del pattern, rimozione della distorsione, crop ottimale e ritaglio di tutte le immagini |
| [02_sdv3d.ipynb](02_sdv3d.ipynb) | Calibrazione della camera: matrice di proiezione `P`, verifica con dataset di validazione, stima dei parametri |
| [03_sdv3d.ipynb](03_sdv3d.ipynb) | Calibrazione della lama di luce: equazione del piano `z = Ax + By + C`, verifica su sotto-insieme di validazione |
| [04_sdv3d.ipynb](04_sdv3d.ipynb) | Acquisizione degli oggetti: baricentro della lama, punti oggetto, profili, nuvola di punti (`.ply`) |

## Struttura

```
images/
  target/     master di taratura a diverse z (30_main.jpg = immagine di riferimento)
  lama/       immagini per la calibrazione della lama di luce
  oggetto/    immagini dei solidi con la lama proiettata
  profilo/    profili estratti e disegni di riferimento dei solidi
outputs/      parametri di calibrazione (.dump) e nuvola di punti (.ply)
docs/         traccia del progetto
```

Le cartelle `images/*/cropped` vengono popolate dal notebook 01.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```
