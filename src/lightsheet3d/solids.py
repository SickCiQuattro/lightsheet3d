"""Profili nominali dei solidi di taratura, ricavati dai disegni tecnici del corso.

I vertici stanno in `profiles.json`, generato da `scripts/extract_drawing_profiles.py` leggendo il profilo
frontale vettoriale dei PDF (archi compresi). Ogni profilo parte dallo spigolo in basso a sinistra
(x = 0, h = 0) e procede verso destra; x lungo il solido e h sopra il piano di appoggio sono in mm.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

PAD = 10.0  # mm di piano di appoggio (h = 0) da ciascun lato, per riconoscere anche i picchi di bordo


@dataclass(frozen=True)
class Solid:
    name: str
    outline: np.ndarray  # (n, 2) x, h dei vertici, da sinistra a destra
    depth: float = 50.0  # mm, estrusione del solido lungo x (50 in tutti i disegni)

    @property
    def width(self):
        return float(self.outline[-1, 0] - self.outline[0, 0])

    def peaks(self):
        """(x, altezza) di ogni picco nominale; su un piano piano è il centro del tratto piatto."""
        x = np.arange(-PAD, self.width + PAD, 0.01)
        h = np.interp(x, *self.outline.T, left=0.0, right=0.0)
        idx, info = find_peaks(h, prominence=5, plateau_size=1)
        centre = (x[info["left_edges"]] + x[info["right_edges"]]) / 2
        return [(float(c), float(h[i])) for c, i in zip(centre, idx)]


def _load():
    data = json.loads((Path(__file__).parent / "profiles.json").read_text())
    return {key: Solid(v["name"], np.array(v["outline"]), v["depth"]) for key, v in data.items()}


SOLIDS = _load()
CORPO1, CORPO2, CORPO3, CORPO4, CORPO5, CORPO6 = (SOLIDS[f"corpo{n}"] for n in range(1, 7))


def align(solid, y, h, clip=5.0, step=0.1, candidates=4, refine=1.0):
    """(segno, traslazione) che portano il profilo nominale sul profilo misurato (y, h).

    Il riferimento Y della scansione è arbitrario e può avere verso opposto: si prova entrambi.
    Lo scarto è limitato a `clip` mm, così i tratti non misurabili (fianchi troppo ripidi) non
    spostano il risultato. Due passi, senza ottimizzatore: lo scarto verticale su griglia di `step` mm
    sceglie i `candidates` migliori per verso; fra questi decide la distanza nel piano (y, h), che con
    rumore alto distingue il verso giusto da quello speculare quando lo scarto verticale è quasi pari.
    """
    y, h = np.asarray(y), np.asarray(h)
    shifts = np.arange(y.min() - solid.width, y.max() + solid.width, step)
    pool = []
    for sign in (1, -1):
        cost = np.empty(len(shifts))
        for i, shift in enumerate(shifts):
            yn, hn = nominal_curve(solid, sign, shift)
            cost[i] = np.minimum(np.abs(h - np.interp(y, yn, hn, left=0.0, right=0.0)), clip).mean()
        chosen = []
        for i in np.argsort(cost):
            if all(abs(shifts[i] - shifts[j]) >= 5 for j in chosen):
                chosen.append(i)
            if len(chosen) == candidates:
                break
        pool += [(sign, shifts[i]) for i in chosen]

    far = y.max() - y.min() + 2 * solid.width  # piano esteso oltre ogni punto, così il costo non dipende dal numero di punti
    best = min(((np.minimum(distances(solid, sign, shift + d, y, h, pad=far), clip).mean(), sign, float(shift + d))
                for sign, shift in pool for d in np.arange(-refine, refine + step / 2, step)))
    return best[1], best[2]


def align_oblique(solid, y, h, clip=5.0, max_scale=1.6):
    """(segno, traslazione, scala): come `align`, con in più l'allungamento lungo Y di un taglio obliquo.

    Verso, scala (da 1 a `max_scale`) e traslazione si cercano insieme, perché con un taglio molto obliquo
    il profilo di un solido quasi simmetrico può sembrare speculare: prima su griglia grossolana con lo
    scarto verticale limitato a `clip` mm, poi su griglia fine attorno al minimo, infine con la distanza nel
    piano (y, h). La scala è sempre >= 1; l'angolo fra la lama e la normale all'asse del solido è arccos(1 / scala).
    """
    y, h = np.asarray(y), np.asarray(h)

    def vertical_cost(sign, shift, scale):
        yn, hn = nominal_curve(solid, sign, shift, scale=scale)
        return np.minimum(np.abs(h - np.interp(y, yn, hn, left=0.0, right=0.0)), clip).mean()

    def search(signs, scales, shifts):
        return min((vertical_cost(sign, shift, scale), sign, scale, shift) for sign in signs for scale in scales for shift in shifts)

    reach = solid.width * max_scale
    _, sign, scale, shift = search((1, -1), np.arange(1.0, max_scale + 1e-9, 0.05), np.arange(y.min() - reach, y.max() + reach, 0.5))
    _, sign, scale, shift = search((sign,), np.arange(max(1.0, scale - 0.05), scale + 0.051, 0.01), np.arange(shift - 1.5, shift + 1.51, 0.1))
    far = (np.ptp(y) + 2 * solid.width) * max_scale
    best = min((np.minimum(distances(solid, sign, shift + d, y, h, pad=far, scale=max(1.0, scale + e)), clip).mean(),
                max(1.0, scale + e), shift + d) for d in np.arange(-0.5, 0.51, 0.1) for e in (-0.01, 0.0, 0.01))
    return sign, float(best[2]), float(best[1])


def nominal_curve(solid, sign, shift, pad=PAD, scale=1.0):
    """(y, h) del profilo nominale nel riferimento della scansione, con ± `pad` mm di piano.

    `scale` allunga il profilo lungo Y: un solido posato di traverso rispetto alla lama è tagliato in
    modo obliquo e il suo profilo risulta allungato di 1/cos(angolo) (vedi `align_oblique`).
    """
    x, h = solid.outline[::sign].T  # con verso invertito i vertici si percorrono da destra
    y = sign * scale * x + shift
    return np.concatenate([[y[0] - pad], y, [y[-1] + pad]]), np.concatenate([[0.0], h, [0.0]])


def distances(solid, sign, shift, y, h, step=0.05, pad=PAD, scale=1.0):
    """Distanza (mm) di ogni punto misurato dal profilo nominale allineato, misurata nel piano
    (y, h): a differenza dello scarto in altezza non amplifica gli errori sui fianchi ripidi."""
    yn, hn = nominal_curve(solid, sign, shift, pad, scale)
    dense = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step)))
                       for a, b in zip(np.column_stack([yn, hn])[:-1], np.column_stack([yn, hn])[1:])])
    return cKDTree(dense).query(np.column_stack([y, h]))[0]
