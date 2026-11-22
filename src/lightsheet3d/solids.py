"""Profili nominali dei solidi di taratura, ricavati dalle quote dei disegni del corso.

Ogni profilo parte dallo spigolo in basso a sinistra (x = 0, h = 0) e procede verso destra;
x lungo il solido e h sopra il piano di appoggio sono in mm. I fianchi sono dati dalla loro
inclinazione rispetto all'orizzontale (180° meno l'angolo esterno quotato nel disegno).
"""
from dataclasses import dataclass
from math import radians, tan

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

PAD = 10.0  # mm di piano di appoggio (h = 0) da ciascun lato, per riconoscere anche i picchi di bordo


def _rise(h, deg):
    """Spostamento orizzontale di un fianco di inclinazione `deg` che sale di `h` mm."""
    return h / tan(radians(deg))


def _walk(steps):
    pts = np.cumsum([(0.0, 0.0), *steps], axis=0)
    return pts


@dataclass(frozen=True)
class Solid:
    name: str
    outline: np.ndarray  # (n, 2) x, h dei vertici, da sinistra a destra

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


# Corpo 6 (SDV3D_Corpo6): 100 mm, gradini laterali di 10 e 5 mm, picchi 45 / 25 / 45 mm.
CORPO6 = Solid("Corpo 6", _walk([
    (0, 10), (5, 0), (0, 5), (5, 0),                    # gradino sinistro, fondo a h = 15
    (_rise(30, 60), 30), (_rise(30, 75), -30), (10, 0),
    (_rise(10.05, 70), 10.05), (_rise(10.05, 65), -10.05), (10, 0),
    (_rise(30, 80), 30), (_rise(30, 55), -30),
    (5, 0), (0, -10), (5, 0), (0, -5),                  # gradino destro, fino al piano
]))

# Corpo 5 (SDV3D_Corpo5): 75 mm, tre piani di 8 mm a 35 / 30 / 45 mm.
CORPO5 = Solid("Corpo 5", _walk([
    (0, 35), (8, 0), (6.85, -6.85 * tan(radians(75))), (11.87, 11.87 * tan(radians(60))), (8, 0),
    (11.38, -11.38 * tan(radians(65))), (6.95, 6.95 * tan(radians(80))), (8, 0),
    (13.95, -13.95 * tan(radians(70))),
]))
CORPO5 = Solid(CORPO5.name, np.vstack([CORPO5.outline, [CORPO5.outline[-1, 0], 0.0]]))

SOLIDS = {"corpo5": CORPO5, "corpo6": CORPO6}


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


def nominal_curve(solid, sign, shift, pad=PAD):
    """(y, h) del profilo nominale nel riferimento della scansione, con ± `pad` mm di piano."""
    x, h = solid.outline[::sign].T  # con verso invertito i vertici si percorrono da destra
    y = sign * x + shift
    return np.concatenate([[y[0] - pad], y, [y[-1] + pad]]), np.concatenate([[0.0], h, [0.0]])


def distances(solid, sign, shift, y, h, step=0.05, pad=PAD):
    """Distanza (mm) di ogni punto misurato dal profilo nominale allineato, misurata nel piano
    (y, h): a differenza dello scarto in altezza non amplifica gli errori sui fianchi ripidi."""
    yn, hn = nominal_curve(solid, sign, shift, pad)
    dense = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step)))
                       for a, b in zip(np.column_stack([yn, hn])[:-1], np.column_stack([yn, hn])[1:])])
    return cKDTree(dense).query(np.column_stack([y, h]))[0]
