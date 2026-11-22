"""Confronto fra il profilo misurato e quello nominale del solido."""
import numpy as np

from .solids import align, distances, nominal_curve


def compare_profile(ax, solid, y, h, peaks):
    """Disegna profilo misurato e nominale su `ax`; restituisce le statistiche del confronto."""
    sign, shift = align(solid, y, h)
    yn, hn = nominal_curve(solid, sign, shift)
    on_solid = (y > yn[1]) & (y < yn[-2])
    d = distances(solid, sign, shift, y[on_solid], h[on_solid])

    ax.plot(yn, hn, "k--", lw=1.2, label="profilo nominale (disegno)")
    ax.plot(y, h, "r.", ms=2, label="punti misurati")
    nominal_peaks = sorted((sign * x + shift, height) for x, height in solid.peaks())
    for p, (_, nominal) in zip(sorted(peaks, key=lambda p: p.y), nominal_peaks):
        ax.annotate(f"{p.height:.1f} mm\n(nominale {nominal:.1f}, {p.height - nominal:+.1f})",
                    (p.y, p.height), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set(xlabel="Y (mm)", ylabel="altezza dal piano di appoggio (mm)", title=solid.name, ylim=(-5, 62))
    ax.legend(loc="upper right")
    return {"sign": sign, "shift": shift, "median": float(np.median(d)), "points": int(on_solid.sum()),
            "rms": float(np.sqrt((d**2).mean())), "p90": float(np.percentile(d, 90))}
