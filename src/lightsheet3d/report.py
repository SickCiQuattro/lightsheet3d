"""Comparison between the measured profile and the nominal one of the solid."""
import matplotlib.pyplot as plt
import numpy as np

from .solids import align, distances, nominal_curve


def compare_profile(ax, solid, y, h, peaks):
    """Plot the measured and nominal profile on `ax`; returns the comparison statistics."""
    sign, shift = align(solid, y, h)
    yn, hn = nominal_curve(solid, sign, shift)
    on_solid = (y > yn[1]) & (y < yn[-2])
    d = distances(solid, sign, shift, y[on_solid], h[on_solid])

    ax.plot(yn, hn, "k--", lw=1.2, label="nominal profile (drawing)")
    ax.plot(y, h, "r.", ms=2, label="measured points")
    nominal_peaks = sorted((sign * x + shift, height) for x, height in solid.peaks())
    for p, (_, nominal) in zip(sorted(peaks, key=lambda p: p.y), nominal_peaks):
        ax.annotate(f"{p.height:.1f} mm\n(nominal {nominal:.1f}, {p.height - nominal:+.1f})",
                    (p.y, p.height), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set(xlabel="Y (mm)", ylabel="height above the support plane (mm)", title=solid.name, ylim=(-5, 62))
    ax.legend(loc="upper right")
    return {"sign": sign, "shift": shift, "median": float(np.median(d)), "points": int(on_solid.sum()),
            "rms": float(np.sqrt((d**2).mean())), "p90": float(np.percentile(d, 90))}


def scan_3d(ax, solid, X, h, sign, shift, vmax=2.0):
    """Measured 3D curve above the nominal solid, extruded along x; returns the distances (mm) of each point.

    `X` (n, 3) are the scan points and `h` the levelled height of each. Each point is
    coloured by its distance from the nominal profile, as in the PLY export.
    """
    from .export import distance_colors

    y = X[:, 1]
    far = np.ptp(y) + 2 * solid.width
    dist = distances(solid, sign, shift, y, h, pad=far)
    yn, hn = nominal_curve(solid, sign, shift)
    grid_y = np.linspace(yn[1], yn[-2], 300)
    half = solid.depth / 2
    centre = (X[:, 0].min() + X[:, 0].max()) / 2
    grid_x = np.linspace(centre - half, centre + half, 2)
    surface = np.interp(grid_y, yn[1:-1], hn[1:-1])
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = np.tile(surface[:, None], (1, 2))
    ax.plot_surface(xx, yy, zz, facecolors=plt.cm.Blues(0.25 + 0.5 * zz / 50), alpha=0.35, linewidth=0, shade=False)
    ax.plot(np.full_like(grid_y, grid_x[0]), grid_y, surface, color="0.4", lw=0.8)  # front edge, to read the shape
    on_solid = (y > yn[1] - 2) & (y < yn[-2] + 2)
    ax.scatter(X[on_solid, 0], y[on_solid], h[on_solid], c=distance_colors(dist[on_solid], vmax) / 255, s=7, depthshade=False)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="height (mm)", ylim=(yn[1] - 5, yn[-2] + 5), zlim=(0, 55))
    ax.set_box_aspect((solid.depth / 2, yn[-2] - yn[1] + 10, 55 * 1.4))
    return dist
