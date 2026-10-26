import numpy as np

from lightsheet3d.measure import level_to_plate, measure_peaks

# Vertici (y mm, altezza mm): piano a 0, gradino a 15, due picchi con fianchi noti.
ANGLES = {"left1": 60.0, "right1": 75.0, "left2": 65.0, "right2": 55.0}


def run(rise, angle):
    return rise / np.tan(np.radians(angle))


def synthetic_profile(step=0.2, noise=0.0, tilt=0.0, seed=0):
    y0 = -45.0
    v = [(-100, 0), (-55, 0), (-55, 15), (y0, 15)]
    y1 = y0 + run(30, ANGLES["left1"]); v.append((y1, 45))
    y2 = y1 + run(30, ANGLES["right1"]); v.append((y2, 15))
    y3 = y2 + 8; v.append((y3, 15))
    y4 = y3 + run(20, ANGLES["left2"]); v.append((y4, 35))
    y5 = y4 + run(20, ANGLES["right2"]); v.append((y5, 15))
    v += [(y5 + 6, 15), (y5 + 6, 0), (y5 + 60, 0)]
    vy, vh = np.array(v, float).T
    y = np.arange(vy[0], vy[-1], step)
    h = np.interp(y, vy, vh) + tilt * y
    h += np.random.default_rng(seed).normal(0, noise, h.shape)
    return y, h, (y1, y4)


def test_peak_heights_and_positions():
    y, h, (y1, y4) = synthetic_profile(noise=0.3)

    peaks = measure_peaks(y, h)

    # l'apice cade fra due campioni: errore fino a ~0.5 mm sui fianchi ripidi
    assert np.allclose([p.height for p in peaks], [45, 35], atol=0.8)
    assert abs(peaks[0].y - y1) < 1.0 and abs(peaks[1].y - y4) < 1.0


def test_flank_angles_are_within_two_degrees():
    y, h, _ = synthetic_profile(noise=0.3)

    p1, p2 = measure_peaks(y, h)

    assert np.allclose([p1.left_angle, p1.right_angle, p2.left_angle, p2.right_angle],
                       [ANGLES["left1"], ANGLES["right1"], ANGLES["left2"], ANGLES["right2"]], atol=2.0)


def test_level_to_plate_removes_a_tilted_base():
    y, h, _ = synthetic_profile(noise=0.2, tilt=0.03)  # base inclinata di ~3 mm su 100 mm

    leveled = level_to_plate(y, h)

    plate = leveled[(y < -60) | (y > 20)]
    assert abs(np.median(plate)) < 0.3
    assert plate.std() < 0.5


def test_no_peaks_in_a_flat_profile():
    y = np.linspace(-50, 50, 500)

    assert measure_peaks(y, np.zeros_like(y)) == []


def test_jittery_vertical_edge_is_a_single_peak():
    # bordo quasi verticale: molti punti nello stesso decimo di mm con altezze sparse
    rng = np.random.default_rng(3)
    y = np.arange(-20, 20, 0.2)
    h = np.zeros_like(y)
    h[(y > -10) & (y < 0)] = (y[(y > -10) & (y < 0)] + 10) * 4.0  # salita fino a 40
    h[(y >= 0) & (y < 3)] = 40.0
    edge_y = rng.uniform(3.0, 3.4, 60)
    edge_h = rng.uniform(2, 38, 60)
    y, h = np.concatenate([y, edge_y]), np.concatenate([h, edge_h])

    assert len(measure_peaks(y, h, prominence=6)) == 1
