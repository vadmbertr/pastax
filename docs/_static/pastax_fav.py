"""pastax favicon — three ensemble trajectories tracing a partial-derivative glyph.

Define one reference path (REF_PTS below). The script generates two sibling
trajectories by offsetting it perpendicular to its tangent ("left of travel"
and "right of travel"), with small smooth perturbations so the three strands
look like an ensemble rather than perfect parallels. All three share the
initial condition exactly.
"""
from random import choice, shuffle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# --- JAX palette (left-to-right in the JAX wordmark) ---------------------
BLUE   = "#5e97f6"
GREEN  = "#02796b"
PURPLE = "#9c27b0"

# --- Shared initial condition (partial tail tip) -------------------------
IC = (40, 90)

# --- Reference trajectory ------------------------------------------------
# Ordered (x, y) waypoints. The spline passes through every waypoint.
# Repeat the bowl-entry waypoint as the final waypoint to close the bowl.
REF_PTS = [
    IC,            # tail tip (shared IC)
    (60, 50),      # tail going up
    (125, 35),     # arc apex
    (190, 70),     # tail descending
    (210, 130),    # bowl entry (upper right)
    (205, 200),    # bowl right
    (155, 255),    # bowl bottom-right
    (80, 250),     # bowl bottom-left
    (35, 200),     # bowl left
    (60, 145),     # bowl top-left
    (140, 125),    # bowl top
    (200, 150),    # close right 2nd
    (190, 210),    # close bowl
]

# --- Sibling-strand generation ------------------------------------------
OFFSET       = 10    # perpendicular shift (px) for the side strands
RAMP         = 0.001   # fraction of curve over which offset ramps up from IC
NOISE_AMP    = 5     # smooth-noise amplitude (px) per side strand
N_PER_COLOR  = 50   # number of trajectories drawn per color
SEED_REF     = 0
SEED_LEFT    = 1
SEED_RIGHT   = 2

# Colors assigned to (left-of-travel, reference, right-of-travel)
LEFT_COLOR  = BLUE
REF_COLOR   = GREEN
RIGHT_COLOR = PURPLE

# --- Display options -----------------------------------------------------
SHOW_WAYPOINTS = False 
SAVE_PATH = "fav.png"
LINEWIDTH = 1


def smooth(points, n=400):
    """Parametric cubic-spline interpolation through 2D waypoints."""
    pts = np.asarray(points, dtype=float)
    t = np.arange(len(pts))
    cs_x = CubicSpline(t, pts[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, pts[:, 1], bc_type="natural")
    tt = np.linspace(0, len(pts) - 1, n)
    return cs_x(tt), cs_y(tt)


def offset_curve(x, y, dist, ramp=RAMP, noise_amp=NOISE_AMP, seed=0):
    """Offset (x, y) perpendicular to its tangent by `dist`.

    `dist > 0` shifts to the right of the direction of travel; `< 0` shifts
    left. The shift tapers from 0 over the first `ramp` fraction of the path
    so all sibling strands share the IC and starting tangent exactly. Adds
    smooth low-frequency noise (amplitude `noise_amp`) for ensemble flavor.
    """
    n = len(x)
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy) + 1e-12
    nx, ny = dy / norm, -dx / norm   # right-hand normal

    s = np.linspace(0, 1, n)
    e = np.clip(s / ramp, 0, 1)
    envelope = 0.5 - 0.5 * np.cos(np.pi * e)

    rng = np.random.default_rng(seed)
    n_knots = max(6, n // 30)
    knots_t = np.linspace(0, 1, n_knots)
    knots_v = rng.standard_normal(n_knots)
    noise = CubicSpline(knots_t, knots_v, bc_type="natural")(s) * noise_amp

    d = (dist + noise) * envelope
    return x + d * nx, y + d * ny


def plot():
    x_ref, y_ref = smooth(REF_PTS)

    # One group per color: same base offset, distinct noise seeds so the
    # N strands of a color start together (offset/noise ramp from 0 at IC)
    # and then diverge along the path.
    groups = [
        (-OFFSET, LEFT_COLOR,  SEED_LEFT),
        (0,       REF_COLOR,   SEED_REF),
        (+OFFSET, RIGHT_COLOR, SEED_RIGHT),
    ]

    strands = []
    for dist, color, base_seed in groups:
        for i in range(N_PER_COLOR):
            x, y = offset_curve(x_ref, y_ref, dist, seed=base_seed * 1000 + i)
            strands.append((x, y, color))

    all_x = np.concatenate([s[0] for s in strands])
    all_y = np.concatenate([s[1] for s in strands])
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    dx = max_x - min_x
    dy = max_y - min_y
    dd = max(dx, dy) + LINEWIDTH
    min_x, max_x = (min_x + max_x) / 2 - dd / 2, (min_x + max_x) / 2 + dd / 2
    min_y, max_y = (min_y + max_y) / 2 - dd / 2, (min_y + max_y) / 2 + dd / 2

    shuffle(strands)
    fig, ax = plt.subplots(figsize=(4, 5.33))
    for x, y, color in strands:
        start = choice(range(0, len(x) // 3))
        end = choice(range(2 * len(x) // 3, len(x)))
        ax.plot(x[start:end], y[start:end], color=color, lw=LINEWIDTH, alpha=0.7,
                solid_capstyle="round", solid_joinstyle="round")
        
    if SHOW_WAYPOINTS:
        px, py = zip(*REF_PTS)
        ax.plot(px, py, "o", color=REF_COLOR, alpha=1, ms=5, zorder=10)

    ax.set_aspect("equal")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.axis("off")
    fig.tight_layout(pad=0)

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=100, transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot()
