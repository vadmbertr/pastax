"""pastax logo — the wordmark "pastax" traced as ensembles of perturbed curves.

Each letter is defined as one or more strokes (ordered (x, y) waypoints in a
shared coordinate frame where the baseline is y=0 and the x-height is ~100).
Every stroke is smoothed with a parametric cubic spline and then drawn as a
small ensemble of sibling curves carrying smooth low-frequency perturbations
(as for the favicon), so each stroke looks hand-traced rather than perfect.

As in the favicon, the three palette colors are mixed within every glyph: each
stroke is drawn as three slightly separated color bands (BLUE / GREEN / PURPLE)
that overlap and blend along the letter.
"""
from random import choice, shuffle, seed as set_pyseed

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# --- JAX palette (left-to-right in the JAX wordmark) ---------------------
BLUE   = "#5e97f6"
GREEN  = "#02796b"
PURPLE = "#9c27b0"
COLORS = [BLUE, GREEN, PURPLE]

# --- Letter definitions --------------------------------------------------
# Each letter is a list of strokes; each stroke is a list of (x, y) waypoints
# in the letter's own local frame (baseline y=0, x-height ~100). Letters are
# advanced left-to-right by ADVANCE[letter] + LETTER_GAP when laid out.

LETTERS = {
    "p": [
        [(12, -40), (12, 40), (12, 95),                                   # stem (with descender)
         (46, 99), (78, 75), (78, 35), (46, 10), (12, 32)],               # bowl
    ],
    "a": [
        [(95, 15), (66, 14), (66, 50), (66, 95),                          # right stem + foot
         (32, 99), (0, 75), (0, 35), (32, 10), (66, 32)],                 # bowl
    ],
    "s": [
        [(81, 75), (47, 99), (20, 87), (20, 65),
         (76, 45), (76, 22), (47, 10), (0, 34)],                          # single S stroke
    ],
    "t": [
        [(5, 85), (50, 85)],                                              # crossbar
        [(30, 140), (5, 60), (0, 25), (5, 10), (30, 15), (50, 35)],       # stem + foot
    ],
    "x": [
        [(0, 97), (25, 94), (45, 75), (45, 35), (25, 15), (0, 12)],       # left arc
        [(95, 97), (70, 94), (50, 75), (50, 35), (70, 15), (95, 12)],     # right arc
    ],
}

# The word, with the per-letter color grouping ("pa" / "st" / "ax").
WORD = [
    ("p", BLUE),
    ("a", BLUE),
    ("s", GREEN),
    ("t", GREEN),
    ("a", PURPLE),
    ("x", PURPLE),
]
GAP = [
    ("p", 0),
    ("a", -15),
    ("s", 0),
    ("t", -5),
    ("a", -10),
    ("x", 0)
]   # extra space (px) between consecutive letters

# --- Ensemble / perturbation options -------------------------------------
N_PER_STROKE = 18     # number of sibling curves drawn per stroke *per color*
NOISE_AMP    = 3.5    # smooth-noise translation amplitude (px) per sibling
SPREAD       = 5      # separation (px) between the three color bands
N_KNOTS      = 2      # spline knots for the translation noise (low = smooth)
CLIP_HEAD    = 10     # 1/CLIP_HEAD of the stroke is the start-clip window
CLIP_TAIL    = 10     # 1/CLIP_TAIL of the stroke is the end-clip window
SAMPLES      = 50     # spline samples per stroke
SEED         = 0

# --- Display options -----------------------------------------------------
SAVE_PATH = "logo.png"
LINEWIDTH = 1
ALPHA     = 0.5
SHOW_WAYPOINTS = False


def smooth(points, n=SAMPLES):
    """Parametric cubic-spline interpolation through 2D waypoints."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 2:                       # straight segment: sample linearly
        tt = np.linspace(0, 1, n)
        return (pts[0, 0] + (pts[1, 0] - pts[0, 0]) * tt,
                pts[0, 1] + (pts[1, 1] - pts[0, 1]) * tt)
    t = np.arange(len(pts))
    cs_x = CubicSpline(t, pts[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, pts[:, 1], bc_type="natural")
    tt = np.linspace(0, len(pts) - 1, n)
    return cs_x(tt), cs_y(tt)


def perturb(x, y, noise_amp=NOISE_AMP, n_knots=N_KNOTS, seed=0):
    """Translate the whole strand by a smooth, low-frequency 2D displacement.

    Every point is shifted by a slowly varying (dx, dy) vector drawn from
    cubic-spline smooth noise. Because the perturbation is a plain translation
    of the strand — no per-point tangent or normal is computed — the result
    stays smooth even for very short or straight segments (e.g. the t crossbar)
    and at the cusps where a stroke reverses direction (e.g. the p stem/bowl).
    """
    n = len(x)
    s = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    knots_t = np.linspace(0, 1, n_knots)
    dx = CubicSpline(knots_t, rng.standard_normal(n_knots), bc_type="natural")(s)
    dy = CubicSpline(knots_t, rng.standard_normal(n_knots), bc_type="natural")(s)
    return x + dx * noise_amp, y + dy * noise_amp


def plot():
    set_pyseed(SEED)

    # Lay out the word: build every stroke in absolute coordinates with color.
    strokes = []   # (x_ref, y_ref, color)
    for letter, color in WORD:
        letter_strokes = []
        for stroke in LETTERS[letter]:
            pts = [(px, py) for px, py in stroke]
            x_ref, y_ref = smooth(pts)
            letter_strokes.append((x_ref, y_ref, color))
        strokes.append(letter_strokes)

    # Per-color base offset: three small translations 120 degrees apart, so the
    # blue / green / purple bands are distinguishable yet overlap and blend.
    angles = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    color_off = list(zip(SPREAD * np.cos(angles), SPREAD * np.sin(angles)))

    # Expand each stroke into an ensemble of perturbed siblings, in all three
    # colors (the colors are mixed within every glyph, as in the favicon).
    strands = []
    offset = 0
    for letter_strokes, (_, gap) in zip(strokes, GAP):
        x_min = 0
        x_max = 0
        letter_strands = []
        for si, (x_ref, y_ref, _) in enumerate(letter_strokes):
            for ci, color in enumerate(COLORS):
                ox, oy = color_off[ci]
                for i in range(N_PER_STROKE):
                    x, y = perturb(x_ref, y_ref, seed=(si * 3 + ci) * 1000 + i)
                    x = x + ox
                    y = y + oy
                    head = max(1, len(x) // CLIP_HEAD)
                    tail = max(1, len(x) // CLIP_TAIL)
                    start = choice(range(0, head))
                    end = choice(range(len(x) - tail, len(x)))
                    x = x[start:end]
                    y = y[start:end]
                    x_min = min(x_min, float(np.min(x)))
                    x_max = max(x_max, float(np.max(x)))
                    x += offset
                    letter_strands.append((x, y, color))
        for x, y, color in letter_strands:
            strands.append((x + np.abs(x_min), y, color))
        offset += ((x_max - x_min) + gap)

    all_x = np.concatenate([s[0] for s in strands])
    all_y = np.concatenate([s[1] for s in strands])
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    pad = 12
    min_x, max_x = min_x - pad, max_x + pad
    min_y, max_y = min_y - pad, max_y + pad

    shuffle(strands)   # interleave colors in draw order so the bands blend
    width = max_x - min_x
    height = max_y - min_y
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    for x, y, color in strands:
        ax.plot(x, y, color=color, lw=LINEWIDTH, alpha=ALPHA,
                solid_capstyle="round", solid_joinstyle="round")

    if SHOW_WAYPOINTS:
        for x_ref, y_ref, color in strokes:
            ax.plot(x_ref, y_ref, color="k", lw=0.5, alpha=0.8, zorder=10)

    ax.set_aspect("equal")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)   # y-up (baseline at bottom)
    ax.axis("off")
    fig.tight_layout(pad=0)

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=200, transparent=True, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    plot()
