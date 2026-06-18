# pastax

**P**arameterizable **A**uto-differentiable **S**imulators for ocean surface **T**rajectories in j**AX**.

`pastax` integrates particle trajectories on the ocean surface by solving ODEs and SDEs over user-supplied forcing fields (e.g. surface currents). Every computation is fully differentiable via JAX automatic differentiation — both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) are supported (forward-mode requires `solve(..., adjoint="forward")`; the default checkpointed adjoint is reverse-mode only).

📖 **Documentation:** <https://vadmbertr.github.io/pastax/> — full API reference and a runnable [tutorial notebook](docs/tutorial.ipynb).

## Project Status

- Bilinear interpolation of rectilinear forcing fields, with neighbourhood cube extraction
- A-grid and NEMO-convention Arakawa C-grid forcing layouts (`Dataset.from_arrays_cgrid` / `from_xarray_cgrid`)
- Coastal robustness on A-grid: NaN-inferred land masks, inverse-distance partial-cell bilinear, and an opt-in partial-slip scheme via `Dataset.velocity_interp`
- Unified `solve` function — ODE / ODE-with-controls / SDE mode selected by caller
- Unified `term` API for specifying the dynamics. The state `y` may be any PyTree (a bare array is the single-leaf case). ODE terms have signature `term(t, y[, args, ctrl]) -> dy` (same structure as `y`). SDE terms have signature `term(t, y[, args, ctrl]) -> (drift, diffusion)`, where `diffusion` is a diagonal PyTree, a matrix, or a `lineax` operator. `args` is a Pytree fixed for the entire integration; `ctrl` is a per-step time-varying Pytree, useful when the term requires external time-varying inputs. PyTree states make second-order dynamics (`dv = f dt + noise`, `dx = v dt`) natural
- ODE solvers: Euler, Heun, RK4, Tsit5, Dopri5. SDE solvers: Euler-Maruyama, Stratonovich Heun, Stratonovich RK4 via the ODE classes, plus dedicated EulerHeun, ItoMilstein, and StratonovichMilstein; SDE solvers draw `z ~ N(0, I_2)` and applies `dW = sqrt(int_dt) * z` internally
- Forward or backwards-in-time integration (pass negative `int_dt` / `save_dt`)
- Geographic unit conversions (metres ↔ degrees)
- Along-trajectory metrics with optional ensemble (vmap) mode
- Proper scoring rules to evaluate and train stochastic simulators
- xarray (zarr/netCDF) dataset loading

## Installation

From Git:

```bash
pip install git+https://github.com/vadmbertr/pastax             # core (JAX, Equinox, jaxtyping)
pip install "git+https://github.com/vadmbertr/pastax[forcing]"  # + xarray, zarr, netCDF4
```

From source:

```bash
git clone https://github.com/vadmbertr/pastax
cd pastax
pip install -e ".[dev]"
```

Installing a JAX **GPU version** should be done prior to installing `pastax`, following
[https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html).

## Quick Start

### ODE and SDE simulation

The term signature is `term(t, y[, args, ctrl])`.
The optional `args` argument is present when `args` is passed to `solve`.
The optional `ctrl` argument is present when `controls` is passed to `solve`
— the solver slices `controls[i]` at each step and forwards it.
The term owns all interpretation of `args` and `ctrl`.

`t0` is a traced JAX scalar — changing it never triggers recompilation. `n_save`,
`int_dt`, and `save_dt` are static Python scalars; the end time is implicit as
`t0 + n_save * save_dt`. Sub-stepping is expressed by setting `int_dt < save_dt`
with `save_dt / int_dt` an integer.

Without `key` passed to `solve` it is ODE mode and the term returns a velocity;
with `key` it is SDE mode and the term returns `(drift, g)`.

```python
import jax.numpy as jnp
import jax.random as jr
from pastax import solve, Heun, RK4, EulerHeun, meters_to_degrees

# --- ODE term (no key) ---
def ode_term(t, y, args):
    dataset = args
    u = dataset["u"].interp(t, y[0], y[1])
    v = dataset["v"].interp(t, y[0], y[1])
    return meters_to_degrees(jnp.array([v, u]), y[0])   # deg/s

y0 = jnp.array([48.0, -4.0])   # [lat, lon]
t0 = jnp.array(0.0)            # start time, seconds
traj = solve(ode_term, y0, t0, n_save=120, int_dt=3600., save_dt=3600., args=dataset)
# shape (121, 2)

# --- ODE term with per-step controls ---
def perturbed_term(t, y, args, ctrl):
    dataset = args
    z = ctrl
    base_vel = ...
    return base_vel + 1e-4 * jnp.tanh(z)

n_fine = 120  # = n_save * round(save_dt / int_dt)
traj = solve(perturbed_term, y0, t0, n_save=120, int_dt=3600., save_dt=3600.,
             args=dataset, controls=jr.normal(jr.key(0), (n_fine, 2)))

# --- SDE term (pass key) ---
def sde_term(t, y, args):
    dataset = args
    u = dataset["u"].interp(t, y[0], y[1])
    v = dataset["v"].interp(t, y[0], y[1])
    drift = meters_to_degrees(jnp.array([v, u]), y[0])
    g     = jnp.full(2, 1e-5)   # diagonal diffusion, deg / sqrt(s)
    return drift, g             # z ~ N(0, I_2) is drawn internally

traj     = solve(sde_term, y0, t0, 120, 3600., 3600., EulerHeun(), args=dataset, key=jr.key(0))
ensemble = solve(sde_term, y0, t0, 120, 3600., 3600., EulerHeun(), args=dataset, key=jr.key(0),
                 n_samples=100)
# shapes: (121, 2) and (100, 121, 2)
```

In SDE mode the solver draws a standard-normal `z` and applies
`dW = sqrt(int_dt) * z` internally; the term never sees `z`. For a flat state `g`
can be shape `(2,)` (diagonal) or `(2, 2)` (full matrix). The `Euler`, `Heun`,
and `RK4` solvers accept both ODE and SDE mode; `ItoMilstein` /
`StratonovichMilstein` give strong order 1.0 for diagonal `g` via `jax.jacfwd`.

### PyTree state and second-order dynamics

The state `y` can be any PyTree (a bare array is the single-leaf case). The term
returns the time derivative `dy` with the **same structure** as `y`, and the
output trajectory is a PyTree shaped like `y0` with a leading time axis on each
leaf. This makes **second-order** dynamics natural — `dv = f(x, t) dt + noise`,
`dx = v dt` — by carrying `y = (x, v)` and putting the noise on the velocity leaf:

```python
from typing import NamedTuple
import jax.numpy as jnp
import jax.random as jr
from pastax import solve, EulerHeun

class State(NamedTuple):
    x: jnp.ndarray   # position [lat, lon]    (deg)
    v: jnp.ndarray   # velocity [v_lat, v_lon] (deg/s)

def sde_term(t, y):
    accel = -(y.v - u_current(t, y.x)) / tau          # your f(x, t[, v]); deg/s^2
    drift = State(x=y.v, v=accel)                      # dx = v, dv = accel
    diff  = State(x=jnp.zeros(2), v=jnp.full(2, 1e-5)) # diagonal noise on velocity only
    return drift, diff

y0   = State(x=jnp.array([48.0, -4.0]), v=jnp.zeros(2))
traj = solve(sde_term, y0, jnp.array(0.0), 120, 3600., 3600., EulerHeun(), key=jr.key(0))
# traj.x, traj.v each have shape (121, 2)   — underdamped Langevin: dx=v dt, dv=accel dt + g dW
```

For a general (matrix / cross-leaf) diffusion, return a
[`lineax`](https://github.com/patrick-kidger/lineax) `AbstractLinearOperator` as
the diffusion (applied via `.mv(dW)`); pass `brownian_structure=` (a PyTree of
`jax.ShapeDtypeStruct`) when the noise space differs from the state space. This
needs the optional `lineax` dependency (`pip install "pastax[sde]"`); the diagonal
per-leaf form above needs no extra dependency. The Milstein solvers remain flat
array only.

### Loading forcing fields from xarray

```python
import xarray as xr
from pastax import Dataset

ds = xr.open_zarr("path/to/currents.zarr")
dataset = Dataset.from_xarray(
    ds,
    fields={"u": "uo", "v": "vo"},
    coordinates={"time": "time", "lat": "latitude", "lon": "longitude"},
)
```

Or directly from numpy/JAX arrays:

```python
import numpy as np

t   = np.linspace(0.0, 4 * 86400.0, 5)  # seconds
lat = np.linspace(40.0, 50.0, 100)
lon = np.linspace(-10.0, 0.0, 100)
u_data = np.ones((5, 100, 100), dtype=np.float32)

dataset = Dataset.from_arrays({"u": u_data}, t=t, lat=lat, lon=lon)
```

### Loading C-grid forcing (NEMO convention)

For data on an Arakawa C-grid, every vector field has its U component on the
east faces of the centre cells (shape `(time, nlat, nlon - 1)`) and its V
component on the north faces (shape `(time, nlat - 1, nlon)`). Vector fields
are declared via the `vectors` mapping; the outer key is a free-form label
for each pair (used in error messages) and the inner tuples give the field
names under which the components are registered in `Dataset.fields` plus
their values. `from_arrays_cgrid` auto-derives the staggered coordinates as
half-cell shifts of the centre grid and shares them across every registered
vector:

```python
from pastax import Dataset

dataset = Dataset.from_arrays_cgrid(
    t, center_lat, center_lon,
    vectors={
        "current": {"u": ("u", u_values), "v": ("v", v_values)},
    },
    tracers={"sst": sst_values},    # optional, at cell centres (T, nlat, nlon)
)
# dataset["u"].stagger == "u_face"
# dataset["v"].stagger == "v_face"
# dataset.grid.stagger_type == "C"
```

Several vector fields can live on the same C-grid — surface current and
10-m wind, or a decomposition into geostrophic / Ekman / Stokes
components — by adding more entries to `vectors`:

```python
dataset = Dataset.from_arrays_cgrid(
    t, center_lat, center_lon,
    vectors={
        "current": {"u": ("uo",  u_curr), "v": ("vo",  v_curr)},
        "wind":    {"u": ("u10", u_wind), "v": ("v10", v_wind)},
    },
)
# dataset.fields.keys() == {"uo", "vo", "u10", "v10"}
# Pick the pair to integrate via velocity_interp(u_name=..., v_name=...).
```

The same `term` you wrote for A-grid forcing works unchanged — each `Field`
reads its (already-shifted) coordinates from the shared `Grid` by stagger role,
so `Field.interp` applies the correct bilinear-on-shifted-coords sample at the
particle position.

xarray analogue:

```python
dataset = Dataset.from_xarray_cgrid(
    ds,
    vectors={
        "current": {"u": ("u", "uo"), "v": ("v", "vo")},
        # each tuple is (internal_field_name, xarray_variable_name)
    },
    coordinates={"time": "time", "lat": "lat", "lon": "lon"},  # centre coords
    tracers={"sst": "thetao"},
)
```

### Coastal forcing

Real ocean forcing has land. By default the loaders detect land cells
automatically:

```python
# u_data has NaN at every land cell (CMEMS / CF convention)
dataset = Dataset.from_arrays({"u": u_data, "v": v_data}, t=t, lat=lat, lon=lon)
# NaN was replaced with 0 in the stored values; a 2-D bool mask
# was inferred from the NaN locations and attached to each Field:
# dataset["u"].mask.shape == (nlat, nlon)
```

If your data marks land with zeros (NEMO convention) or a custom flag,
pass an explicit mask instead:

```python
land_mask = (raw_bathy == 0)               # True where land
dataset = Dataset.from_arrays(
    {"u": u_data, "v": v_data}, t=t, lat=lat, lon=lon,
    masks={"u": land_mask, "v": land_mask},
)
```

The mask is consumed by `Field.interp` to switch from naive bilinear to
**inverse-distance partial-cell weighting** whenever a cell straddles
the coast: land corners are dropped and the remaining ocean corners are
weighted by `1 / d²` from the query point. Fully land-bound cells
return `0`. This eliminates the "stuck particle" artefact that plagues
naive bilinear interpolation on A-grid coastal data — particles
released near a coast slide along it at the correct ocean velocity
instead of stalling.

For richer wall-physics control, use `Dataset.velocity_interp` to
interpolate `(U, V)` jointly with an opt-in partial-slip correction:

```python
def my_term(t, y, args):
    dataset = args
    vel = dataset.velocity_interp(t, y[0], y[1], scheme="partialslip")
    return meters_to_degrees(vel, y[0])   # vel is [v, u] = [dlat/dt, dlon/dt]
```

`scheme="default"` (the default) composes per-field `Field.interp`
(inverse-distance when a mask is present). `scheme="partialslip"`
applies a tunable wall-slip correction near fully-land edges:
`U` near a latitudinal coast is rescaled by `(slip_a + slip_b * wl)`,
and `V` near a longitudinal coast by `(slip_a + slip_b * wj)`. The
default `slip_a = slip_b = 0.5` gives a half-slip wall; `slip_a = 1,
slip_b = 0` recovers full free-slip. Partial-slip is A-grid only —
calling it on a C-grid dataset raises `NotImplementedError`.

C-grid forcing handles coasts correctly without any mask, **provided
U and V at land-adjacent faces are exactly zero** (the NEMO output
convention): the face-normal velocity then vanishes at the coast by
construction.

All three coastal paths (inverse-distance, partial-slip, naive
bilinear) are gradient-safe under `jax.grad` and `jax.jvp`.

### Neighbourhood extraction

```python
# Extract a 5×5×5 patch of raw grid values around a query point
patch = dataset["u"].neighborhood(t, lat, lon, t_window=2, lat_window=2, lon_window=2)
# shape (5, 5, 5)

# Or for all fields at once
patches = dataset.neighborhood(t, lat, lon, lat_window=1, lon_window=1)
# dict[str, Array] with shape (3, 3, 3) per field
```

### Geographic conversions

```python
from pastax import meters_to_degrees, degrees_to_meters

disp_m = jnp.array([1000.0, 500.0])  # [north, east] metres
lat_ref = jnp.array(45.0)
disp_deg = meters_to_degrees(disp_m, lat_ref)   # [dlat, dlon] degrees
```

### Backwards-in-time integration

Pass negative `int_dt` and `save_dt` to integrate backwards. All solvers handle
this transparently:

```python
y0_end = jnp.array([48.0, -4.0])
backtrack = solve(my_term, y0_end,
                  t0=jnp.array(86400.0 * 5), n_save=120, int_dt=-3600., save_dt=-3600.,
                  solver=RK4(),
                  args=dataset)
# backtrack[-1] is the source position 5 days earlier.
```

### Differentiability

```python
import jax

# Reverse-mode gradient through the ODE solver (default adjoint="checkpointed":
# low-memory binomial checkpointing, O(sqrt(n)) memory by default)
grad = jax.grad(lambda y0: solve(ode_term, y0, t0, n_save, int_dt, save_dt, args=dataset).sum())(y0)

# Forward-mode JVP requires adjoint="forward" (the checkpointed adjoint is reverse-mode only)
traj, tangent = jax.jvp(
    lambda y0: solve(ode_term, y0, t0, n_save, int_dt, save_dt, args=dataset, adjoint="forward"),
    (y0,), (jnp.ones(2),),
)
```

The `solve` `adjoint` argument selects the differentiation strategy: `"checkpointed"`
(default) uses binomial checkpointing for low reverse-mode memory but does not support
`jax.jvp`; `"forward"` uses a plain `jax.lax.scan` (no per-step checkpoint), ideal for
`jax.jvp` / `jax.jacfwd`. The `checkpoints` argument tunes the memory/recompute
tradeoff of the checkpointed adjoint (default `None` → `equinox.internal.scan`'s
O(sqrt(n)) Stumm–Walther schedule; larger → more memory, less recompute; `"all"` →
checkpoint every step).

### Trajectory metrics

```python
from pastax import separation_distance, normalized_separation_distance, liu_index

# Single-trajectory metrics
sep = separation_distance(trajectory, reference)          # (T,), metres
nsd = normalized_separation_distance(trajectory, reference)  # (T,), dimensionless
li  = liu_index(trajectory, reference)                    # (T,), dimensionless

# Ensemble metrics — broadcasting the ensemble leading axis
sep_ens = separation_distance(ensemble, reference)  # (S, T)
li_ens  = liu_index(ensemble, reference)            # (S, T)
```

### Scoring rules

```python
from pastax import dawid_sebastiani, energy_score, squared_error, variogram_score

# Along trajectory scores
ds_ts = dawid_sebastiani(ens, ref, reduce=None)  # (T,)
es_ts = energy_score(ens, ref, reduce=None)  # (T,)
se_ts = squared_error(ens, ref, reduce=None)  # (T,)
vs_ts = variogram_score(ens, ref, reduce=None)  # (T,)

# Final scores
ds_t1 = dawid_sebastiani(ens, ref, reduce="last")  # scalar
es_t1 = energy_score(ens, ref, reduce="last")  # scalar
se_t1 = squared_error(ens, ref, reduce="last")  # scalar
vs_t1 = variogram_score(ens, ref, reduce="last")  # scalar

# Aggregated scores
ds_agg = dawid_sebastiani(ens, ref, reduce="sum")  # scalar
es_agg = energy_score(ens, ref, reduce="sum")  # scalar
se_agg = squared_error(ens, ref, reduce="sum")  # scalar
vs_agg = variogram_score(ens, ref, reduce="sum")  # scalar

# Custom score kernel (relevant for the energy score and the square error only)
es_agg = energy_score(ens, ref, kernel=separation_distance)
se_agg = squared_error(ens, ref, kernel=separation_distance)
```

## API Reference

The full API reference — every public symbol, signature, and docstring — lives on the documentation site: <https://vadmbertr.github.io/pastax/api>.

## Dependencies

- [JAX](https://github.com/google/jax) ≥ 0.4.30
- [Equinox](https://github.com/patrick-kidger/equinox) ≥ 0.11.0
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping) ≥ 0.2.30
- xarray, Zarr, netCDF4 (optional, for forcing loading)

## License

Apache-2.0
