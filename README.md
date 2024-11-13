# pastax

<p align="center">
    <img src="https://raw.githubusercontent.com/vadmbertr/pastax/refs/heads/main/docs/_static/pastax-md.png" alt="pastax logo" width="33%">
</p>

<p align="center">
    <b>P</b>arameterizable <b>A</b>uto-differentiable <b>S</b>imulators of ocean <b>T</b>rajectories in j<b>AX</b>.
</p>

## Installation

`pastax` is Pip-installable:

```shell
pip install pastax
```

## Usage

Documentation is under construction.
Meanwhile, see the [demonstration notebook](docs/demo.ipynb) and the (messy) [API documentation](https://pastax.readthedocs.io/en/latest/api/).

## Work in progress

This package in under active developement and should still be considered as work in progress.

In particular, the following changes are considered:

- `pastax.grid`
  - add support for C-grids,
- `pastax.trajectory`
  - use `unxt.Quantity` in place of `Unitful` (see [unxt](https://unxt.readthedocs.io/en/latest/)),
  - remove `__add__` and like methods in favour of registered functions (see [quax](https://docs.kidger.site/quax/)),
- `pastax.simulator`
  - improve how the product operation is performed between the vector field and the control (support for `Location`, `Time` or `State` objects) (see `diffrax` doc [here](https://docs.kidger.site/diffrax/api/terms/#diffrax.ControlTerm)),
  - add support/examples for interrupting the solve when a trajectory reaches land (see `diffrax` doc [here](https://docs.kidger.site/diffrax/api/events/)).

And I should stress that the package lacks (unit-)tests for now.

## Related projects

Several other open-source projects already exist with close objectives.
The two closest ones are probably [(Ocean)Parcels](https://github.com/OceanParcels/parcels) and [Drifters.jl](https://github.com/JuliaClimate/Drifters.jl).

Here is a (probably) non-comprehensive (and hopefuly correct, please reach-out if not) use-cases comparison between those:

- you use Python: go with `pastax` or `Parcels`,
- you use Julia: go with `Drifters.jl`,
- you want I/O inter operability with `xarray` Datasets: go with `pastax`, `Parcels` or `Drifters.jl`,
- you need support for Arakawa C-grid: go with `Parcels` or `Drifters.jl` (but keep an eye on `pastax` as it might come in the future),
- you want some post-precessing routines: go with `Drifters.jl` (but keep an eye on `pastax` as some might come in the future),
- you want a better control of the right-hand-side term of your Differential Equation: go with `pastax` (probably the most flexible) or `Parcels`,
- you solve Stochastic Differential Equations: go with `pastax` (probably the most flexible) or `Parcels`,
- you need a **wide** range of solvers: go with `pastax` or `Drifters.jl` (if you solve ODE),
- you want to calibrate your simulator ***on-line*** (i.e. by differenting through your simulator): go with `pastax`!

It might be worth mentionning also [py-eddy-tracker](https://github.com/AntSimi/py-eddy-tracker) altough it targets more specifically eddy related routines.

## Contributing

Contributions are welcomed!
See [CONTRIBUTING.md](https://github.com/vadmbertr/pastax/blob/main/CONDUCT.md) and [CONDUCT.md](https://github.com/vadmbertr/pastax/blob/main/CONDUCT.md) to get started.
