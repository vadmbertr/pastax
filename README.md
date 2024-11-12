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
  - improve how the product operation is performed between the vector field and the control (support for `Location`, `Time` or `State` objects).

## Contributing

Contributions are welcomed!
See [CONTRIBUTING.md](https://github.com/vadmbertr/pastax/blob/main/CONDUCT.md) and [CONDUCT.md](https://github.com/vadmbertr/pastax/blob/main/CONDUCT.md) to get started.
