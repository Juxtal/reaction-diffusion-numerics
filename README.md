# reaction-diffusion-numerics

Small, self-contained numerical toolkit for steady nonlinear reaction-diffusion
problems: a 1-D boundary-value solver based on the shooting method, and a
coupled 2-D axisymmetric solver based on Picard (fixed-point) iteration over a
sparse finite-difference Laplacian.

This is a generic numerics library. It ships with synthetic demonstration
reactions only — no project-specific kinetics, fitted parameters, or
experimental data.

## Structure

```
examples/
  bvp_1d.py            1-D shooting-method demo
  axisymmetric_2d.py   2-D coupled Picard-iteration demo
figures/                output figures from the examples
src/reaction_diffusion/
  grid.py               uniform axisymmetric (r, z) grid
  operators.py          sparse axisymmetric Laplacian + Dirichlet BC helper
  picard.py             coupled Picard fixed-point iteration
  shooting.py           1-D second-order BVP solver (shooting method)
  diagnostics.py        residual checks and convergence plots
tests/
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
python examples/bvp_1d.py
python examples/axisymmetric_2d.py
```

Both scripts print convergence information and save a figure under
`figures/`.

## Running tests

```bash
pytest
```

## API overview

- `AxisymmetricGrid(nr, nz)` — a uniform grid on the dimensionless domain
  `(r, z) in [0, 1] x [0, 1]`.
- `build_axisymmetric_laplacian(grid, r_scale=1.0, z_scale=1.0)` — assembles
  the (unscaled) axisymmetric Laplacian as a sparse matrix, with the
  symmetric limit applied at the `r = 0` axis.
- `apply_dirichlet_bc(operator, rhs, mask, values)` — stamps Dirichlet rows
  into a sparse linear system.
- `FieldSpec` / `picard_iterate(operator, specs, reaction, ...)` — solves a
  coupled steady reaction-diffusion system `D_k * (L @ f_k) = reaction(f)[k]`
  by under-relaxed fixed-point iteration.
- `solve_second_order_bvp(rhs, x_span, y_left, y_right, slope_bracket, ...)`
  — solves `y'' = rhs(x, y, y')` with Dirichlet boundary conditions via the
  shooting method.
- `steady_residual` / `residual_inf_norm` / `plot_convergence` — diagnostics
  for checking convergence and residual size.
