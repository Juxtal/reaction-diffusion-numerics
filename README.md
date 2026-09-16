# reaction-diffusion-numerics

Small, self-contained numerical toolkit for steady nonlinear reaction-diffusion
problems: a 1-D boundary-value solver based on the shooting method, and a
coupled 2-D axisymmetric solver based on Picard (fixed-point) iteration over a
sparse finite-difference Laplacian.

This is a generic numerics library. It ships with synthetic demonstration
reactions only — no project-specific kinetics, fitted parameters, or
experimental data.

## Background

This library started as a refactor of internship code: pulling ad-hoc,
one-off reaction-diffusion solving scripts apart into a small, reusable,
tested toolkit. It's a simplified, standalone version — generalized to
synthetic demo reactions, with no internship-specific kinetics, parameters,
or data.

## Structure

```
examples/
  bvp_1d.py                     1-D shooting-method demo
  axisymmetric_2d.py            2-D coupled Picard-iteration demo
  custom_reaction_and_shape.py  same solver, custom reaction + domain shape
figures/                        output figures from the examples
src/reaction_diffusion/
  grid.py                       uniform axisymmetric (r, z) grid
  operators.py                  sparse axisymmetric Laplacian + Dirichlet BC helper
  picard.py                     coupled Picard fixed-point iteration
  axisymmetric.py               solve_axisymmetric_reaction_diffusion convenience wrapper
  shooting.py                   1-D second-order BVP solver (shooting method)
  diagnostics.py                residual checks and convergence plots
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
python examples/custom_reaction_and_shape.py
```

All three scripts print convergence information and save a figure under
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
- `solve_axisymmetric_reaction_diffusion(reaction, field_names, *, nr=60,
  nz=60, boundary_mask=None, diffusivities=None, boundary_values=None,
  initial_values=None, ...)` — convenience wrapper that assembles the grid,
  operator, and `FieldSpec`s and runs `picard_iterate` in one call. The
  reaction and the domain shape (grid resolution and Dirichlet boundary
  region) are the pluggable parts; everything else falls back to a uniform
  default per field. Returns `(grid, result)`.
- `solve_second_order_bvp(rhs, x_span, y_left, y_right, slope_bracket, ...)`
  — solves `y'' = rhs(x, y, y')` with Dirichlet boundary conditions via the
  shooting method.
- `steady_residual` / `residual_inf_norm` / `plot_convergence` — diagnostics
  for checking convergence and residual size.
