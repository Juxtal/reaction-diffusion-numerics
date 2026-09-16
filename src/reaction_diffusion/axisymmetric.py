from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .grid import AxisymmetricGrid
from .operators import build_axisymmetric_laplacian
from .picard import FieldSpec, PicardResult, picard_iterate


def default_boundary_mask(grid: AxisymmetricGrid) -> np.ndarray:
    """Dirichlet on the full domain edge: r = 1, z = 0, z = 1."""
    mask = np.zeros(grid.shape, dtype=bool)
    mask[-1, :] = True  # r = 1
    mask[:, 0] = True  # z = 0
    mask[:, -1] = True  # z = 1
    return mask


def solve_axisymmetric_reaction_diffusion(
    reaction: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    field_names: Sequence[str],
    *,
    nr: int = 60,
    nz: int = 60,
    boundary_mask: np.ndarray | None = None,
    diffusivities: Mapping[str, float] | None = None,
    boundary_values: Mapping[str, float] | None = None,
    initial_values: Mapping[str, float] | None = None,
    omega: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[AxisymmetricGrid, PicardResult]:
    """
    Solve a coupled steady axisymmetric reaction-diffusion system.

    `reaction` and the domain shape are the pluggable parts: `reaction` sets
    the coupling between `field_names`, and the shape is set via the grid
    resolution (`nr`, `nz`) and the Dirichlet boundary region
    (`boundary_mask`, a `(nr, nz)` boolean array; defaults to the full
    rectangle edge from `default_boundary_mask`).

    `diffusivities` / `boundary_values` / `initial_values` are optional
    `{field_name: float}` maps (uniform over the domain); any field left
    unspecified defaults to diffusivity 1.0, boundary value 0.0, initial 0.0.

    Returns `(grid, result)` where `result` is a `PicardResult`.
    """
    grid = AxisymmetricGrid(nr=nr, nz=nz)
    operator = build_axisymmetric_laplacian(grid)

    mask = default_boundary_mask(grid) if boundary_mask is None else boundary_mask
    mask = mask.reshape(-1)

    diffusivities = diffusivities or {}
    boundary_values = boundary_values or {}
    initial_values = initial_values or {}

    ones = np.ones(grid.size)
    specs = {
        name: FieldSpec(
            diffusivity=diffusivities.get(name, 1.0),
            dirichlet_mask=mask,
            dirichlet_values=boundary_values.get(name, 0.0) * ones,
            initial=initial_values.get(name, 0.0) * ones,
        )
        for name in field_names
    }

    result = picard_iterate(operator, specs, reaction, omega=omega, tol=tol, max_iter=max_iter)
    return grid, result
