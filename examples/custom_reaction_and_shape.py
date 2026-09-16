"""
Shows how to call `solve_axisymmetric_reaction_diffusion` directly with a
custom reaction and a custom domain shape, instead of the fixed two-field
setup in `axisymmetric_2d.py`.

Two things are swapped out here:
  - the reaction: a single-field, self-inhibited decay,
    `rate = k * c / (1 + k_inhibit * c)`
  - the shape: an interior disk held at a fixed value (like an embedded
    sink), added on top of the usual outer boundary.

Note: the outer domain edge (r = 1, z = 0, z = 1) must always stay in
`boundary_mask`. The discretization only has a valid equation there once a
Dirichlet condition is stamped in; an unmasked edge row leaves the linear
system singular. Extra *interior* points, like the obstacle disk below, can
be added freely on top of that.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reaction_diffusion import AxisymmetricGrid, solve_axisymmetric_reaction_diffusion


def inhibited_decay_reaction(fields):
    """
    Synthetic single-field reaction: first-order decay of `c`, self-inhibited
    so the rate saturates at high concentration. Demonstration only.
    """
    k = 3.0
    k_inhibit = 0.5

    # Positive rate here means "c is being consumed" (D * L @ c = rate), the
    # same sign convention as the `u` field in axisymmetric_2d.py: it pulls
    # the interior below the boundary value.
    rate = k * fields["c"] / (1.0 + k_inhibit * fields["c"])
    return {"c": rate}


def obstacle_boundary_mask(grid: AxisymmetricGrid):
    """
    Default outer edge, plus an interior disk (center r=0.3, z=0.5,
    radius 0.12) held at a fixed value -- a different domain shape than the
    plain rectangle used in axisymmetric_2d.py.
    """
    mask = default_edge_mask(grid)

    r, z = grid.meshgrid()
    obstacle = (r - 0.3) ** 2 + (z - 0.5) ** 2 <= 0.12**2
    mask |= obstacle

    return mask


def default_edge_mask(grid: AxisymmetricGrid):
    mask = np.zeros(grid.shape, dtype=bool)
    mask[-1, :] = True  # r = 1
    mask[:, 0] = True  # z = 0
    mask[:, -1] = True  # z = 1
    return mask


def main() -> None:
    nr, nz = 60, 60
    grid = AxisymmetricGrid(nr=nr, nz=nz)
    mask = obstacle_boundary_mask(grid)

    grid, result = solve_axisymmetric_reaction_diffusion(
        inhibited_decay_reaction,
        field_names=("c",),
        nr=nr,
        nz=nz,
        boundary_mask=mask,
        diffusivities={"c": 1.0},
        boundary_values={"c": 1.0},
        initial_values={"c": 0.0},
    )
    print(f"Converged: {result.converged} after {result.iterations} iterations")

    c = result.fields["c"].reshape(grid.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.pcolormesh(grid.z, grid.r, c, shading="auto")
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.set_title("Custom reaction + custom shape")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "custom_reaction_and_shape.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
