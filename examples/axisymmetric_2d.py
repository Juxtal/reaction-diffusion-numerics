from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reaction_diffusion import (
    AxisymmetricGrid,
    FieldSpec,
    build_axisymmetric_laplacian,
    picard_iterate,
    plot_convergence,
)


def reaction(fields):
    """
    Generic saturating (Michaelis-Menten-style) coupling between two fields:
    consumption of `u` produces `v` at the same local rate.

    This is a synthetic demonstration reaction, not tied to any specific
    physical system or fitted parameter set.
    """
    k_max = 5.0
    k_half = 0.3

    rate = k_max * fields["u"] / (k_half + fields["u"])
    return {"u": rate, "v": -rate}


def main() -> None:
    grid = AxisymmetricGrid(nr=60, nz=60)
    operator = build_axisymmetric_laplacian(grid)

    boundary = np.zeros(grid.shape, dtype=bool)
    boundary[-1, :] = True  # r = 1
    boundary[:, 0] = True  # z = 0
    boundary[:, -1] = True  # z = 1
    boundary = boundary.reshape(-1)

    ones = np.ones(grid.size)
    specs = {
        "u": FieldSpec(
            diffusivity=1.0,
            dirichlet_mask=boundary,
            dirichlet_values=1.0 * ones,
            initial=0.5 * ones,
        ),
        "v": FieldSpec(
            diffusivity=0.5,
            dirichlet_mask=boundary,
            dirichlet_values=0.0 * ones,
            initial=0.0 * ones,
        ),
    }

    result = picard_iterate(operator, specs, reaction, omega=0.5, tol=1e-8, max_iter=200)
    print(f"Converged: {result.converged} after {result.iterations} iterations")

    u = result.fields["u"].reshape(grid.shape)
    v = result.fields["v"].reshape(grid.shape)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, field, title in zip(axes[:2], (u, v), ("u", "v")):
        im = ax.pcolormesh(grid.z, grid.r, field, shading="auto")
        ax.set_xlabel("z")
        ax.set_ylabel("r")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plot_convergence(result.history, ax=axes[2])

    fig.tight_layout()

    out_path = Path(__file__).resolve().parent.parent / "figures" / "axisymmetric_2d.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
