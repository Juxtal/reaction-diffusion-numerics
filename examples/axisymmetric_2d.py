from pathlib import Path

import matplotlib.pyplot as plt

from reaction_diffusion import plot_convergence, solve_axisymmetric_reaction_diffusion


def saturating_uptake_reaction(fields):
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
    grid, result = solve_axisymmetric_reaction_diffusion(
        saturating_uptake_reaction,
        field_names=("u", "v"),
        nr=60,
        nz=60,
        diffusivities={"u": 1.0, "v": 0.5},
        boundary_values={"u": 1.0, "v": 0.0},
        initial_values={"u": 0.5, "v": 0.0},
    )
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
