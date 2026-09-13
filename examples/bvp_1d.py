from pathlib import Path

import matplotlib.pyplot as plt

from reaction_diffusion.shooting import solve_second_order_bvp


def reaction_diffusion_rhs(x, c, dcdx):
    """
    Synthetic nonlinear reaction-diffusion example.

    Governing equation:

        c'' = Da * c / (K + c)

    This is a generic demonstration model and does not contain
    project-specific kinetic parameters.
    """

    da = 2.0
    k = 0.2

    return da * c / (k + c)


result = solve_second_order_bvp(
    rhs=reaction_diffusion_rhs,
    x_span=(0.0, 1.0),
    y_left=0.2,
    y_right=1.0,
    slope_bracket=(0.0, 5.0),
)


print(f"Converged: {result.converged}")
print(f"Initial slope: {result.initial_slope:.8f}")


plt.plot(
    result.x,
    result.y,
)

plt.xlabel("Dimensionless position")
plt.ylabel("Dimensionless concentration")
plt.title("1D nonlinear reaction-diffusion BVP")

plt.tight_layout()

out_path = Path(__file__).resolve().parent.parent / "figures" / "bvp_1d.png"
plt.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")