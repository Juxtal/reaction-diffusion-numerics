from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix


def steady_residual(
    operator: csr_matrix,
    fields: Mapping[str, np.ndarray],
    diffusivities: Mapping[str, float],
    rates: Mapping[str, np.ndarray],
    dirichlet_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Evaluate the steady-state PDE residual

        R_k = D_k * (operator @ f_k) - rate_k

    for each field, with Dirichlet nodes zeroed out (they are satisfied by
    construction once the boundary values are stamped into the solve).
    """
    residuals: dict[str, np.ndarray] = {}
    for name, f in fields.items():
        r = diffusivities[name] * (operator @ f) - rates[name]
        r = r.copy()
        r[dirichlet_mask] = 0.0
        residuals[name] = r
    return residuals


def residual_inf_norm(residuals: Mapping[str, np.ndarray]) -> float:
    """Largest absolute residual across all fields."""
    return max(float(np.max(np.abs(r))) for r in residuals.values())


def plot_convergence(history: Sequence[float], ax=None, *, label: str | None = None):
    """Semi-log convergence plot for a Picard (or other fixed-point) iteration."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.semilogy(range(1, len(history) + 1), history, marker="o", label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max-norm update")
    ax.set_title("Picard iteration convergence")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax
