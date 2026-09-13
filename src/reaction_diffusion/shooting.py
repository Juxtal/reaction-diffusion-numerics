from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


@dataclass
class ShootingResult:
    x: np.ndarray
    y: np.ndarray
    initial_slope: float
    converged: bool


def solve_second_order_bvp(
    rhs: Callable[[float, float, float], float],
    x_span: tuple[float, float],
    y_left: float,
    y_right: float,
    slope_bracket: tuple[float, float],
    *,
    num_points: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> ShootingResult:
    """
    Solve a second-order boundary-value problem

        y'' = rhs(x, y, y')

    with Dirichlet boundary conditions

        y(x_left)  = y_left
        y(x_right) = y_right

    using a shooting method.

    The unknown initial slope y'(x_left) is determined with
    a scalar root finder.
    """

    x_left, x_right = x_span
    x_eval = np.linspace(x_left, x_right, num_points)

    def integrate(initial_slope: float):
        def first_order_system(x, state):
            y, dydx = state
            d2ydx2 = rhs(x, y, dydx)
            return [dydx, d2ydx2]

        return solve_ivp(
            first_order_system,
            (x_left, x_right),
            [y_left, initial_slope],
            t_eval=x_eval,
            rtol=rtol,
            atol=atol,
        )

    def boundary_residual(initial_slope: float) -> float:
        solution = integrate(initial_slope)

        if not solution.success:
            raise RuntimeError(
                f"IVP integration failed: {solution.message}"
            )

        return solution.y[0, -1] - y_right

    root = root_scalar(
        boundary_residual,
        bracket=slope_bracket,
        method="bisect",
        xtol=1e-10,
    )

    if not root.converged:
        raise RuntimeError("Shooting method did not converge.")

    final_solution = integrate(root.root)

    return ShootingResult(
        x=final_solution.t,
        y=final_solution.y[0],
        initial_slope=root.root,
        converged=root.converged,
    )