import numpy as np

from reaction_diffusion.shooting import solve_second_order_bvp


def test_linear_bvp_matches_analytic_solution():
    # y'' = 0 with y(0) = 0, y(1) = 1 has the exact solution y(x) = x.
    result = solve_second_order_bvp(
        rhs=lambda x, y, dydx: 0.0,
        x_span=(0.0, 1.0),
        y_left=0.0,
        y_right=1.0,
        slope_bracket=(-10.0, 10.0),
    )

    assert result.converged
    assert np.allclose(result.y, result.x, atol=1e-6)
