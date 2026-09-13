import numpy as np

from reaction_diffusion.grid import AxisymmetricGrid
from reaction_diffusion.operators import apply_dirichlet_bc, build_axisymmetric_laplacian


def test_laplacian_matches_analytic_quadratic():
    # For f(r, z) = r^2 + z^2, the axisymmetric Laplacian
    # d2f/dr2 + (1/r) df/dr + d2f/dz2 evaluates to the constant 4 + 2 = 6
    # everywhere, including the r = 0 axis (by the symmetric limit).
    grid = AxisymmetricGrid(nr=40, nz=40)
    L = build_axisymmetric_laplacian(grid)

    R, Z = grid.meshgrid()
    f = (R**2 + Z**2).reshape(-1)

    Lf = (L @ f).reshape(grid.shape)

    interior = Lf[:-1, 1:-1]  # exclude the untouched outer-boundary rows
    assert np.allclose(interior, 6.0, atol=1e-6)


def test_apply_dirichlet_bc_stamps_values():
    grid = AxisymmetricGrid(nr=5, nz=5)
    L = build_axisymmetric_laplacian(grid)
    rhs = np.zeros(grid.size)

    mask = np.zeros(grid.shape, dtype=bool)
    mask[-1, :] = True
    mask = mask.reshape(-1)
    values = np.full(grid.size, 3.0)

    L_bc, rhs_bc = apply_dirichlet_bc(L, rhs, mask, values)

    assert np.allclose(rhs_bc[mask], 3.0)
    for k in np.flatnonzero(mask):
        row = L_bc.getrow(k).toarray().ravel()
        assert row[k] == 1.0
        assert np.allclose(np.delete(row, k), 0.0)
