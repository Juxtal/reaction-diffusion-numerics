from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .grid import AxisymmetricGrid


def build_axisymmetric_laplacian(
    grid: AxisymmetricGrid,
    *,
    r_scale: float = 1.0,
    z_scale: float = 1.0,
) -> csr_matrix:
    """
    Assemble the (unscaled) axisymmetric Laplacian operator

        L = (1 / r_scale**2) * (d2/dr2 + (1/r) d/dr) + (1 / z_scale**2) * d2/dz2

    as a sparse matrix acting on a flattened (nr * nz,) field, using
    second-order central differences.

    The r = 0 axis is handled with the standard symmetric limit
    (1/r) df/dr -> d2f/dr2 as r -> 0.

    Rows on the outer domain boundary (r = 1, z = 0, z = 1) are left as zero
    rows; combine with `apply_dirichlet_bc` (or your own boundary treatment)
    before solving.
    """
    nr, nz = grid.nr, grid.nz
    dr, dz = grid.dr, grid.dz
    n = grid.size

    inv_r2 = 1.0 / (r_scale**2)
    inv_z2 = 1.0 / (z_scale**2)

    L = lil_matrix((n, n))

    for i in range(1, nr - 1):
        inv_r = 1.0 / grid.r[i]
        for j in range(1, nz - 1):
            k = grid.index(i, j)
            k_rp, k_rm = grid.index(i + 1, j), grid.index(i - 1, j)
            k_zp, k_zm = grid.index(i, j + 1), grid.index(i, j - 1)

            L[k, k] = inv_r2 * (-2.0 / dr**2) + inv_z2 * (-2.0 / dz**2)
            L[k, k_rp] = inv_r2 * (1.0 / dr**2 + inv_r / (2.0 * dr))
            L[k, k_rm] = inv_r2 * (1.0 / dr**2 - inv_r / (2.0 * dr))
            L[k, k_zp] = inv_z2 * (1.0 / dz**2)
            L[k, k_zm] = inv_z2 * (1.0 / dz**2)

    # Symmetry axis r = 0: (1/r) df/dr -> d2f/dr2, doubling the radial term.
    i = 0
    for j in range(1, nz - 1):
        k = grid.index(i, j)
        k_rp = grid.index(i + 1, j)
        k_zp, k_zm = grid.index(i, j + 1), grid.index(i, j - 1)

        L[k, k] = inv_r2 * (-4.0 / dr**2) + inv_z2 * (-2.0 / dz**2)
        L[k, k_rp] = inv_r2 * (4.0 / dr**2)
        L[k, k_zp] = inv_z2 * (1.0 / dz**2)
        L[k, k_zm] = inv_z2 * (1.0 / dz**2)

    return L.tocsr()


def apply_dirichlet_bc(
    operator: csr_matrix,
    rhs: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Overwrite rows of `operator` / `rhs` where `mask` is True with an
    identity row enforcing `field == values`, leaving all other rows
    untouched.

    `mask` and `values` are flattened (n,) arrays matching `rhs`. Applying
    this after scaling the operator by a diffusivity (e.g. `D * operator`)
    is safe: the boundary rows are replaced outright, independent of `D`.
    """
    result = operator.tolil()
    rhs = rhs.copy()

    for k in np.flatnonzero(mask):
        result.rows[k] = [k]
        result.data[k] = [1.0]
    rhs[mask] = values[mask]

    return result.tocsr(), rhs
