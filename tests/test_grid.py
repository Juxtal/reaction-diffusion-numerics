import numpy as np

from reaction_diffusion.grid import AxisymmetricGrid


def test_grid_shape_and_spacing():
    grid = AxisymmetricGrid(nr=11, nz=21)

    assert grid.shape == (11, 21)
    assert grid.size == 11 * 21
    assert np.isclose(grid.dr, 1.0 / 10)
    assert np.isclose(grid.dz, 1.0 / 20)


def test_index_is_bijective():
    grid = AxisymmetricGrid(nr=5, nz=4)

    seen = set()
    for i in range(grid.nr):
        for j in range(grid.nz):
            k = grid.index(i, j)
            assert k not in seen
            seen.add(k)

    assert seen == set(range(grid.size))
