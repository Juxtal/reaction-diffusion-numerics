import numpy as np

from reaction_diffusion.grid import AxisymmetricGrid
from reaction_diffusion.operators import build_axisymmetric_laplacian
from reaction_diffusion.picard import FieldSpec, picard_iterate


def test_picard_recovers_constant_field_with_no_reaction():
    # With zero reaction and constant Dirichlet data on the whole boundary,
    # the steady solution of the Laplace equation is that same constant
    # everywhere in the interior.
    grid = AxisymmetricGrid(nr=15, nz=15)
    L = build_axisymmetric_laplacian(grid)

    boundary = np.zeros(grid.shape, dtype=bool)
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    boundary = boundary.reshape(-1)

    specs = {
        "u": FieldSpec(
            diffusivity=1.0,
            dirichlet_mask=boundary,
            dirichlet_values=np.full(grid.size, 2.0),
            initial=np.zeros(grid.size),
        )
    }

    def zero_reaction(fields):
        return {"u": np.zeros_like(fields["u"])}

    result = picard_iterate(L, specs, zero_reaction, omega=1.0, tol=1e-10, max_iter=50)

    assert result.converged
    assert np.allclose(result.fields["u"], 2.0, atol=1e-6)
