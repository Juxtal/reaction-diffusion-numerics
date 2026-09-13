from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AxisymmetricGrid:
    """
    Uniform grid on a dimensionless axisymmetric domain (r, z) in [0, 1] x [0, 1].

    `r` is the radial coordinate, with a symmetry axis at r = 0, and `z` is the
    axial coordinate.
    """

    nr: int
    nz: int
    r: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    dr: float = field(init=False)
    dz: float = field(init=False)

    def __post_init__(self) -> None:
        if self.nr < 3 or self.nz < 3:
            raise ValueError("nr and nz must each be at least 3")

        self.r = np.linspace(0.0, 1.0, self.nr)
        self.z = np.linspace(0.0, 1.0, self.nz)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nr, self.nz)

    @property
    def size(self) -> int:
        return self.nr * self.nz

    def index(self, i: int, j: int) -> int:
        """Flatten a (radial, axial) grid index into a 1-D vector index."""
        return i * self.nz + j

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 2-D (r, z) coordinate arrays, each shaped like the grid."""
        return np.meshgrid(self.r, self.z, indexing="ij")
