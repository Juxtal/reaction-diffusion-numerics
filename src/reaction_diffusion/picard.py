from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .operators import apply_dirichlet_bc


@dataclass
class FieldSpec:
    """Static description of one field in a coupled Picard iteration."""

    diffusivity: float
    dirichlet_mask: np.ndarray
    dirichlet_values: np.ndarray
    initial: np.ndarray


@dataclass
class PicardResult:
    fields: dict[str, np.ndarray]
    history: list[float]
    converged: bool
    iterations: int


def picard_iterate(
    operator: csr_matrix,
    specs: Mapping[str, FieldSpec],
    reaction: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    *,
    omega: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 250,
) -> PicardResult:
    """
    Solve a steady, coupled reaction-diffusion system

        D_k * (operator @ f_k) = reaction(f)[k]      for each field k

    by fixed-point (Picard) iteration with under-relaxation `omega`.

    `reaction` is evaluated on the current field values and must return one
    right-hand-side array per field name in `specs`. Each linear
    sub-problem is solved against the shared `operator`, scaled by that
    field's diffusivity, with its own Dirichlet boundary condition stamped
    in via `apply_dirichlet_bc`.

    Iteration stops when the largest max-norm update across all fields
    drops below `tol`, or after `max_iter` iterations.
    """
    fields = {name: spec.initial.copy() for name, spec in specs.items()}
    history: list[float] = []

    for iteration in range(1, max_iter + 1):
        rates = reaction(fields)

        updated: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            system = spec.diffusivity * operator
            system_bc, rhs_bc = apply_dirichlet_bc(
                system, rates[name], spec.dirichlet_mask, spec.dirichlet_values
            )
            candidate = spsolve(system_bc, rhs_bc)
            updated[name] = (1.0 - omega) * fields[name] + omega * candidate

        error = max(
            float(np.max(np.abs(updated[name] - fields[name]))) for name in specs
        )
        history.append(error)
        fields = updated

        if error < tol:
            return PicardResult(
                fields=fields, history=history, converged=True, iterations=iteration
            )

    return PicardResult(
        fields=fields, history=history, converged=False, iterations=max_iter
    )
