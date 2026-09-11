"""Discontinuous Galerkin and Riemann solver in JAX."""

from importlib import metadata

from dg_solvax.solvers import DiscontinuousGalerkinSolver, RiemannSolver
from dg_solvax.utils.models import SystemPartialDiffEqs, SystemPartialDiffEqsProblem
from dg_solvax.utils.utils import get_orthonormal_spatial_functions

try:
    __version__ = metadata.version("dg-solvax")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "DiscontinuousGalerkinSolver",
    "RiemannSolver",
    "SystemPartialDiffEqs",
    "SystemPartialDiffEqsProblem",
    "get_orthonormal_spatial_functions",
]
