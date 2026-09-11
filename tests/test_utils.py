"""Tests for the utilities."""

import pytest
from sympy import integrate

from dg_solvax import get_orthonormal_spatial_functions


@pytest.mark.parametrize(
    "max_order, cell_size",
    [(0, 1.0), (1, 1.0), (2, 2.0), (3, 0.5), (4, 1.0)],
)
def test_get_orthonormal_spatial_functions(max_order, cell_size, _x):
    """Test that spatial functions returned by get_orthonormal_spatial_functions are orthonormal over one cell."""
    functions = get_orthonormal_spatial_functions(max_order, cell_size)

    for i, f_i in enumerate(functions):
        for j, f_j in enumerate(functions):
            inner_product = float(
                integrate(f_i * f_j, (_x, -cell_size / 2.0, cell_size / 2.0)).evalf()
            )

            assert inner_product == pytest.approx(float(i == j), abs=1e-12)
            # float(i == j) represents the Kronecker delta
