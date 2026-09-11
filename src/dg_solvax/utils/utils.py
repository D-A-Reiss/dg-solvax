"""Symbolic math utilities built on sympy and JAX.

Provides orthonormal spatial basis functions via Gram-Schmidt and the L2 function inner product over one cell.
"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array
from sympy import Expr, Symbol, integrate, sqrt


def get_orthonormal_spatial_functions(max_order: int, cell_size: float) -> list[Expr]:
    """Return orthonormal spatial basis functions up to the given order.

    The monomials up to ``max_order`` are orthonormalized over one cell of
    the given size.

    Args:
        max_order: Highest polynomial order of the basis functions.
        cell_size: Size of the spatial cell the basis functions are
            orthonormal over.

    Returns:
        Orthonormal sympy expressions of the position variable ``x``.
    """
    x = Symbol("x")

    return _compute_orthogonal_vectors_by_Gram_Schmidt(
        [x**n for n in range(max_order + 1)],
        inner_product_func=partial(_compute_function_inner_product, cell_size),
    )


def _compute_orthogonal_vectors_by_Gram_Schmidt(
    vectors: list[Array | Expr],
    inner_product_func: Callable = jnp.inner,
    normalize: bool = True,
) -> list[Array | Expr]:
    try:
        inner_product_func(vectors[0], vectors[0])

    except Exception:  # noqa: BLE001
        raise ValueError(
            f"input {inner_product_func=} is not applicable to all vectors {vectors=}"
        )

    orthogonal_vectors = []

    for k, vk in enumerate(vectors):
        uk = vk - sum(
            inner_product_func(vk, uj) / inner_product_func(uj, uj) * uj
            for uj in orthogonal_vectors
        )
        if normalize:
            orthogonal_vectors.append(uk / sqrt(inner_product_func(uk, uk)))
        else:
            orthogonal_vectors.append(uk)

    return orthogonal_vectors


def _compute_function_inner_product(
    interval_length: float,
    func_of_x_1: Expr,
    func_of_x_2: Expr,
) -> Expr:
    x = Symbol("x")

    return integrate(
        func_of_x_1 * func_of_x_2, (x, -interval_length / 2.0, interval_length / 2.0)
    )


def _validate_is_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"{value} is not positive")
    return value


def _validate_is_non_negative(value: int) -> int:
    if value < 0:
        raise ValueError(f"{value} is negative")
    return value


def _validate_is_at_least_2(value: int) -> int:
    if value < 2:
        raise ValueError(f"{value} must be >= 2")
    return value


def _validate_is_square_matrix(
    matrix: list[list[float]],
) -> list[list[float]]:
    num_rows = len(matrix)

    for row in matrix:
        if len(row) != num_rows:
            raise ValueError(f"{matrix=} is not a square matrix")

    return matrix
