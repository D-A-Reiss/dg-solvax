"""Tests for the discontinuous Galerkin solver and the Riemann solver."""

import math

import jax.numpy as jnp
import pytest
from diffrax import SaveAt, Tsit5
from sympy import Integer, Symbol

from dg_solvax import (
    DiscontinuousGalerkinSolver,
    RiemannSolver,
    get_orthonormal_spatial_functions,
)


@pytest.fixture
def _spatial_domain_boundaries():
    return 0.0, 10.0


@pytest.fixture
def _num_spatial_cells():
    return 10


@pytest.fixture
def _solver(_spatial_domain_boundaries, _num_spatial_cells):
    return DiscontinuousGalerkinSolver(
        spatial_domain_boundaries=_spatial_domain_boundaries,
        num_spatial_cells=_num_spatial_cells,
        orthonormal_spatial_functions=get_orthonormal_spatial_functions(0, 1.0),
    )


@pytest.fixture
def _advection_problem():
    return {
        "diffeq_solver": Tsit5(),
        "t0": jnp.array(0.0),
        "t1": jnp.array(1.0),
        "dt0": jnp.array(0.125),
        "spatial_derivative_operator": jnp.array([[1.0, 0.0], [0.0, -1.0]]),
        "no_derivative_operator": jnp.zeros((2, 2)),
        "saveat_t": SaveAt(ts=[0.0, 1.0]),
        "saveat_x": jnp.linspace(0.0, 10.0, 101),
    }


class TestDiscontinuousGalerkinSolver:
    """Tests for ``DiscontinuousGalerkinSolver``."""

    def test_init_rejects_multivariate_spatial_function(
        self, _spatial_domain_boundaries, _num_spatial_cells, _x
    ):
        """Spatial functions depending on more than one variable are rejected."""
        y = Symbol("y")

        with pytest.raises(ValueError):
            DiscontinuousGalerkinSolver(
                spatial_domain_boundaries=_spatial_domain_boundaries,
                num_spatial_cells=_num_spatial_cells,
                orthonormal_spatial_functions=[_x + y],
            )

    def test_init_accepts_constant_spatial_function(
        self, _spatial_domain_boundaries, _num_spatial_cells
    ):
        """Variable-free (constant) spatial functions do not crash the setup."""
        constant_basis_solver = DiscontinuousGalerkinSolver(
            spatial_domain_boundaries=_spatial_domain_boundaries,
            num_spatial_cells=_num_spatial_cells,
            orthonormal_spatial_functions=[Integer(1)],
        )

        assert constant_basis_solver._num_spatial_functions == 1

    def test_coeffs_from_constant_initial_conditions(self, _solver, _num_spatial_cells):
        """Constant initial condition yield constant coefficients."""
        coeffs = _solver.compute_coeffs_from_initial_conditions_functions([Integer(2)])

        assert coeffs.shape == (_num_spatial_cells, 1, 1)
        assert jnp.allclose(coeffs, 2.0)

    def test_coeffs_from_linear_initial_conditions_order_0(
        self, _solver, _x, _num_spatial_cells
    ):
        """With order-0 spatial functions, a linear field is projected to cell averages."""
        coeffs = _solver.compute_coeffs_from_initial_conditions_functions([_x])

        assert jnp.allclose(coeffs.squeeze(), jnp.arange(_num_spatial_cells) + 0.5)

    def test_coeffs_from_linear_initial_conditions_order_1(self, _x):
        """With order-0 and order-1 spatial functions, a linear field is represented exactly."""
        order_1_solver = DiscontinuousGalerkinSolver(
            spatial_domain_boundaries=(0.0, 2.0),
            num_spatial_cells=2,
            orthonormal_spatial_functions=get_orthonormal_spatial_functions(1, 1.0),
        )
        coeffs = order_1_solver.compute_coeffs_from_initial_conditions_functions(
            [2 + 3 * _x]
        )

        assert coeffs.shape == (2, 2, 1)
        assert jnp.allclose(
            coeffs.squeeze(),
            jnp.array([[3.5, math.sqrt(3.0) / 2.0], [6.5, math.sqrt(3.0) / 2.0]]),
            atol=1e-6,
        )

    def test_coeffs_reject_multivariate_initial_conditions(self, _solver, _x):
        """Initial conditions depending on more than one variable are rejected."""
        t = Symbol("t")

        with pytest.raises(ValueError):
            _solver.compute_coeffs_from_initial_conditions_functions([_x + t])

    def test_check_representable_is_silent_for_exact_initial_conditions(self, _x):
        """Exactly representable initial conditions pass the Parseval check."""
        order_1_solver = DiscontinuousGalerkinSolver(
            spatial_domain_boundaries=(0.0, 2.0),
            num_spatial_cells=2,
            orthonormal_spatial_functions=get_orthonormal_spatial_functions(1, 1.0),
        )
        coeffs = order_1_solver.compute_coeffs_from_initial_conditions_functions(
            [2 + 3 * _x], check_whether_representable=True
        )

        assert coeffs.shape == (2, 2, 1)

    def test_partialdiffeqsolve_preserves_constant_state(
        self, _solver, _advection_problem
    ):
        """A constant state with matching exterior limits is an exact solution."""
        _, ys = _solver.partialdiffeqsolve(
            **_advection_problem,
            yt0=[Integer(2), Integer(2)],
            yx0=jnp.array([[2.0, 2.0]]),
            yx1=jnp.array([[2.0, 2.0]]),
        )

        assert ys.shape == (2, 101, 2)
        assert jnp.allclose(ys, 2.0, atol=1e-6)

    def test_partialdiffeqsolve_conserves_total_density(self):
        """TODO."""
        # TODO: implement this test


class TestRiemannSolver:
    """Tests for ``RiemannSolver``."""

    @pytest.mark.parametrize(
        "spatial_derivative_operator, y_left, y_right, expected_result",
        [
            (jnp.ones((1, 1)), jnp.ones(1), jnp.zeros(1), jnp.ones(1)),
            (jnp.ones((1, 1)), jnp.zeros(1), jnp.ones(1), jnp.zeros(1)),
            (-jnp.ones((1, 1)), jnp.ones(1), jnp.zeros(1), jnp.zeros(1)),
            (-jnp.ones((1, 1)), jnp.zeros(1), jnp.ones(1), jnp.ones(1)),
            (
                jnp.array([[1.0, 0.0], [0.0, -1.0]]),
                jnp.ones(2),
                jnp.zeros(2),
                jnp.array([1.0, 0.0]),
            ),
            (
                jnp.array([[1.0, 0.0], [0.0, -1.0]]),
                jnp.zeros(2),
                jnp.ones(2),
                jnp.array([0.0, 1.0]),
            ),
            (
                jnp.array([[0.0, 1.0], [1.0, 0.0]]),
                jnp.ones(2),
                jnp.zeros(2),
                jnp.ones(2),
            ),
            (
                jnp.array([[0.0, 1.0], [1.0, 0.0]]),
                jnp.zeros(2),
                jnp.ones(2),
                jnp.zeros(2),
            ),
        ],
    )
    def test_compute_cell_boundary_flux(
        self, spatial_derivative_operator, y_left, y_right, expected_result
    ):
        """Test the upwind computation of the cell boundary values."""
        result = RiemannSolver.compute_cell_boundary_values(
            spatial_derivative_operator, y_left, y_right
        )
        assert jnp.allclose(
            result,
            expected_result,
        ), f"\n{result=}\n{expected_result=}\n"

    @pytest.mark.parametrize(
        "reflection_coeffs, transmission_coeffs, y_left, y_right, expected_result_left_boundary, expected_result_right_boundary",
        [
            (
                jnp.ones((1, 1)),
                jnp.zeros((1, 1)),
                jnp.array([1.0, 0.0]),
                jnp.zeros(2),
                jnp.zeros(2),
                jnp.array([0.0, 1.0]),
            ),
            (
                jnp.ones((1, 1)),
                jnp.zeros((1, 1)),
                jnp.zeros(2),
                jnp.ones(2),
                jnp.array([1.0, 0.0]),
                jnp.zeros(2),
            ),
            (
                jnp.zeros((1, 1)),
                jnp.ones((1, 1)),
                jnp.array([1.0, 0.0]),
                jnp.zeros(2),
                jnp.array([1.0, 0.0]),
                jnp.zeros(2),
            ),
            (
                jnp.zeros((1, 1)),
                jnp.ones((1, 1)),
                jnp.zeros(2),
                jnp.ones(2),
                jnp.zeros(2),
                jnp.array([0.0, 1.0]),
            ),
        ],
    )
    def test_compute_domain_boundary_flux(
        self,
        reflection_coeffs,
        transmission_coeffs,
        y_left,
        y_right,
        expected_result_left_boundary,
        expected_result_right_boundary,
    ):
        """Test the domain boundary values with reflection and transmission."""
        spatial_derivative_operator = jnp.array([[1.0, 0.0], [0.0, -1.0]])

        result = RiemannSolver.compute_domain_boundary_values(
            spatial_derivative_operator,
            y_left,
            y_right,
            reflection_coeffs_left_to_left=reflection_coeffs,
            reflection_coeffs_right_to_right=None,
            transmission_coeffs_left_to_right=None,
            transmission_coeffs_right_to_left=transmission_coeffs,
        )
        assert jnp.allclose(
            result,
            expected_result_right_boundary,
        ), f"\n{result=}\n{expected_result_right_boundary=}\n"

        result = RiemannSolver.compute_domain_boundary_values(
            spatial_derivative_operator,
            y_left,
            y_right,
            reflection_coeffs_left_to_left=None,
            reflection_coeffs_right_to_right=reflection_coeffs,
            transmission_coeffs_left_to_right=transmission_coeffs,
            transmission_coeffs_right_to_left=None,
        )
        assert jnp.allclose(
            result,
            expected_result_left_boundary,
        ), f"\n{result=}\n{expected_result_left_boundary=}\n"

    def test_compute_cell_boundary_values_batched(self):
        """Boundaries stacked in a leading axis are upwinded elementwise."""
        spatial_derivative_operator = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        y_left = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_right = jnp.zeros((3, 2))

        result = RiemannSolver.compute_cell_boundary_values(
            spatial_derivative_operator, y_left, y_right
        )

        assert result.shape == (3, 2)
        assert jnp.allclose(
            result,
            jnp.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        ), f"\n{result=}\n"

    def test_compute_domain_boundary_values_requires_exactly_one_pair(self):
        """Both or neither boundary coefficient pairs are rejected."""
        spatial_derivative_operator = jnp.eye(2)

        with pytest.raises(ValueError):
            RiemannSolver.compute_domain_boundary_values(
                spatial_derivative_operator,
                jnp.zeros(2),
                jnp.zeros(2),
                reflection_coeffs_left_to_left=jnp.ones((1, 1)),
                reflection_coeffs_right_to_right=jnp.ones((1, 1)),
                transmission_coeffs_left_to_right=jnp.ones((1, 1)),
                transmission_coeffs_right_to_left=jnp.ones((1, 1)),
            )

        with pytest.raises(ValueError):
            RiemannSolver.compute_domain_boundary_values(
                spatial_derivative_operator,
                jnp.zeros(2),
                jnp.zeros(2),
                reflection_coeffs_left_to_left=None,
                reflection_coeffs_right_to_right=None,
                transmission_coeffs_left_to_right=None,
                transmission_coeffs_right_to_left=None,
            )

    def test_compute_domain_boundary_values_rejects_dimension_mismatch(self):
        """Scatter matrices not matching the system dimension are rejected."""
        spatial_derivative_operator = jnp.eye(2)

        with pytest.raises(ValueError):
            RiemannSolver.compute_domain_boundary_values(
                spatial_derivative_operator,
                jnp.zeros(2),
                jnp.zeros(2),
                reflection_coeffs_left_to_left=None,
                reflection_coeffs_right_to_right=jnp.ones((2, 2)),
                transmission_coeffs_left_to_right=jnp.ones((2, 2)),
                transmission_coeffs_right_to_left=None,
            )
