"""Discontinuous Galerkin and Riemann solver on JAX."""

from functools import partial
from warnings import warn

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from diffrax import (
    AbstractProgressMeter,
    AbstractSolver,
    AbstractStepSizeController,
    ConstantStepSize,
    NoProgressMeter,
    ODETerm,
    SaveAt,
    Solution,
    diffeqsolve,
)
from jax import Array
from sympy import Expr, Symbol, diff, integrate

type Position = float
type Time = float
type SaveAtX = Array
type ExprOfT = Expr
type ExprOfX = Expr

_SAVEAT_T = SaveAt(t1=True)
_STEPSIZE_CONTROLLER = ConstantStepSize()
_PROGRESS_METER = NoProgressMeter()


class RiemannSolver:
    """Solve local Riemann problems at cell and domain boundaries.

    The limits of a vector field at a boundary are decomposed into
    the eigenvectors of the spatial derivative operator; the upwind side is
    selected per eigenvalue (left for positive, right for negative).
    """

    @staticmethod
    @partial(jax.jit, static_argnums=3)
    # -> separate XLA compilation per value of sort -> sort must be a compile-time constant!
    # -> adapt "static_argnums=3" if arguments are reordered
    def _compute_eigenvectors_coefficients(
        spatial_derivative_operator: Array,
        y_left: Array,
        y_right: Array,
        sort: bool = False,
    ) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = jnp.linalg.eig(spatial_derivative_operator)
        # eigenvectors resulting shape: (..., N, N), each column is one eigenvector

        if sort:
            indices = jnp.argsort(eigenvalues, axis=-1)
            eigenvalues = eigenvalues[indices]
            eigenvectors = jnp.take_along_axis(
                eigenvectors, jnp.expand_dims(indices, axis=-2), axis=-1
            )

        coefficients_left = jsc.linalg.solve(eigenvectors, y_left[..., None]).squeeze(
            -1
        )
        coefficients_right = jsc.linalg.solve(eigenvectors, y_right[..., None]).squeeze(
            -1
        )
        # coefficients resulting shape: (..., N)

        return eigenvectors.real, jnp.where(
            eigenvalues > 0, coefficients_left, coefficients_right
        ).real

    @staticmethod
    @jax.jit
    def compute_cell_boundary_values(
        spatial_derivative_operator: Array,
        y_left: Array,
        y_right: Array,
    ) -> Array:
        """Compute the upwind values at the cell boundaries.

        Args:
            spatial_derivative_operator: Constant matrix multiplying the
                spatial derivative of a vector field.
            y_left: Limits of a vector field on the left side of
                the boundaries.
            y_right: Limits of a vector field on the right side
                of the boundaries.

        Returns:
            The upwind values of the vector field at the cell
            boundaries.
        """
        eigenvectors, coefficients = RiemannSolver._compute_eigenvectors_coefficients(
            spatial_derivative_operator, y_left, y_right
        )
        coefficients = jnp.expand_dims(
            coefficients, axis=-2
        )  # for broadcasting in next step

        return jnp.sum(coefficients * eigenvectors, axis=-1)

    @staticmethod
    @jax.jit
    def compute_domain_boundary_values(
        spatial_derivative_operator: Array,
        y_left: Array,
        y_right: Array,
        *,
        reflection_coeffs_left_to_left: Array | None,
        reflection_coeffs_right_to_right: Array | None,
        transmission_coeffs_left_to_right: Array | None,
        transmission_coeffs_right_to_left: Array | None,
    ) -> Array:
        """Compute the value at a domain boundary.

        The boundary is described by reflection and transmission coefficients,
        from which a scatter matrix is assembled and applied to the
        sorted-eigenvalue decomposition of the limits.

        Args:
            spatial_derivative_operator: Constant matrix multiplying the
                spatial derivative of a vector field.
            y_left: Limit of a vector field on the left of the
                boundary.
            y_right: Limit of a vector field on the right of the
                boundary.
            reflection_coeffs_left_to_left: Reflection coefficients from the
                left side back to the left side, at the right domain
                boundary.
            reflection_coeffs_right_to_right: Reflection coefficients from
                the right side back to the right side, at the left domain
                boundary.
            transmission_coeffs_left_to_right: Transmission coefficients from
                the left to the right side, at the left domain boundary.
            transmission_coeffs_right_to_left: Transmission coefficients from
                the right to the left side, at the right domain boundary.

        Returns:
            The values of a vector field at the domain boundary.

        Raises:
            ValueError: If not exactly one of the two boundary coefficient
                pairs is given, or if the assembled scatter matrix does not
                match the dimension of the spatial derivative operator.
        """
        # initial definitions and checks
        is_left_domain_boundary = (
            reflection_coeffs_right_to_right is not None
            and transmission_coeffs_left_to_right is not None
        )
        is_right_domain_boundary = (
            reflection_coeffs_left_to_left is not None
            and transmission_coeffs_right_to_left is not None
        )

        if not (is_left_domain_boundary ^ is_right_domain_boundary):
            raise ValueError(
                f"exactly one of {is_left_domain_boundary=} and {is_right_domain_boundary=} must be true"
            )

        if is_left_domain_boundary:
            scatter_matrix = jnp.concatenate(
                (reflection_coeffs_right_to_right, transmission_coeffs_left_to_right),
                axis=0,
            )
            scatter_matrix = jnp.concatenate(
                (jnp.zeros_like(scatter_matrix), scatter_matrix), axis=1
            )

        if is_right_domain_boundary:
            scatter_matrix = jnp.concatenate(
                (transmission_coeffs_right_to_left, reflection_coeffs_left_to_left),
                axis=0,
            )
            scatter_matrix = jnp.concatenate(
                (scatter_matrix, jnp.zeros_like(scatter_matrix)), axis=1
            )

        if scatter_matrix.shape[0] != spatial_derivative_operator.shape[-2]:
            raise ValueError(
                f"{scatter_matrix.shape[0]=} == {spatial_derivative_operator.shape[-2]} is required"
            )

        # computations
        eigenvectors, coefficients = RiemannSolver._compute_eigenvectors_coefficients(
            spatial_derivative_operator,
            y_left,
            y_right,
            sort=True,
        )

        return jnp.einsum(
            "ij,...i,...kj->...k",  # because each column of "eigenvectors" is one eigenvector
            scatter_matrix,
            coefficients,
            eigenvectors,
        )


class DiscontinuousGalerkinSolver:
    """Solve (systems of) partial differential equations (PDEs) via the discontinuous Galerkin method on a uniform grid.

    The input vector field is represented per cell by coefficients w.r.t.
    orthonormal spatial basis functions.
    """

    def __init__(
        self,
        spatial_domain_boundaries: tuple[Position, Position],
        num_spatial_cells: int,
        orthonormal_spatial_functions: list[ExprOfX],
    ):
        """Initialize the grid and the orthonormal spatial basis functions.

        Args:
            spatial_domain_boundaries: Left and right boundary of the
                spatial domain.
            num_spatial_cells: Number of uniform cells the spatial domain is
                divided into.
            orthonormal_spatial_functions: Orthonormal SymPy expressions of
                a single position variable, used as per-cell basis.

        Raises:
            ValueError: If a basis function depends on more than 1 free variable.
        """
        self._x = Symbol("x")

        for f in orthonormal_spatial_functions:
            if len(f.free_symbols) > 1:
                raise ValueError(
                    f"function {f=} in orthonormal_spatial_functions depends on more than 1 free variable; ambiguity cannot be treated"
                )

        self._spatial_domain_boundaries = spatial_domain_boundaries
        self._num_spatial_cells = num_spatial_cells
        self._spatial_step_size = (
            spatial_domain_boundaries[1] - spatial_domain_boundaries[0]
        ) / num_spatial_cells
        self._spatial_cell_boundaries = jnp.linspace(
            spatial_domain_boundaries[0],
            spatial_domain_boundaries[1],
            num_spatial_cells + 1,
        )
        self.spatial_cell_middles = (
            self._spatial_cell_boundaries[:-1] + self._spatial_step_size / 2
        )

        self._orthonormal_spatial_functions = [
            f.subs(f.free_symbols.pop(), self._x) if f.free_symbols else f
            for f in orthonormal_spatial_functions
        ]
        self._num_spatial_functions = len(self._orthonormal_spatial_functions)

    def _compute_basis_functions_cell_boundary_values(self) -> Array:
        boundary_values = []

        for i in range(self._num_spatial_functions):
            boundary_values.append([])

            for x in (-self._spatial_step_size / 2.0, +self._spatial_step_size / 2.0):
                boundary_values[i].append(
                    float(
                        self._orthonormal_spatial_functions[i].evalf(subs={self._x: x})
                    )
                )

        return jnp.array(boundary_values)

    def _compute_basis_functions_values(self, positions: Array) -> tuple[Array, Array]:
        left_cell_boundaries = self._spatial_cell_boundaries[:-1]
        right_cell_boundaries = self._spatial_cell_boundaries[1:]

        positions = jnp.expand_dims(positions, axis=-1)
        is_larger = positions >= left_cell_boundaries
        is_smaller = positions <= right_cell_boundaries
        # resulting shape: M, N, where M is no. of positions and N is no. of cells
        is_in_cell = is_larger & is_smaller
        cell_numbers = jnp.argmax(is_in_cell, axis=1, keepdims=True)
        cell_middles = jnp.take_along_axis(
            jnp.expand_dims(self.spatial_cell_middles, axis=0), cell_numbers, axis=1
        ).squeeze(axis=1)
        relative_positions = positions - cell_middles

        function_values = []

        for i, x in enumerate(relative_positions):
            function_values.append([])

            for f in self._orthonormal_spatial_functions:
                function_values[i].append(float(f.evalf(subs={self._x: x})))

        return jnp.array(function_values), cell_numbers

    def _compute_functions_values(self, positions: Array, coeffs: Array) -> Array:
        # coeffs shape: ..., N, F, Y, where N is no. of cells, F is no. of basis functions, and Y is vector field dim.
        function_values, cell_numbers = self._compute_basis_functions_values(positions)
        cell_numbers = jnp.expand_dims(jnp.expand_dims(cell_numbers, axis=-1), axis=-1)
        cell_numbers = jnp.broadcast_to(
            cell_numbers, coeffs.shape[:-3] + cell_numbers.shape
        )
        coeffs = jnp.expand_dims(coeffs, axis=-4)
        coeffs_for_positions = jnp.take_along_axis(
            coeffs,
            cell_numbers,
            axis=-3,
        ).squeeze(axis=-3)
        return jnp.sum(
            coeffs_for_positions * jnp.expand_dims(function_values, axis=-1),
            axis=-2,  # corresponds to sum over basis functions
        )

    def _compute_stiffness_matrix(self) -> Array:
        stiffness_matrix = []
        derivatives_basis_function = [
            diff(f, self._x) for f in self._orthonormal_spatial_functions
        ]

        for i in range(self._num_spatial_functions):
            stiffness_matrix.append([])

            for j in range(self._num_spatial_functions):
                stiffness_matrix[i].append(
                    float(
                        integrate(
                            self._orthonormal_spatial_functions[i]
                            * derivatives_basis_function[j],
                            (
                                self._x,
                                -self._spatial_step_size / 2,
                                +self._spatial_step_size / 2,
                            ),
                        )
                    )
                )

        return jnp.array(stiffness_matrix)

    @staticmethod
    def _vector_field(
        t: float,
        coefficients: Array,
        args: tuple[
            Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
        ],
    ) -> Array:
        (
            y_x0,
            y_x1,
            reflection_coeffs_left_to_left,
            reflection_coeffs_right_to_right,
            transmission_coeffs_left_to_right,
            transmission_coeffs_right_to_left,
            spatial_derivative_operator,
            no_derivative_operator,
            stiffness_matrix,
            basis_functions_cell_boundary_values,
        ) = args

        intra_cell_vector_field = (
            jnp.einsum(
                "...ij,...ab,...jb->...ia",  # TODO: check whether stiffness_matrix and spatial_derivative_operator are broadcasted when not depending on cell index n
                stiffness_matrix,
                spatial_derivative_operator,
                coefficients,
            )
            + jnp.einsum(
                "...ab,...ib->...ia",  # TODO: check whether no_derivative_operator is broadcasted when not depending on cell index n
                no_derivative_operator,
                coefficients,
            )
        )

        # to compute fluxes:
        # 1. compute y_left and y_right at each boundary
        # 2. compute flux at each boundary
        # 3. add (subtract) flux at right (left) boundary of each cell
        y_left = jnp.einsum(
            "i,...ia->...ia",
            basis_functions_cell_boundary_values[:, 1],
            coefficients,
        )
        y_right = jnp.einsum(
            "i,...ia->...ia",
            basis_functions_cell_boundary_values[:, 0],
            coefficients,
        )
        y_left = jnp.concatenate(
            (jnp.expand_dims(y_x0, axis=-3), y_left), axis=-3
        )  # axis of cell index
        y_right = jnp.concatenate(
            (y_right, jnp.expand_dims(y_x1, axis=-3)), axis=-3
        )  # axis of cell index

        cell_boundary_values = RiemannSolver.compute_cell_boundary_values(
            spatial_derivative_operator,
            y_left,
            y_right,
        )

        cell_left_boundary_fluxes = jnp.einsum(
            "i,...ab,...ib->...ia",  # TODO: check whether spatial_derivative_operator is broadcasted when not depending on cell index n
            basis_functions_cell_boundary_values[:, 0],
            spatial_derivative_operator,
            cell_boundary_values,
        )
        cell_right_boundary_fluxes = jnp.einsum(
            "i,...ab,...ib->...ia",  # TODO: check whether spatial_derivative_operator is broadcasted when not depending on cell index n
            basis_functions_cell_boundary_values[:, 1],
            spatial_derivative_operator,
            cell_boundary_values,
        )

        inter_cell_vector_field = (
            -cell_left_boundary_fluxes[..., 1:, :, :]
            + cell_right_boundary_fluxes[..., :-1, :, :]
        )

        # TODO: apply corrections above also to computations below
        if (
            reflection_coeffs_right_to_right is not None
            and transmission_coeffs_left_to_right is not None
        ):
            domain_left_boundary_values = RiemannSolver.compute_domain_boundary_values(
                spatial_derivative_operator,
                y_left[..., 0, :, :],
                y_right[..., 0, :, :],
                reflection_coeffs_left_to_left=None,
                reflection_coeffs_right_to_right=reflection_coeffs_right_to_right,
                transmission_coeffs_left_to_right=transmission_coeffs_left_to_right,
                transmission_coeffs_right_to_left=None,
            )
        else:
            domain_left_boundary_values = jnp.zeros_like(
                inter_cell_vector_field[..., 0, :, :]
            )

        if (
            reflection_coeffs_left_to_left is not None
            and transmission_coeffs_right_to_left is not None
        ):
            domain_right_boundary_values = RiemannSolver.compute_domain_boundary_values(
                spatial_derivative_operator,
                y_left[..., -1, :, :],
                y_right[..., -1, :, :],
                reflection_coeffs_left_to_left=reflection_coeffs_left_to_left,
                reflection_coeffs_right_to_right=None,
                transmission_coeffs_left_to_right=None,
                transmission_coeffs_right_to_left=transmission_coeffs_right_to_left,
            )
        else:
            domain_right_boundary_values = jnp.zeros_like(
                inter_cell_vector_field[..., -1, :, :]
            )

        domain_left_boundary_fluxes = jnp.einsum(
            "i,...ab,...ib->...ia",  # TODO: check whether spatial_derivative_operator is broadcasted when not depending on cell index n
            basis_functions_cell_boundary_values[:, 0],
            spatial_derivative_operator,
            domain_left_boundary_values,
        )
        domain_right_boundary_fluxes = jnp.einsum(
            "i,...ab,...ib->...ia",  # TODO: check whether spatial_derivative_operator is broadcasted when not depending on cell index n
            basis_functions_cell_boundary_values[:, 1],
            spatial_derivative_operator,
            domain_right_boundary_values,
        )

        domain_vector_field = jnp.zeros_like(inter_cell_vector_field)
        domain_vector_field = domain_vector_field.at[..., 0, :, :].set(
            domain_left_boundary_fluxes
        )
        domain_vector_field = domain_vector_field.at[..., -1, :, :].set(
            -domain_right_boundary_fluxes
        )

        return intra_cell_vector_field + inter_cell_vector_field + domain_vector_field

    def compute_coeffs_from_initial_conditions_functions(
        self,
        yt0: list[ExprOfX],
        *,
        check_whether_representable: bool = False,
    ) -> Array:
        """Compute the per-cell basis coefficients of the initial conditions.

        Each component of ``yt0`` is projected onto the orthonormal spatial
        functions cell-by-cell by SymPy integration.

        Args:
            yt0: Initial conditions as SymPy expressions of the single
                position variable ``x``, one per component of the unknown
                vector field.
            check_whether_representable: Whether to check exact
                representability of the initial conditions by the basis
                functions via Parseval's identity; a UserWarning is emitted
                if the check fails.

        Returns:
            Per-cell basis coefficients with shape
            ``(num_spatial_cells, num_spatial_functions, num_components)``.

        Raises:
            ValueError: If a component of ``yt0`` depends on a variable
                other than ``x``.
        """
        for yt0_component in yt0:
            if len(yt0_component.free_symbols) > 1:
                raise ValueError(
                    f"function {yt0_component=} in orthonormal_spatial_functions depends on more than 1 free variable; ambiguity cannot be treated"
                )

        yt0 = [
            yt0_component.subs(yt0_component.free_symbols.pop(), self._x)
            if yt0_component.free_symbols
            else yt0_component
            for yt0_component in yt0
        ]
        x0, _x1 = self._spatial_domain_boundaries
        delta_x = self._spatial_step_size
        coeffs = []
        # TODO: continue reworking from here on
        for n in range(self._num_spatial_cells):
            coeffs.append([])
            # TODO: use self.spatial_cell_boundaries and self.spatial_cell_middles instead
            cell_start = x0 + n * delta_x
            cell_middle = cell_start + delta_x / 2
            cell_end = cell_middle + delta_x / 2

            for i, f_i in enumerate(self._orthonormal_spatial_functions):
                coeffs[n].append([])

                for y_component in yt0:
                    coeffs[n][i].append(
                        float(
                            integrate(
                                f_i.subs(self._x, self._x - cell_middle) * y_component,
                                (self._x, cell_start, cell_end),
                            )
                        )
                    )

        coeffs = jnp.array(coeffs)

        if check_whether_representable:
            # input function is exactly representable by linear combination of self.orthonormal_spatial_functions
            # => Parseval's identity holds -> check that
            sum_coeffs_squared = jnp.sum(coeffs**2, axis=1)
            yt0_squared = []

            for n in range(self._num_spatial_cells):
                yt0_squared.append([])
                # TODO: use self.spatial_cell_boundaries instead
                a = x0 + n * delta_x
                b = a + delta_x

                for y_component in yt0:
                    yt0_squared[n].append(
                        float(integrate(y_component**2, (self._x, a, b)))
                    )

            if not jnp.allclose(sum_coeffs_squared, jnp.array(yt0_squared)):
                warn(
                    "the initial conditions are not exactly representable by the "
                    "orthonormal spatial functions; the projection is only approximate"
                )

        return coeffs

    def compute_coeffs_from_initial_conditions_array(
        self, yt0: Array, *, check_whether_representable: bool = False
    ) -> Array:
        """Compute the coefficients of initial conditions given as an array.

        Not yet implemented.

        Args:
            yt0: Initial conditions sampled on the spatial grid.
            check_whether_representable: Whether to check exact
                representability of the initial conditions by the basis
                functions.

        Returns:
            Per-cell basis coefficients of the initial conditions.

        Raises:
            NotImplementedError: Always, as this is not yet implemented.
        """
        raise NotImplementedError

    def partialdiffeqsolve(
        self,
        *,
        diffeq_solver: AbstractSolver,
        t0: Array,
        t1: Array,
        dt0: Array | None,
        yt0: Array | list[ExprOfX],
        yx0: Array | list[ExprOfT] | None,
        yx1: Array | list[ExprOfT] | None,
        reflection_coeffs_left_to_left: Array | None = None,
        reflection_coeffs_right_to_right: Array | None = None,
        transmission_coeffs_left_to_right: Array | None = None,
        transmission_coeffs_right_to_left: Array | None = None,
        spatial_derivative_operator: Array,
        no_derivative_operator: Array,
        saveat_t: SaveAt = _SAVEAT_T,
        saveat_x: SaveAtX | None = None,
        stepsize_controller: AbstractStepSizeController = _STEPSIZE_CONTROLLER,
        max_steps: int | None = None,
        progress_meter: AbstractProgressMeter = _PROGRESS_METER,
    ) -> tuple[Solution, Array]:
        """Solve the PDE in time, evaluating at the requested positions.

        Initial conditions may be given as SymPy expressions or coefficient
        arrays; time integration is delegated to diffrax.

        Args:
            diffeq_solver: diffrax solver used for the time integration.
            t0: Initial time.
            t1: Final time.
            dt0: Initial time step; None if the step size controller chooses
                the first step.
            yt0: Initial conditions, either as SymPy expressions of ``x``
                (one per component of the unknown vector field) or as a
                coefficient array.
            yx0: Exterior limit of the unknown vector field at the left
                boundary of the spatial domain, as an array or as SymPy
                expressions of ``t``.
            yx1: Exterior limit of the unknown vector field at the right
                boundary of the spatial domain, as an array or as SymPy
                expressions of ``t``.
            reflection_coeffs_left_to_left: Reflection coefficients of the
                right domain boundary; None for no reflection there.
            reflection_coeffs_right_to_right: Reflection coefficients of the
                left domain boundary; None for no reflection there.
            transmission_coeffs_left_to_right: Transmission coefficients of
                the left domain boundary; None for no transmission there.
            transmission_coeffs_right_to_left: Transmission coefficients of
                the right domain boundary; None for no transmission there.
            spatial_derivative_operator: Constant matrix multiplying the
                spatial derivative of the unknown vector field.
            no_derivative_operator: Constant matrix multiplying the unknown
                vector field.
            saveat_t: diffrax SaveAt object specifying the times at which
                the solution is stored.
            saveat_x: Positions at which the unknown vector field is
                evaluated; if None, self.spatial_cell_middles is used.
            stepsize_controller: diffrax step size controller.
            max_steps: Maximum number of steps of the time integration.
            progress_meter: diffrax progress meter.

        Returns:
            The diffrax solution of the time integration and the unknown
            vector field evaluated at the positions ``saveat_x``.

        Raises:
            TypeError: If ``yt0`` has an unsupported type.
        """
        if isinstance(yt0, Array):
            coeffs_t0 = self.compute_coeffs_from_initial_conditions_array(
                yt0, check_whether_representable=False
            )

        elif isinstance(yt0, list) and all(
            isinstance(yt0_component, Expr) for yt0_component in yt0
        ):
            coeffs_t0 = self.compute_coeffs_from_initial_conditions_functions(
                yt0, check_whether_representable=False
            )

        else:
            raise TypeError(
                "isinstance(yt0, Array) or (isinstance(yt0, list) and all(isinstance(yt0_component, Expr) for yt0_component in yt0)) is required (or is currently not implemented yet)"
            )

        if saveat_x is None:
            saveat_x = self.spatial_cell_middles

        stiffness_matrix = self._compute_stiffness_matrix()
        basis_functions_cell_boundary_values = (
            self._compute_basis_functions_cell_boundary_values()
        )

        solution = diffeqsolve(
            ODETerm(self._vector_field),
            diffeq_solver,
            t0,
            t1,
            dt0,
            coeffs_t0,
            args=(
                yx0,
                yx1,
                reflection_coeffs_left_to_left,
                reflection_coeffs_right_to_right,
                transmission_coeffs_left_to_right,
                transmission_coeffs_right_to_left,
                spatial_derivative_operator,
                no_derivative_operator,
                stiffness_matrix,
                basis_functions_cell_boundary_values,
            ),
            saveat=saveat_t,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            progress_meter=progress_meter,
        )

        return solution, self._compute_functions_values(saveat_x, solution.ys)
