"""Pydantic models of systems of partial differential equations (PDEs) and corresponding PDE problems."""

from typing import Annotated, Any

from jax import Array
from jax import numpy as jnp
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from sympy import Expr

from dg_solvax.utils.utils import _validate_is_square_matrix


def _optional_jnp_array(values: list[list[float]] | None) -> Array | None:
    return jnp.array(values) if values is not None else None


class SystemPartialDiffEqs(BaseModel):
    """A system of PDEs with constant derivative and no-derivative operators."""

    spatial_derivative_operator: Annotated[
        list[list[float]],
        AfterValidator(_validate_is_square_matrix),
        "Matrix with which the spatial derivative of the unknown vector field is to be multiplied (constant in space and time)",
    ]
    no_derivative_operator: Annotated[
        list[list[float]],
        AfterValidator(_validate_is_square_matrix),
        "Matrix with which the unknown vector field is to be multiplied (constant in space and time)",
    ]

    @model_validator(mode="after")
    def _validate_are_dimensions_compatible(self):
        if len(self.spatial_derivative_operator) != len(self.no_derivative_operator):
            raise ValueError(
                f"{len(self.spatial_derivative_operator)=} == {len(self.no_derivative_operator)} is required"
            )

        return self


class SystemPartialDiffEqsProblem(BaseModel):
    """A PDE problem: the system with initial and boundary values."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    system_partial_diff_eqs: SystemPartialDiffEqs
    vector_field_values_time_0: Annotated[
        list[Expr],
        "Initial conditions as sympy expressions of the position variable "
        "``x``, one per component of the unknown vector field",
    ]
    vector_field_values_left_boundary: Annotated[
        list[list[float]] | None,
        "Exterior values of the unknown vector field at the left domain "
        "boundary, one inner list per spatial basis function",
    ]
    vector_field_values_right_boundary: Annotated[
        list[list[float]] | None,
        "Exterior values of the unknown vector field at the right domain "
        "boundary, one inner list per spatial basis function",
    ]
    reflection_coeffs_left_to_left: Annotated[
        list[list[float]] | None,
        "Reflection coefficients at the right domain boundary; None for no "
        "reflection there",
    ] = None
    reflection_coeffs_right_to_right: Annotated[
        list[list[float]] | None,
        "Reflection coefficients at the left domain boundary; None for no "
        "reflection there",
    ] = None
    transmission_coeffs_left_to_right: Annotated[
        list[list[float]] | None,
        "Transmission coefficients at the left domain boundary; None for no "
        "transmission there",
    ] = None
    transmission_coeffs_right_to_left: Annotated[
        list[list[float]] | None,
        "Transmission coefficients at the right domain boundary; None for no "
        "transmission there",
    ] = None

    @model_validator(mode="after")
    def _validate_are_dimensions_compatible(self):
        num_components = len(self.system_partial_diff_eqs.no_derivative_operator)

        if len(self.vector_field_values_time_0) != num_components:
            raise ValueError(
                f"{len(self.vector_field_values_time_0)=} == {num_components} is required"
            )

        for values in (
            self.vector_field_values_left_boundary,
            self.vector_field_values_right_boundary,
        ):
            if values is not None and any(len(row) != num_components for row in values):
                raise ValueError(
                    f"each inner list of the boundary values must have {num_components} entries"
                )

        return self

    def provide_partialdiffeqsolve_args(self) -> dict[str, Any]:
        """Return the keyword arguments for ``partialdiffeqsolve`` as a dict.

        Returns:
            Keyword arguments ready to be unpacked into
            ``partialdiffeqsolve``.
        """
        return {
            "yt0": self.vector_field_values_time_0,
            "spatial_derivative_operator": jnp.array(
                self.system_partial_diff_eqs.spatial_derivative_operator
            ),
            "no_derivative_operator": jnp.array(
                self.system_partial_diff_eqs.no_derivative_operator
            ),
            "yx0": _optional_jnp_array(self.vector_field_values_left_boundary),
            "yx1": _optional_jnp_array(self.vector_field_values_right_boundary),
            "reflection_coeffs_left_to_left": _optional_jnp_array(
                self.reflection_coeffs_left_to_left
            ),
            "reflection_coeffs_right_to_right": _optional_jnp_array(
                self.reflection_coeffs_right_to_right
            ),
            "transmission_coeffs_left_to_right": _optional_jnp_array(
                self.transmission_coeffs_left_to_right
            ),
            "transmission_coeffs_right_to_left": _optional_jnp_array(
                self.transmission_coeffs_right_to_left
            ),
        }

    def print_partialdiffeqsolve_args(self):
        """Print the partialdiffeqsolve arguments.

        Prints each of the keyword arguments returned by
        ``provide_partialdiffeqsolve_args``.
        """
        for variable, value in self.provide_partialdiffeqsolve_args().items():
            print(f"{variable}:\n{value}")
