"""Pydantic models for configurations describing complete numerical experiments."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from sympy import Expr

from dg_solvax.utils.utils import (
    _validate_is_at_least_2,
    _validate_is_positive,
)


class SpatialDomainConfig(BaseModel):
    """Spatial domain: boundaries and cell size."""

    spatial_cell_size: Annotated[float, AfterValidator(_validate_is_positive)]
    spatial_domain_boundaries: tuple[float, float]

    @model_validator(mode="after")
    def _validate_boundaries(self):
        if self.spatial_domain_boundaries[1] <= self.spatial_domain_boundaries[0]:
            raise ValueError(
                "spatial_domain_boundaries[1] must be > spatial_domain_boundaries[0]"
            )
        return self


class TimeIntegrationConfig(BaseModel):
    """Time integration settings: interval, step size and sampling points."""

    t0: float
    t1: float
    dt0: Annotated[float, AfterValidator(_validate_is_positive), "step size"]
    num_time_points: Annotated[int, AfterValidator(_validate_is_at_least_2)]
    num_space_points: Annotated[int, AfterValidator(_validate_is_at_least_2)]

    @model_validator(mode="after")
    def _validate_t1_gt_t0(self):
        if self.t1 <= self.t0:
            raise ValueError("t1 must be > t0")
        return self


class BoundaryCoefficientsConfig(BaseModel):
    """Reflection and transmission coefficients at the domain boundaries."""

    reflection_coeffs_left_to_left: list[list[float]] | None = None
    reflection_coeffs_right_to_right: list[list[float]] | None = None
    transmission_coeffs_left_to_right: list[list[float]] | None = None
    transmission_coeffs_right_to_left: list[list[float]] | None = None


class VisualizationConfig(BaseModel):
    """Labels and title used for plotting."""

    component_labels: list[str]
    title: str


class ProblemConfigBase(BaseModel):
    """Common fields of all problem configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_field_values_time_0: list[Expr]
    vector_field_values_left_boundary: list[list[float]] | None = None
    vector_field_values_right_boundary: list[list[float]] | None = None
