"""Shared pytest fixtures."""

import pytest
from sympy import Symbol


@pytest.fixture
def _x():
    return Symbol("x")
