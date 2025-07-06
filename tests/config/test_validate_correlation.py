# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import tomllib
from pathlib import Path

from pydantic import ValidationError

from firestarter.config.config import MarketAssumptions
from firestarter.config.correlation_matrix import CorrelationMatrix

# A known valid correlation matrix (identity matrix)
VALID_ASSETS = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
VALID_MATRIX = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
VALID_MATRIX_DATA = {"assets": VALID_ASSETS, "matrix": VALID_MATRIX}


def test_valid_correlation_matrix():
    """
    Tests that a valid correlation matrix passes validation.
    """
    try:
        CorrelationMatrix(**VALID_MATRIX_DATA)
    except ValidationError as e:
        pytest.fail(f"Valid correlation matrix failed validation: {e}")


def test_invalid_non_square_matrix():
    """
    Tests that a non-square matrix (6x5) fails validation.
    """
    invalid_data = {
        "assets": VALID_ASSETS,
        "matrix": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    }
    with pytest.raises(ValidationError, match="Correlation matrix must be square."):
        CorrelationMatrix(**invalid_data)


def test_invalid_ragged_matrix():
    """
    Tests that a ragged matrix fails validation.
    """
    invalid_matrix = VALID_MATRIX.copy()
    invalid_matrix[0] = [1.0, 0.0, 0.0, 0.0, 0.0]  # Make one row shorter
    invalid_data = {"assets": VALID_ASSETS, "matrix": invalid_matrix}
    with pytest.raises(
        ValidationError,
        match="Correlation matrix contains non-numeric values or is ragged.",
    ):
        CorrelationMatrix(**invalid_data)


def test_invalid_range_matrix():
    """
    Tests that a matrix with elements outside [-1, 1] fails validation.
    """
    invalid_matrix = VALID_MATRIX.copy()
    invalid_matrix[0] = [1.0, 1.5, 0.0, 0.0, 0.0, 0.0]
    invalid_data = {"assets": VALID_ASSETS, "matrix": invalid_matrix}
    with pytest.raises(
        ValidationError,
        match="All elements of the correlation matrix must be between -1 and 1.",
    ):
        CorrelationMatrix(**invalid_data)


def test_invalid_diagonal_matrix():
    """
    Tests that a matrix without 1s on the diagonal fails validation.
    """
    invalid_matrix = VALID_MATRIX.copy()
    invalid_matrix[0] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    invalid_data = {"assets": VALID_ASSETS, "matrix": invalid_matrix}
    with pytest.raises(
        ValidationError,
        match="All diagonal elements of the correlation matrix must be 1.",
    ):
        CorrelationMatrix(**invalid_data)


def test_asymmetric_matrix():
    """
    Tests that an asymmetric matrix fails validation.
    """
    invalid_matrix = VALID_MATRIX.copy()
    invalid_matrix[0] = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    invalid_matrix[1] = [0.4, 1.0, 0.0, 0.0, 0.0, 0.0]
    invalid_data = {"assets": VALID_ASSETS, "matrix": invalid_matrix}
    with pytest.raises(ValidationError, match="Correlation matrix must be symmetric."):
        CorrelationMatrix(**invalid_data)


def test_not_positive_semi_definite_matrix():
    """
    Tests that a matrix that is not positive semi-definite fails validation.
    """
    # This matrix is symmetric, has 1s on the diagonal, and all elements
    # are in [-1, 1], but it is not positive semi-definite.
    not_psd_matrix = [
        [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
        [0.9, 1.0, 0.9, 0.0, 0.0, 0.0],
        [0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    not_psd_data = {"assets": VALID_ASSETS, "matrix": not_psd_matrix}
    with pytest.raises(
        ValidationError, match="Correlation matrix must be positive semi-definite"
    ):
        CorrelationMatrix(**not_psd_data)


# Boilerplate for assets in TOML format.
# These values are required for MarketAssumptions to validate, but are not the focus of these tests.
TOML_ASSETS_BOILERPLATE = """
[market_assumptions.assets.stocks]
mu = 0.08
sigma = 0.15
is_liquid = true
withdrawal_priority = 1
[market_assumptions.assets.bonds]
mu = 0.03
sigma = 0.05
is_liquid = true
withdrawal_priority = 2
[market_assumptions.assets.str]
mu = 0.01
sigma = 0.02
is_liquid = true
withdrawal_priority = 3
[market_assumptions.assets.fun]
mu = 0.05
sigma = 0.10
is_liquid = true
withdrawal_priority = 4
[market_assumptions.assets.real_estate]
mu = 0.04
sigma = 0.08
is_liquid = false
[market_assumptions.assets.inflation]
mu = 0.02
sigma = 0.03
is_liquid = false
"""


def test_load_valid_correlation_matrix_from_toml():
    """
    Tests that a valid correlation matrix can be loaded from a TOML file
    as part of the MarketAssumptions model.
    """
    valid_matrix_toml = """
[market_assumptions.correlation_matrix]
assets = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
matrix = [
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
"""
    config_content = TOML_ASSETS_BOILERPLATE + valid_matrix_toml
    config_file = Path("tests/config/correlation_test.toml")
    config_file.write_text(config_content)

    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)

    try:
        MarketAssumptions(**config_data["market_assumptions"])
    except ValidationError as e:
        pytest.fail(f"Loading valid config from TOML failed: {e}")


def test_load_invalid_correlation_matrix_from_toml():
    """
    Tests that loading an invalid correlation matrix from a TOML file fails.
    """
    invalid_matrix_toml = """
# Invalid matrix (asymmetric)
[market_assumptions.correlation_matrix]
assets = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
matrix = [
  [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
  [0.4, 1.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
"""
    config_content = TOML_ASSETS_BOILERPLATE + invalid_matrix_toml
    config_file = Path("tests/config/correlation_test.toml")
    config_file.write_text(config_content)

    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)

    with pytest.raises(ValidationError, match="Correlation matrix must be symmetric."):
        MarketAssumptions(**config_data["market_assumptions"])
