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
VALID_MATRIX_DATA = {
    "stocks": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "bonds": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "str": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "fun": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "real_estate": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "inflation": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


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
        "stocks": [1.0, 0.0, 0.0, 0.0, 0.0],
        "bonds": [0.0, 1.0, 0.0, 0.0, 0.0],
        "str": [0.0, 0.0, 1.0, 0.0, 0.0],
        "fun": [0.0, 0.0, 0.0, 1.0, 0.0],
        "real_estate": [0.0, 0.0, 0.0, 0.0, 1.0],
        "inflation": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    with pytest.raises(ValidationError, match="Correlation matrix must be square."):
        CorrelationMatrix(**invalid_data)


def test_invalid_ragged_matrix():
    """
    Tests that a ragged matrix fails validation.
    Due to the implementation, this is caught as a non-numeric value error.
    """
    invalid_data = VALID_MATRIX_DATA.copy()
    invalid_data["stocks"] = [1.0, 0.0, 0.0, 0.0, 0.0]  # Make one row shorter
    with pytest.raises(
        ValidationError, match="Correlation matrix contains non-numeric values."
    ):
        CorrelationMatrix(**invalid_data)


def test_invalid_range_matrix():
    """
    Tests that a matrix with elements outside [-1, 1] fails validation.
    """
    invalid_data = VALID_MATRIX_DATA.copy()
    invalid_data["stocks"] = [1.0, 1.5, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(
        ValidationError,
        match="All elements of the correlation matrix must be between -1 and 1.",
    ):
        CorrelationMatrix(**invalid_data)


def test_invalid_diagonal_matrix():
    """
    Tests that a matrix without 1s on the diagonal fails validation.
    """
    invalid_data = VALID_MATRIX_DATA.copy()
    invalid_data["stocks"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(
        ValidationError,
        match="All diagonal elements of the correlation matrix must be 1.",
    ):
        CorrelationMatrix(**invalid_data)


def test_asymmetric_matrix():
    """
    Tests that an asymmetric matrix fails validation.
    """
    invalid_data = VALID_MATRIX_DATA.copy()
    invalid_data["stocks"] = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    invalid_data["bonds"] = [0.4, 1.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValidationError, match="Correlation matrix must be symmetric."):
        CorrelationMatrix(**invalid_data)


def test_not_positive_semi_definite_matrix():
    """
    Tests that a matrix that is not positive semi-definite fails validation.
    """
    # This matrix is symmetric, has 1s on the diagonal, and all elements
    # are in [-1, 1], but it is not positive semi-definite.
    not_psd_data = {
        "stocks": [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
        "bonds": [0.9, 1.0, 0.9, 0.0, 0.0, 0.0],
        "str": [0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
        "fun": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "real_estate": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "inflation": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(
        ValidationError, match="Correlation matrix must be positive semi-definite"
    ):
        CorrelationMatrix(**not_psd_data)


# Boilerplate for market assumptions in TOML format.
# These values are required for MarketAssumptions to validate, but are not the focus of these tests.
TOML_MARKET_ASSUMPTIONS_BOILERPLATE = """
[market_assumptions]
stock_mu = 0.08
stock_sigma = 0.15
bond_mu = 0.03
bond_sigma = 0.05
str_mu = 0.01
str_sigma = 0.02
fun_mu = 0.05
fun_sigma = 0.10
real_estate_mu = 0.04
real_estate_sigma = 0.08
pi_mu = 0.02
pi_sigma = 0.03
"""


def test_load_valid_correlation_matrix_from_toml():
    """
    Tests that a valid correlation matrix can be loaded from a TOML file
    as part of the MarketAssumptions model.
    """
    valid_matrix_toml = """
[market_assumptions.correlation_matrix]
stocks = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bonds = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
str = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
fun = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
real_estate = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
inflation = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
"""
    config_content = TOML_MARKET_ASSUMPTIONS_BOILERPLATE + valid_matrix_toml
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
stocks = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
bonds = [0.4, 1.0, 0.0, 0.0, 0.0, 0.0]
str = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
fun = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
real_estate = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
inflation = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
"""
    config_content = TOML_MARKET_ASSUMPTIONS_BOILERPLATE + invalid_matrix_toml
    config_file = Path("tests/config/correlation_test.toml")
    config_file.write_text(config_content)

    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)

    with pytest.raises(ValidationError, match="Correlation matrix must be symmetric."):
        MarketAssumptions(**config_data["market_assumptions"])
