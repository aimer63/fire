#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import pytest


from pydantic import ValidationError

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
VALID_MATRIX_DATA = {"assets_order": VALID_ASSETS, "matrix": VALID_MATRIX}


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
        "assets_order": VALID_ASSETS,
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
    invalid_data = {"assets_order": VALID_ASSETS, "matrix": invalid_matrix}
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
    invalid_data = {"assets_order": VALID_ASSETS, "matrix": invalid_matrix}
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
    invalid_data = {"assets_order": VALID_ASSETS, "matrix": invalid_matrix}
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
    invalid_data = {"assets_order": VALID_ASSETS, "matrix": invalid_matrix}
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
    not_psd_data = {"assets_order": VALID_ASSETS, "matrix": not_psd_matrix}
    with pytest.raises(
        ValidationError, match="Correlation matrix must be positive semi-definite"
    ):
        CorrelationMatrix(**not_psd_data)
