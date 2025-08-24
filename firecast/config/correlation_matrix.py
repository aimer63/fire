#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import numpy as np
from pydantic import BaseModel, model_validator


class CorrelationMatrix(BaseModel):
    """
    Pydantic model for a correlation matrix with built-in validation.

    This model ensures that the provided matrix is square, symmetric, has 1s on
    the diagonal, has all elements between -1 and 1, and is positive
    semi-definite. It also checks that the number of assets in the list
    matches the dimensions of the matrix.
    """

    assets_order: list[str]
    matrix: list[list[float]]

    @model_validator(mode="after")
    def validate_matrix(self) -> "CorrelationMatrix":
        """
        Validates the correlation matrix after the model is created.
        """
        # 1. Check that number of assets matches matrix dimensions
        if len(self.assets_order) != len(self.matrix):
            raise ValueError(
                "Number of assets must match the number of rows in the correlation matrix."
            )

        try:
            matrix = np.array(self.matrix, dtype=float)
        except ValueError:
            # This can happen if the matrix is ragged (rows of different lengths)
            raise ValueError(
                "Correlation matrix contains non-numeric values or is ragged."
            )

        # 2. Check if square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")

        # 3. Check element range
        if not np.all((matrix >= -1) & (matrix <= 1)):
            raise ValueError(
                "All elements of the correlation matrix must be between -1 and 1."
            )

        # 4. Check for 1s on the diagonal
        if not np.all(np.diag(matrix) == 1):
            raise ValueError(
                "All diagonal elements of the correlation matrix must be 1."
            )

        # 5. Check for symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Correlation matrix must be symmetric.")

        # 6. Check for positive semi-definiteness
        # Use eigvalsh as it's optimized for symmetric matrices and avoids complex numbers
        eigenvalues = np.linalg.eigvalsh(matrix)
        # Use a small tolerance for floating point inaccuracies
        if not np.all(eigenvalues >= -1e-8):
            raise ValueError(
                (
                    "Correlation matrix must be positive semi-definite "
                    + "(all eigenvalues must be non-negative)."
                )
            )

        return self

    def to_numpy(self) -> np.ndarray:
        """Converts the correlation matrix to a NumPy array."""
        return np.array(self.matrix, dtype=float)
