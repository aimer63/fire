import numpy as np
from pydantic import BaseModel, model_validator


class CorrelationMatrix(BaseModel):
    """
    Pydantic model for a correlation matrix with built-in validation.

    This model ensures that the provided matrix is square, symmetric, has 1s on
    the diagonal, has all elements between -1 and 1, and is positive
    semi-definite.
    """

    stocks: list[float]
    bonds: list[float]
    str: list[float]
    fun: list[float]
    real_estate: list[float]
    inflation: list[float]

    @model_validator(mode="after")
    def validate_matrix(self) -> "CorrelationMatrix":
        """
        Validates the correlation matrix after the model is created.
        """
        matrix_data = [
            self.stocks,
            self.bonds,
            self.str,
            self.fun,
            self.real_estate,
            self.inflation,
        ]

        try:
            matrix = np.array(matrix_data, dtype=float)
        except ValueError:
            raise ValueError("Correlation matrix contains non-numeric values.")

        # 1. Check if square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")

        # 2. Check element range
        if not np.all((matrix >= -1) & (matrix <= 1)):
            raise ValueError("All elements of the correlation matrix must be between -1 and 1.")

        # 3. Check for 1s on the diagonal
        if not np.all(np.diag(matrix) == 1):
            raise ValueError("All diagonal elements of the correlation matrix must be 1.")

        # 4. Check for symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Correlation matrix must be symmetric.")

        # 5. Check for positive semi-definiteness
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
