# Modeling Correlation in the FIRE Simulation

## 1. Overview

This document outlines the plan to enhance the FIRE simulation engine by introducing statistical
correlation between the returns of different asset classes and the rate of inflation.

Currently, the simulation models asset returns and inflation as independent random variables,
drawing a separate random number for each at every time step. This does not capture the real-world
tendency for these economic variables to move in relation to one another (e.g., stocks and bonds
often being negatively correlated).

The proposed change will make the simulation more realistic by modeling these interdependencies,
leading to more robust and credible financial projections.

---

## 2. Proposed Implementation

The core of the implementation will be to shift from sampling independent random variables to
sampling from a **multivariate distribution** that respects a user-defined correlation structure.
This change will be almost entirely contained within the `_precompute_sequences` method.

### Key Steps

1. **User Configuration**: The user will provide a full, symmetric **correlation matrix** in the
   configuration file. This matrix will define the correlation coefficient for every pair of
   variables (e.g., Stocks vs. Bonds, Bonds vs. Inflation).

2. **Build Covariance Matrix**: At the start of a simulation run, the engine will take the
   user-provided correlation matrix and the already-configured standard deviations for each asset
   and inflation. It will use these to construct a **covariance matrix**.

3. **Multivariate Sampling with Cholesky Decomposition**: Instead of drawing independent random
   numbers, the engine will: a. Perform a **Cholesky decomposition** of the covariance matrix. b.
   For each month, generate a vector of _independent_ standard normal random numbers. c. Use the
   Cholesky factor to transform this vector into a new vector of _correlated_ random numbers that
   conform to the specified covariance structure. These numbers represent the logarithmic returns.

4. **Generate Final Sequences**: The correlated logarithmic returns will be scaled and exponentiated
   to produce the final monthly return and inflation sequences, which are then used by the rest of
   the simulation logic as before.

---

## 3. Configuration

To minimize user error, the configuration will require the full, symmetric correlation matrix to be
specified in `config.toml`. This makes the relationships explicit and prevents accidental omissions.

### Example `config.toml` structure

```toml
# In [market_assumptions]
[market_assumptions.correlation_matrix]
# The order of columns must be consistent for each row:
#             ["Stk", "Bond", "STR", "Fun", "R_e", "pi"]
stocks      = [1.00, -0.20,  0.00,  0.70,  0.60, -0.10]
bonds       = [-0.20,  1.00,  0.20, -0.10,  0.10, -0.30]
str         = [0.00,  0.20,  1.00,  0.00,  0.00,  0.10]
fun         = [0.70, -0.10,  0.00,  1.00,  0.50, -0.05]
real_estate = [0.60,  0.10,  0.00,  0.50,  1.00,  0.40]
inflation   = [-0.10, -0.30,  0.10, -0.05,  0.40,  1.00]
```

### Validation

The simulation will perform several validation checks on the user-provided correlation matrix at
startup to ensure it is mathematically valid. A valid correlation matrix must be:

1. **Square**: The number of rows must equal the number of columns.
2. **Diagonal of Ones**: All elements on the main diagonal must be exactly `1.0`.
3. **Symmetric**: The element at `matrix[i][j]` must be equal to the element at `matrix[j][i]`.
4. **Element Range**: All elements must be between `-1.0` and `1.0`, inclusive.
5. **Positive Semi-Definite**: The matrix must be positive semi-definite. This is a critical
   property ensuring that the described correlations are internally consistent and possible. The
   check is performed by calculating the matrix's eigenvalues; all eigenvalues must be non-negative.
   This validation is essential because the **Cholesky decomposition** step will fail if the matrix
   is not positive semi-definite.
