# Modeling Correlation in the FIRE Simulation

## 1. Overview

This document outlines the plan to enhance the FIRE simulation engine by introducing statistical
correlation between the returns of different asset classes and the rate of inflation.

Currently, the simulation models asset returns and inflation as independent random variables,
drawing a separate random number for each at every time step. This does not capture the real-world
tendency for these economic variables to move in relation to one another (e.g., stocks and bonds
often(but not always) being negatively correlated).

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
   user provided correlation matrix and the already configured annual sample mean and standard
   deviation for each asset and inflation. Convert them to monthly lognormal parameters.
   It will use these to construct a **covariance matrix** M = DCD, wher D = diag(monthly_sigma_log),
   and C is the correlation matrix.

3. **Multivariate Sampling**: Instead of drawing independent random numbers, the engine will:
   - draw the log of correlated return factors R from a normal distribution.
   - convert them to rates via exponentiations, i.e. rates = exp(R) - 1.

---

## 3. Configuration

The configuration will require the full, symmetric correlation matrix to be
specified in `config.toml`. This makes the relationships explicit and prevents accidental omissions.

### Example `config.toml` structure

```toml
# ==============================================================================
# 5. CORRELATION MATRIX 
# ============================================================================== 
# Correlation matrix for asset returns and inflation.
# The `assets` list must match keys from the [assets] tables, plus "inflation".
# The `matrix` must be square and correspond to the `assets` list order.
[correlation_matrix]
assets_order = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]

# Matrix provided by Google Gemini 2.5 pro, reliable? Ahahah
# matrix = [
   # Stk,   Bnd,   STR,   Fun,   R.Est, Infl
   [1.00, -0.30, 0.00, 0.45, 0.15, -0.20], # Stocks
   [-0.30, 1.00, 0.40, -0.10, 0.05, 0.10], # Bonds
   [0.00, 0.40, 1.00, -0.05, 0.00, 0.60],  # STR
   [0.45, -0.10, -0.05, 1.00, 0.25, 0.15], # Fun
   [0.15, 0.05, 0.00, 0.25, 1.00, 0.05],   # Real Estate
   [-0.20, 0.10, 0.60, 0.15, 0.05, 1.00],  # Inflation
]
```

### Validation

The simulation will perform several validation checks on the user-provided correlation matrix at

1. **Square**: The number of rows must equal the number of columns.
2. **Diagonal of Ones**: All elements on the main diagonal must be exactly `1.0`.
3. **Symmetric**: The element at `matrix[i][j]` must be equal to the element at `matrix[j][i]`.
4. **Element Range**: All elements must be between `-1.0` and `1.0`, inclusive.
5. **Positive Semi-Definite**: The matrix must be positive semi-definite. This is a critical
   property ensuring that the described correlations are internally consistent and possible. The
   check is performed by calculating the matrix's eigenvalues; all eigenvalues must be non-negative.
