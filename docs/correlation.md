# Modeling Correlation in the FIRE Simulation

## 1. Implementation

### Key Steps

1. **User Configuration**: The user will provide a full, symmetric positive semi-definite
   **correlation matrix** in the configuration file. This matrix will define the
   correlation coefficient for every pair of assets and inflation (e.g., stocks vs. bonds,
   bonds vs. inflation).

2. **Build Covariance Matrix**: At the start of a simulation run, the engine will take the
   user provided correlation matrix and the already configured annual sample mean and standard
   deviation for each asset and inflation and convert them to monthly lognormal parameters.
   It will use these to construct a **covariance matrix** M = DCD, wher D = diag(monthly_sigma_log),
   and C is the correlation matrix.

3. **Multivariate Sampling**: Instead of drawing independent random numbers, the engine will:
   - draw the log of correlated return factors R from a normal distribution.
   - convert them to rates via exponentiations, i.e. rates = exp(R) - 1.
   - to draw independent variables just provide the identity as the correlation matrix.

---

## 3. Configuration

The configuration will require the full, symmetric correlation matrix to be
specified in `config.toml`. This makes the relationships explicit and prevents
accidental omissions.

### Example `config.toml` structure

```toml
# ==============================================================================
# 5. CORRELATION MATRIX
# ==============================================================================
# Correlation matrix for asset returns and inflation.
# The `assets` list must match keys from the [assets] tables, plus "inflation".
# The `matrix` must be square and correspond to the `assets` list order.
[correlation_matrix]
assets_order = ["stocks", "bonds", "str", "gold", "inflation"]

# Matrix provided by Google Gemini 2.5 pro, reliable? Ahahah
matrix = [
   # stk,  bnd,  str,  Au,   infl
   [1.00, -0.30, 0.00, 0.15, -0.20], # stocks
   [-0.30, 1.00, 0.40, 0.05, 0.10],  # bonds
   [0.00,  0.40, 1.00, 0.00, 0.60],  # str (short term rate money market)
   [0.15,  0.05, 0.00, 1.00, 0.05],  # Au (gold)
   [-0.20, 0.10, 0.60, 0.05, 1.00],  # inflation
]
```

### Validation

The simulation will perform several validation checks on the user-provided correlation matrix:

1. **Square**: The number of rows must equal the number of columns.
2. **Diagonal of Ones**: All elements on the main diagonal must be exactly `1.0`.
3. **Symmetric**: The element at `Cij` must be equal to the element at `Cji`.
4. **Element Range**: All elements must be between `-1.0` and `1.0`, inclusive.
5. **Positive Semi-Definite**: The matrix must be positive semi-definite. This is a critical
   property ensuring that the described correlations are internally consistent and possible. The
   check is performed by calculating the matrix's eigenvalues; all eigenvalues must be non-negative.
