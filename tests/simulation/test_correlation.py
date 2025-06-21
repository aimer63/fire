import pytest
import numpy as np

from firestarter.config.config import MarketAssumptions
from firestarter.config.correlation_matrix import CorrelationMatrix


@pytest.fixture
def market_assumptions_with_correlation() -> MarketAssumptions:
    """
    Provides a MarketAssumptions object that includes a non-identity
    correlation matrix for testing correlated asset movements.
    """
    correlation_matrix = CorrelationMatrix(
        # Corresponds to: ["Stk", "Bond", "STR", "Fun", "R_e", "pi"]
        stocks=[1.00, -0.20, 0.00, 0.70, 0.60, -0.10],
        bonds=[-0.20, 1.00, 0.20, -0.10, 0.10, -0.30],
        str=[0.00, 0.20, 1.00, 0.00, 0.00, 0.10],
        fun=[0.70, -0.10, 0.00, 1.00, 0.50, -0.05],
        real_estate=[0.60, 0.10, 0.00, 0.50, 1.00, 0.40],
        inflation=[-0.10, -0.30, 0.10, -0.05, 0.40, 1.00],
    )
    return MarketAssumptions(
        stock_mu=0.07,
        stock_sigma=0.15,
        bond_mu=0.03,
        bond_sigma=0.05,
        str_mu=0.01,
        str_sigma=0.01,
        fun_mu=0.10,
        fun_sigma=0.30,
        real_estate_mu=0.04,
        real_estate_sigma=0.10,
        pi_mu=0.02,
        pi_sigma=0.01,
        correlation_matrix=correlation_matrix,
    )


def test_correlated_sequence_generation(market_assumptions_with_correlation):
    """
    Tests the generation of correlated random sequences for asset returns and inflation.

    This test implements the core logic that will be moved into the main simulation engine.
    It performs the following steps:
    1.  Defines the number of sequences and time steps for the test.
    2.  Extracts arithmetic means, sigmas, and the correlation matrix from the fixture.
    3.  Converts the annual arithmetic parameters to monthly log-normal parameters.
    4.  Constructs a monthly log-normal covariance matrix using the sigmas and the
        correlation matrix.
    5.  Uses `np.random.multivariate_normal` to generate correlated log-normal returns.
    6.  Converts the generated log-normal returns back to arithmetic returns.
    7.  Calculates the empirical correlation matrix from the generated sequences.
    8.  Asserts that the empirical correlation matrix is statistically close to the
        input correlation matrix.
    """
    # --- 1. Setup ---
    num_sequences = 50_000  # Use a large number for statistical significance
    num_years = 30
    num_steps = num_years * 12
    np.random.seed(42)  # for reproducibility

    # --- 2. Extract Parameters ---
    ma = market_assumptions_with_correlation
    assets_and_inflation_order = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
    mu_arith = np.array(
        [ma.stock_mu, ma.bond_mu, ma.str_mu, ma.fun_mu, ma.real_estate_mu, ma.pi_mu]
    )
    sigma_arith = np.array(
        [
            ma.stock_sigma,
            ma.bond_sigma,
            ma.str_sigma,
            ma.fun_sigma,
            ma.real_estate_sigma,
            ma.pi_sigma,
        ]
    )
    corr_matrix = np.array(list(ma.correlation_matrix.model_dump().values()))

    # --- 3. Convert to Monthly Log-Normal Parameters ---
    # Convert annual arithmetic returns to annual log-normal parameters
    ex = 1.0 + mu_arith
    vx = sigma_arith**2
    # Avoid division by zero if ex is zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_log_annual = np.log(ex) - 0.5 * np.log(1 + vx / ex**2)
        sigma_log_annual = np.sqrt(np.log(1 + vx / ex**2))

    # Handle potential NaNs from the calculation if ex is zero or negative
    mu_log_annual = np.nan_to_num(mu_log_annual)
    sigma_log_annual = np.nan_to_num(sigma_log_annual)

    # Scale annual log-normal parameters to monthly
    monthly_mu_log = mu_log_annual / 12
    monthly_sigma_log = sigma_log_annual / np.sqrt(12)

    # --- 4. Construct Monthly Log-Normal Covariance Matrix ---
    # D = diagonal matrix of standard deviations
    # Cov = D * Corr * D
    D = np.diag(monthly_sigma_log)
    monthly_cov_log = D @ corr_matrix @ D

    # --- 5. Generate Correlated Log-Normal Returns ---
    # This is the core step: draw from a multivariate normal distribution.
    # The output represents the logarithm of correlated monthly return factors (log(1+r)).
    log_of_correlated_return_factors = np.random.multivariate_normal(
        mean=monthly_mu_log, cov=monthly_cov_log, size=(num_sequences, num_steps)
    )
    # Shape: (num_sequences, num_steps, num_assets)

    # --- 6. Convert to Arithmetic Returns ---
    # The simulation uses arithmetic return rates (e.g., 0.01 for 1%).
    # We exponentiate the log of the factors and subtract 1 to get the rates.
    correlated_return_rates = np.exp(log_of_correlated_return_factors) - 1.0

    # --- 7. Calculate Empirical Correlation ---
    # Reshape the data to (total_months, num_assets) to calculate correlation
    # across all months from all sequences
    reshaped_returns = correlated_return_rates.reshape(-1, len(assets_and_inflation_order))
    empirical_corr_matrix = np.corrcoef(reshaped_returns, rowvar=False)

    # --- 8. Assert Results ---
    # Check that the generated correlation matrix is close to the input matrix.
    # A relative tolerance (rtol) is used because this is a statistical process.
    # With 50k sequences, we expect a reasonably close match.
    np.testing.assert_allclose(
        empirical_corr_matrix,
        corr_matrix,
        rtol=0.1,
        atol=0.05,
        err_msg="Generated correlation matrix does not match the input matrix.",
    )


@pytest.fixture
def market_assumptions_with_identity_correlation() -> MarketAssumptions:
    """
    Provides a MarketAssumptions object with an identity correlation matrix,
    representing zero correlation between all assets and inflation.
    """
    identity_matrix = CorrelationMatrix(
        stocks=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        bonds=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        str=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        fun=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        real_estate=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        inflation=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
    return MarketAssumptions(
        stock_mu=0.07,
        stock_sigma=0.15,
        bond_mu=0.03,
        bond_sigma=0.05,
        str_mu=0.01,
        str_sigma=0.01,
        fun_mu=0.10,
        fun_sigma=0.30,
        real_estate_mu=0.04,
        real_estate_sigma=0.10,
        pi_mu=0.02,
        pi_sigma=0.01,
        correlation_matrix=identity_matrix,
    )
