# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np

from firestarter.config.config import MarketAssumptions
from firestarter.config.correlation_matrix import CorrelationMatrix
from firestarter.core.sequence_generator import SequenceGenerator


@pytest.fixture
def market_assumptions_with_correlation() -> MarketAssumptions:
    """
    Provides a MarketAssumptions object that includes a non-identity
    correlation matrix for testing correlated asset movements.
    """
    from firestarter.config.config import Asset

    assets = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
    matrix = [
        [1.00, -0.20, 0.00, 0.70, 0.60, -0.10],
        [-0.20, 1.00, 0.20, -0.10, 0.10, -0.30],
        [0.00, 0.20, 1.00, 0.00, 0.00, 0.10],
        [0.70, -0.10, 0.00, 1.00, 0.50, -0.05],
        [0.60, 0.10, 0.00, 0.50, 1.00, 0.40],
        [-0.10, -0.30, 0.10, -0.05, 0.40, 1.00],
    ]
    correlation_matrix = CorrelationMatrix(assets=assets, matrix=matrix)
    return MarketAssumptions(
        assets={
            "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=2),
            "bonds": Asset(mu=0.03, sigma=0.05, is_liquid=True, withdrawal_priority=1),
            "str": Asset(mu=0.01, sigma=0.01, is_liquid=True, withdrawal_priority=0),
            "fun": Asset(mu=0.10, sigma=0.30, is_liquid=True, withdrawal_priority=3),
            "real_estate": Asset(mu=0.04, sigma=0.10, is_liquid=False),
            "inflation": Asset(mu=0.02, sigma=0.01, is_liquid=False),
        },
        correlation_matrix=correlation_matrix,
    )


def test_correlated_sequence_generation(market_assumptions_with_correlation):
    """
    Tests the generation of correlated random sequences for asset returns and inflation.
    It uses the SequenceGenerator to create the sequences and then verifies that the
    empirical correlation matrix of the generated data is statistically close to the
    input correlation matrix.
    """
    # --- 1. Setup ---
    num_sequences = 50_000
    num_years = 30
    ma = market_assumptions_with_correlation

    # --- 2. Generate Sequences ---
    generator = SequenceGenerator(
        market_assumptions=ma,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.correlated_monthly_returns

    # --- 3. Calculate Empirical Correlation ---
    # Reshape the data to (total_months, num_assets) to calculate correlation
    # across all months from all sequences
    reshaped_returns = correlated_return_rates.reshape(
        -1, len(ma.correlation_matrix.assets)
    )
    empirical_corr_matrix = np.corrcoef(reshaped_returns, rowvar=False)

    # --- 4. Assert Results ---
    # Check that the generated correlation matrix is close to the input matrix.
    # A relative tolerance (rtol) is used because this is a statistical process.
    # With 50k sequences, we expect a reasonably close match.
    np.testing.assert_allclose(
        empirical_corr_matrix,
        ma.correlation_matrix.matrix,
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
    from firestarter.config.config import Asset

    assets = ["stocks", "bonds", "str", "fun", "real_estate", "inflation"]
    matrix = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    identity_matrix = CorrelationMatrix(assets=assets, matrix=matrix)
    return MarketAssumptions(
        assets={
            "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=2),
            "bonds": Asset(mu=0.03, sigma=0.05, is_liquid=True, withdrawal_priority=1),
            "str": Asset(mu=0.01, sigma=0.01, is_liquid=True, withdrawal_priority=0),
            "fun": Asset(mu=0.10, sigma=0.30, is_liquid=True, withdrawal_priority=3),
            "real_estate": Asset(mu=0.04, sigma=0.10, is_liquid=False),
            "inflation": Asset(mu=0.02, sigma=0.01, is_liquid=False),
        },
        correlation_matrix=identity_matrix,
    )


def test_uncorrelated_sequence_generation(
    market_assumptions_with_identity_correlation,
):
    """
    Tests that sequence generation with an identity correlation matrix produces
    statistically uncorrelated returns.
    """
    # --- 1. Setup ---
    num_sequences = 50_000
    num_years = 30
    ma = market_assumptions_with_identity_correlation

    # --- 2. Generate Sequences ---
    generator = SequenceGenerator(
        market_assumptions=ma,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.correlated_monthly_returns

    # --- 3. Calculate Empirical Correlation ---
    reshaped_returns = correlated_return_rates.reshape(
        -1, len(ma.correlation_matrix.assets)
    )
    empirical_corr_matrix = np.corrcoef(reshaped_returns, rowvar=False)

    # --- 4. Assert Results ---
    # With an identity matrix, off-diagonal correlations should be close to 0.
    np.testing.assert_allclose(
        empirical_corr_matrix,
        ma.correlation_matrix.matrix,
        rtol=0.1,
        atol=0.05,
        err_msg="Generated correlation matrix for uncorrelated assets is not close to identity.",
    )
