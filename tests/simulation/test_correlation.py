# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from firestarter.core.sequence_generator import SequenceGenerator


def test_correlated_sequence_generation(basic_assets, basic_correlation_matrix):
    """
    Tests the generation of correlated random sequences for asset returns and inflation.
    It uses the SequenceGenerator to create the sequences and then verifies that the
    empirical correlation matrix of the generated data is statistically close to the
    input correlation matrix.
    """
    num_sequences = 50_000
    num_years = 30

    generator = SequenceGenerator(
        assets=basic_assets,
        correlation_matrix=basic_correlation_matrix,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.correlated_monthly_returns

    reshaped_returns = correlated_return_rates.reshape(
        -1, len(basic_correlation_matrix.assets_order)
    )
    empirical_corr_matrix = np.corrcoef(reshaped_returns, rowvar=False)

    np.testing.assert_allclose(
        empirical_corr_matrix,
        basic_correlation_matrix.matrix,
        rtol=0.1,
        atol=0.05,
        err_msg="Generated correlation matrix does not match the input matrix.",
    )


def test_uncorrelated_sequence_generation(basic_assets, identity_correlation_matrix):
    """
    Tests that sequence generation with an identity correlation matrix produces
    statistically uncorrelated returns.
    """
    num_sequences = 50_000
    num_years = 30

    generator = SequenceGenerator(
        assets=basic_assets,
        correlation_matrix=identity_correlation_matrix,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.correlated_monthly_returns

    reshaped_returns = correlated_return_rates.reshape(
        -1, len(identity_correlation_matrix.assets_order)
    )
    empirical_corr_matrix = np.corrcoef(reshaped_returns, rowvar=False)

    np.testing.assert_allclose(
        empirical_corr_matrix,
        identity_correlation_matrix.matrix,
        rtol=0.1,
        atol=0.05,
        err_msg="Generated correlation matrix for uncorrelated assets is not close to identity.",
    )
