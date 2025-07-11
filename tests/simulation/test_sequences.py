#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
import numpy as np

from firestarter.core.sequence_generator import SequenceGenerator


def test_correlated_sequence_generation(basic_assets, basic_correlation_matrix):
    """
    Tests the generation of correlated random sequences for asset returns and inflation.
    It uses the SequenceGenerator to create the sequences and then verifies that the
    empirical correlation matrix of the generated data is statistically close to the
    input correlation matrix.
    """
    num_sequences = 10_000
    num_years = 30

    generator = SequenceGenerator(
        assets=basic_assets,
        correlation_matrix=basic_correlation_matrix,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.monthly_return_rates

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


def test_indipendent_sequence_generation(basic_assets, identity_correlation_matrix):
    """
    Tests that sequence generation with an identity correlation matrix produces
    statistically uncorrelated returns.
    """
    num_sequences = 10_000
    num_years = 30

    generator = SequenceGenerator(
        assets=basic_assets,
        correlation_matrix=identity_correlation_matrix,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=42,
    )
    correlated_return_rates = generator.monthly_return_rates

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


def test_sequence_mean_and_std_match_asset_params(
    basic_assets, identity_correlation_matrix
):
    """
    Verifies that the sample mean and standard deviation of generated sequences
    are close to the configured asset parameters (mu, sigma).
    """
    num_sequences = 10_000
    num_years = 30

    generator = SequenceGenerator(
        assets=basic_assets,
        correlation_matrix=identity_correlation_matrix,
        num_sequences=num_sequences,
        simulation_years=num_years,
        seed=123,
    )
    correlated_return_rates = (
        generator.monthly_return_rates
    )  # shape: (num_sequences, months, assets)
    reshaped_returns = correlated_return_rates.reshape(
        -1, len(identity_correlation_matrix.assets_order)
    )

    for i, asset_name in enumerate(identity_correlation_matrix.assets_order):
        asset = basic_assets[asset_name]
        # Use exact formulas for annualizing mean and std of returns
        sample_mean = (
            np.log(reshaped_returns[:, i] + 1).mean()
        ) * 12  # log-returns annualized mean
        sample_std = (np.log(reshaped_returns[:, i] + 1).std(ddof=1)) * np.sqrt(
            12
        )  # log-returns annualized std
        # Convert back to arithmetic mean and std for comparison
        annualized_arith_mean = np.exp(sample_mean + 0.5 * sample_std**2) - 1
        annualized_arith_std = np.sqrt(
            (np.exp(sample_std**2) - 1) * np.exp(2 * sample_mean + sample_std**2)
        )
        assert annualized_arith_mean == pytest.approx(asset.mu, rel=0.05)
        assert annualized_arith_std == pytest.approx(asset.sigma, rel=0.05)
