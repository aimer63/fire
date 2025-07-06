# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generates precomputed stochastic sequences for market returns and inflation.
"""

import numpy as np
from firestarter.config.config import MarketAssumptions
from firestarter.config.correlation_matrix import CorrelationMatrix


class SequenceGenerator:
    """
    Generates and holds all stochastic sequences for a simulation set.
    """

    def __init__(
        self,
        market_assumptions: MarketAssumptions,
        num_sequences: int,
        simulation_years: int,
        seed: int | None = None,
    ):
        self.market_assumptions = market_assumptions
        self.num_sequences = num_sequences
        self.num_steps = simulation_years * 12
        self.seed = seed
        self.seed = seed

        if market_assumptions.correlation_matrix:
            self.correlation_matrix = market_assumptions.correlation_matrix
        else:
            # Create an identity matrix if none is provided
            asset_names = list(market_assumptions.assets.keys())
            num_assets = len(asset_names)
            self.correlation_matrix = CorrelationMatrix(
                assets=asset_names,
                matrix=np.identity(num_assets).tolist(),
            )

        self.asset_and_inflation_order = self.correlation_matrix.assets
        self.correlated_monthly_returns = self._generate_correlated_sequences()

    def _generate_correlated_sequences(self) -> np.ndarray:
        """
        Generates correlated random sequences for all assets and inflation.

        Returns:
            A numpy array of shape (num_sequences, num_steps, num_assets)
            containing the correlated monthly arithmetic return rates.
        """
        rng = np.random.default_rng(self.seed)

        # --- 1. Extract Annual Arithmetic Parameters ---
        ma = self.market_assumptions
        mu_arith = np.array(
            [ma.assets[asset].mu for asset in self.asset_and_inflation_order]
        )
        sigma_arith = np.array(
            [ma.assets[asset].sigma for asset in self.asset_and_inflation_order]
        )
        corr_matrix = np.array(self.correlation_matrix.matrix)

        # --- 2. Convert to Monthly Log-Normal Parameters ---
        ex = 1.0 + mu_arith
        vx = sigma_arith**2
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_log_annual = np.log(ex) - 0.5 * np.log(1 + vx / ex**2)
            sigma_log_annual = np.sqrt(np.log(1 + vx / ex**2))
        mu_log_annual = np.nan_to_num(mu_log_annual)
        sigma_log_annual = np.nan_to_num(sigma_log_annual)
        monthly_mu_log = mu_log_annual / 12
        monthly_sigma_log = sigma_log_annual / np.sqrt(12)

        # --- 3. Construct Monthly Log-Normal Covariance Matrix ---
        D = np.diag(monthly_sigma_log)
        monthly_cov_log = D @ corr_matrix @ D

        # --- 4. Generate Correlated Log-Normal Returns ---
        log_of_correlated_return_factors = rng.multivariate_normal(
            mean=monthly_mu_log,
            cov=monthly_cov_log,
            size=(self.num_sequences, self.num_steps),
        )

        # --- 5. Convert to Arithmetic Returns ---
        correlated_return_rates = np.exp(log_of_correlated_return_factors) - 1.0
        return correlated_return_rates
