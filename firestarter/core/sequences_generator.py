#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
This module provides the SequenceGenerator class, which generates precomputed stochastic
sequences for asset returns and inflation used in FIRE Monte Carlo simulations.

Key features:
- Supports correlated random sampling of asset returns and inflation using a user-supplied
  correlation matrix.
- Converts annual arithmetic mean and standard deviation parameters to monthly log-normal
  parameters for realistic simulation.
- Handles both user-supplied and default (identity) correlation matrices.
- Produces reproducible results with optional random seed control.
- Outputs sequences as numpy arrays for efficient downstream simulation.

Typical usage:
    generator = SequenceGenerator(
        assets=assets_dict,
        correlation_matrix=correlation_matrix,
        num_sequences=1000,
        simulation_years=40,
        seed=42,
    )
    monthly_return_rates = generator.monthly_return_rates

Classes:
    - SequenceGenerator: Generates and holds all stochastic sequences for a simulation set.
"""

import numpy as np
from firestarter.config.config import Asset
from firestarter.config.correlation_matrix import CorrelationMatrix


class SequencesGenerator:
    """
    Generates and holds all stochastic sequences for a simulation set.
    """

    def __init__(
        self,
        assets: dict[str, "Asset"],
        correlation_matrix: CorrelationMatrix,
        num_sequences: int,
        simulation_years: int,
        seed: int | None = None,
    ):
        self.assets = assets
        self.correlation_matrix = correlation_matrix
        self.num_sequences = num_sequences
        self.num_steps = simulation_years * 12
        self.seed = seed

        if self.correlation_matrix:
            self.asset_and_inflation_order = self.correlation_matrix.assets_order
        else:
            # Create an identity matrix if none is provided
            asset_names = list(self.assets.keys())
            num_assets = len(asset_names)
            self.correlation_matrix = CorrelationMatrix(
                assets_order=asset_names,
                matrix=np.identity(num_assets).tolist(),
            )
            self.asset_and_inflation_order = asset_names

        self.monthly_return_rates = self._generate_sequences()

    def _generate_sequences(self) -> np.ndarray:
        """
        Generates correlated random sequences for all assets and inflation.

        Returns:
            A numpy array of shape (num_sequences, num_steps, num_assets)
            containing the correlated monthly arithmetic return rates.
        """
        rng = np.random.default_rng(self.seed)

        # --- 1. Extract Annual Arithmetic Parameters ---
        mu_arith = np.array([self.assets[asset].mu for asset in self.asset_and_inflation_order])
        sigma_arith = np.array(
            [self.assets[asset].sigma for asset in self.asset_and_inflation_order]
        )
        corr_matrix = np.array(self.correlation_matrix.matrix)

        # --- 2. Convert to Monthly Log-Normal Parameters ---
        ex = 1.0 + mu_arith
        vx = sigma_arith**2
        mu_log_annual = np.log(ex) - 0.5 * np.log(1 + vx / ex**2)
        sigma_log_annual = np.sqrt(np.log(1 + vx / ex**2))
        monthly_mu_log = mu_log_annual / 12
        monthly_sigma_log = sigma_log_annual / np.sqrt(12)

        # --- 3. Construct Monthly Log-Normal Covariance Matrix ---
        D = np.diag(monthly_sigma_log)
        monthly_cov_log = D @ corr_matrix @ D

        # --- 4. Generate Correlated Log-Normal Returns ---
        log_of_return_factors = rng.multivariate_normal(
            mean=monthly_mu_log,
            cov=monthly_cov_log,
            size=(self.num_sequences, self.num_steps),
        )

        # --- 5. Convert to Arithmetic Returns ---
        return_rates = np.exp(log_of_return_factors) - 1.0
        return return_rates
