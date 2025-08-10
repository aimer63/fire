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
- Converts annual sample mean and standard deviation parameters to monthly log-normal
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

# OU model parameters (fixed for all assets)
THETA = 0.5  # Mean reversion speed
MU = 0.08  # Long-term mean
SIGMA = 0.15  # Volatility
INFLATION = 0.02  # Constant monthly inflation rate


class SequencesGenerator:
    """
    Generates and holds all stochastic sequences for a simulation set.

    Args:
        model (str): Model for sequence generation ("lognormal" or "OU").
    """

    def __init__(
        self,
        assets: dict[str, "Asset"],
        correlation_matrix: CorrelationMatrix,
        num_sequences: int,
        simulation_years: int,
        seed: int | None = None,
        model: str = "lognormal",
    ):
        self.assets = assets
        self.correlation_matrix = correlation_matrix
        self.num_sequences = num_sequences
        self.num_steps = simulation_years * 12
        self.seed = seed
        self.model = model

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

        if self.model == "lognormal":
            self.monthly_return_rates: np.ndarray = self._generate_sequences_lognormal()
        elif self.model == "OU":
            self.monthly_return_rates: np.ndarray = self._generate_sequences_ou()
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _generate_sequences_lognormal(self) -> np.ndarray:
        """Generates correlated monthly return rates for assets and inflation.

        This method implements a standard financial modeling technique to generate
        stochastic return sequences. The process ensures that the generated monthly
        return rates (R) result in return factors (1 + R) that are log-normally
        distributed, which is a common assumption for asset prices as it prevents
        them from becoming negative.

        The generation follows these steps:
        1.  **Parameter Conversion**: Converts the user-provided annual sample
            mean and standard deviation for each asset into the corresponding
            parameters (mu, sigma) of an annual log-normal distribution.
        2.  **Temporal Scaling**: Scales the annual log-normal parameters down to
            their monthly equivalents. These become the parameters for the
            underlying normal distribution of the logarithm of monthly returns.
        3.  **Covariance Construction**: Builds a monthly covariance matrix from the
            monthly standard deviations and the user-provided correlation matrix.
        4.  **Multivariate Normal Sampling**: Draws correlated random samples from a
            multivariate normal distribution. Each sample represents the
            logarithm of a monthly return factor, `ln(1 + R)`.
        5.  **Log-Normal Transformation**: Exponentiates the normal samples to
            obtain the log-normally distributed monthly return factors `(1 + R)`.
        6.  **Return Rate Calculation**: Subtracts 1 to yield the final monthly
            return rates `R`.

        Returns:
            A numpy array of shape (num_sequences, num_steps, num_assets)
            containing the correlated monthly return rates.
        """
        rng = np.random.default_rng(self.seed)

        # Extract Sample Parameters (they are in annual values)
        mu_sample = np.array(
            [self.assets[asset].mu for asset in self.asset_and_inflation_order]
        )
        sigma_sample = np.array(
            [self.assets[asset].sigma for asset in self.asset_and_inflation_order]
        )
        corr_matrix = np.array(self.correlation_matrix.matrix)

        # Convert mu and sigma to log-return equivalents
        ex = 1.0 + mu_sample
        vx = sigma_sample**2
        mu = np.log(ex) - 0.5 * np.log(1 + vx / ex**2)
        sigma = np.sqrt(np.log(1 + vx / ex**2))

        # Scale to monthly log returns
        monthly_mu = mu / 12
        monthly_sigma = sigma / np.sqrt(12)

        # Construct Monthly Normal Covariance Matrix
        D = np.diag(monthly_sigma)
        monthly_cov = D @ corr_matrix @ D

        # Generate Correlated Log-Normal Returns (i.e log(1 + rates))
        log_of_return_factors = rng.multivariate_normal(
            mean=monthly_mu,
            cov=monthly_cov,
            size=(self.num_sequences, self.num_steps),
        )

        # Convert to return rates where (1 + return_rate) is log-normal
        return_rates = np.exp(log_of_return_factors) - 1.0
        return return_rates

    def _generate_sequences_ou(self) -> np.ndarray:
        """
        Generates Ornstein-Uhlenbeck paths for all assets and all sequences
        in annual return rate space, using config parameters.
        Returns:
            np.ndarray: shape (num_sequences, num_steps, num_assets)
        """
        num_assets = len(self.asset_and_inflation_order)
        mu = np.array(
            [self.assets[asset].mu for asset in self.asset_and_inflation_order]
        )
        sigma = np.array(
            [self.assets[asset].sigma for asset in self.asset_and_inflation_order]
        )
        r0 = mu

        dt = 1.0 / 12.0  # Monthly step in years
        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal((self.num_sequences, self.num_steps, num_assets))

        return_rates = np.zeros(
            (self.num_sequences, self.num_steps, num_assets), dtype=np.float64
        )
        return_rates[:, 0, :] = r0

        for t in range(1, self.num_steps):
            return_rates[:, t, :] = (
                return_rates[:, t - 1, :]
                + THETA * (mu - return_rates[:, t - 1, :]) * dt
                + sigma * np.sqrt(dt) * Z[:, t, :]
            )

        # Clamp rates to avoid invalid values (e.g., below -1)
        return_rates = np.clip(return_rates, -0.99, None)
        # Convert annual return rates to monthly rates
        monthly_return_rates = (1.0 + return_rates) ** (1.0 / 12.0) - 1.0
        return monthly_return_rates
