# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Configuration models for the FIRE Monte Carlo simulation tool.

This module defines Pydantic models for validating and loading user-supplied
configuration data from TOML files. The models correspond to sections in the
configuration file and ensure that all required parameters for the simulation
are present and correctly typed.

Classes:
    - DeterministicInputs: User-controllable financial plan parameters loaded from the
      [deterministic_inputs] section of config.toml.
    - EconomicAssumptions: Economic and market return assumptions loaded from the
      [economic_assumptions] section of config.toml.

These models provide type safety and validation for the simulation engine.
"""

# config.py
# config.py
from pydantic import BaseModel, Field, ConfigDict, model_validator
import numpy as np
from typing import List, Dict

from firestarter.config.correlation_matrix import CorrelationMatrix


class PlannedContribution(BaseModel):
    """Represents a planned, single-year, contribution."""

    amount: float = Field(
        ..., description="Real (today's money) amount of the contribution."
    )
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when the contribution occurs."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannedExtraExpense(BaseModel):
    """Represents a planned, single-year, extra expense."""

    amount: float = Field(
        ..., description="Real (today's money) amount of the expense."
    )
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when the expense occurs."
    )
    description: str | None = Field(
        default=None, description="Optional description of the expense."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class Asset(BaseModel):
    """Represents a single financial asset class."""

    mu: float = Field(..., description="Expected annual arithmetic mean return.")
    sigma: float = Field(
        ..., description="Expected annual standard deviation of returns."
    )
    is_liquid: bool = Field(
        ...,
        description="True if the asset is part of the liquid, rebalanceable portfolio.",
    )
    withdrawal_priority: int | None = Field(
        default=None,
        description="Order for selling to cover cash shortfalls (lower is sold first). Required for liquid assets.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def check_withdrawal_priority(self) -> "Asset":
        """Ensure withdrawal_priority is set correctly based on liquidity."""
        is_liquid = self.is_liquid
        priority = self.withdrawal_priority

        if is_liquid and priority is None:
            raise ValueError("withdrawal_priority is required for liquid assets")
        if not is_liquid and priority is not None:
            raise ValueError("withdrawal_priority must not be set for illiquid assets")
        return self


class DeterministicInputs(BaseModel):
    """
    Pydantic model representing the deterministic financial inputs for the simulation.
    These parameters are loaded from the 'deterministic_inputs' section of config.toml.
    """

    initial_portfolio: dict[str, float] = Field(
        ...,
        description="Initial value of portfolio assets, mapping asset name to amount.",
    )
    initial_bank_balance: float = Field(
        ..., description="Initial bank account balance."
    )

    bank_lower_bound: float = Field(
        ...,
        description=(
            "Minimum desired bank balance in real (today's money) terms. "
            "If balance drops below this, funds are transferred from investment."
        ),
    )
    bank_upper_bound: float = Field(
        ...,
        description=(
            "Maximum desired bank balance in real (today's money) terms. "
            "If balance exceeds this, excess funds are transferred to investment."
        ),
    )

    years_to_simulate: int = Field(
        ..., description="Total number of years the retirement simulation will run."
    )

    monthly_salary: float = Field(
        ..., description="Initial real (today's money) monthly salary."
    )
    salary_inflation_factor: float = Field(
        ...,
        description=(
            "Factor by which salary adjusts to inflation (1.0 = tracks inflation, "
            "1.01 = 1% above inflation)."
        ),
    )
    salary_start_year: int = Field(
        ...,
        description=(
            "Year index (0-indexed) when salary income starts. "
            "E.g., 0 for immediate start."
        ),
    )
    salary_end_year: int = Field(
        ...,
        description=(
            "Year index (0-indexed) when salary income ends (exclusive). "
            "E.g., 5 for 5 years of salary, meaning salary ends *before* Year 5 begins."
        ),
    )

    monthly_pension: float = Field(
        ..., description="Initial real (today's money) monthly pension."
    )
    pension_inflation_factor: float = Field(
        ...,
        description=(
            "Factor by which pension adjusts to inflation (e.g., 1.0 for full adjustment, "
            "0.6 for 60% adjustment)."
        ),
    )
    pension_start_year: int = Field(
        ..., description="Year index (0-indexed) when pension income starts."
    )

    planned_contributions: list[PlannedContribution] = Field(
        default_factory=list,
        description=(
            "List of planned contributions. e.g. [{amount = 10000, year = 2}, ...]"
        ),
    )

    annual_fund_fee: float = Field(
        ...,
        description="Total Expense Ratio (TER) as an annual percentage of investment assets.",
    )
    monthly_expenses: float = Field(
        ...,
        description="Initial real (today's money) fixed monthly expenses for living costs.",
    )
    planned_extra_expenses: list[PlannedExtraExpense] = Field(
        default_factory=list,
        description=(
            "List of planned extra expenses. e.g. [{amount = 15000, year = 3, description = 'Car'}, ...]"
        ),
    )

    planned_house_purchase_cost: float = Field(
        ...,
        description=(
            "Initial real (today's money) cost of a house "
            "to be purchased at house_purchase_year (or rebalancing_year if not set)."
        ),
    )
    house_purchase_year: int | None = Field(
        default=None,
        description=(
            "Year index (0-based) when the house is purchased. "
            "If None, defaults to rebalancing_year."
        ),
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketAssumptions(BaseModel):
    """
    Pydantic model representing the economic assumptions for the simulation.
    These parameters are loaded from the 'market_assumptions' section of config.toml.
    """

    assets: dict[str, Asset] = Field(
        ..., description="A dictionary containing all defined financial assets."
    )
    correlation_matrix: CorrelationMatrix | None = Field(
        default=None,
        description=(
            "Optional correlation matrix for asset returns and inflation. "
            "If not provided, assets are assumed to be uncorrelated."
        ),
    )

    @model_validator(mode="after")
    def check_assets(self) -> "MarketAssumptions":
        """
        Validates two conditions across all assets:
        1. Liquid assets must have a `withdrawal_priority`.
        2. `withdrawal_priority` must be unique among all liquid assets.
        """
        priorities = set()
        for asset_name, asset in self.assets.items():
            if asset.is_liquid:
                if asset.withdrawal_priority is None:
                    raise ValueError(
                        f"Liquid asset '{asset_name}' must have a withdrawal_priority."
                    )
                if asset.withdrawal_priority in priorities:
                    raise ValueError(
                        "Withdrawal priorities for liquid assets must be unique."
                    )
                priorities.add(asset.withdrawal_priority)
        return self

    @staticmethod
    def _convert_to_lognormal(
        arith_mu: float, arith_sigma: float
    ) -> tuple[float, float]:
        """
        Helper function to convert arithmetic mean and standard deviation
        to log-normal parameters (mu_log, sigma_log).

        This matches the implementation of _convert_arithmetic_to_lognormal in helpers.py.
        """
        if arith_mu <= -1.0:
            raise ValueError(
                f"Arithmetic mean ({arith_mu}) must be strictly greater than -1 "
                + "to convert to log-normal parameters."
            )

        ex: float = 1.0 + arith_mu
        stdx: float = arith_sigma

        if stdx == 0.0:
            sigma_log: float = 0.0
        else:
            sigma_log = float(np.sqrt(np.log(1.0 + (stdx / ex) ** 2)))

        mu_log: float = float(np.log(ex) - 0.5 * sigma_log**2)

        return mu_log, sigma_log

    @property
    def lognormal(self) -> dict[str, tuple[float, float]]:
        """Return log-normal parameters for all assets and inflation."""
        return {
            asset_name: self._convert_to_lognormal(asset.mu, asset.sigma)
            for asset_name, asset in self.assets.items()
        }

    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioRebalance(BaseModel):
    """
    Represents a single portfolio rebalance event.

    Attributes:
        year (int): The year (0-indexed) when this rebalance occurs.
        weights (dict[str, float]): A dictionary mapping liquid asset names to their
                                    target weights, which must sum to 1.0.
        description (str | None): Optional description of the rebalance event.
    """

    year: int
    description: str | None = None
    weights: dict[str, float] = {}

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def build_weights(cls, values: dict) -> dict:
        """
        Collect all undefined fields into a 'weights' dictionary.
        """
        defined_fields = {"year", "description", "weights"}
        weights = values.get("weights", {})

        for key, value in list(values.items()):
            if key not in defined_fields:
                weights[key] = values.pop(key)

        values["weights"] = weights
        return values

    @model_validator(mode="after")
    def check_weights(self) -> "PortfolioRebalance":
        """
        Validate that weights are not empty and sum to 1.0.
        """
        if not self.weights:
            raise ValueError("Rebalance weights cannot be empty.")

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Rebalance weights must sum to 1.0.")

        return self


class PortfolioRebalances(BaseModel):
    """A container for a list of portfolio rebalance events."""

    rebalances: List[PortfolioRebalance] = Field(
        ..., description="List of portfolio rebalances with year and weights."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def check_unique_years(self) -> "PortfolioRebalances":
        """Validate that rebalance years are unique."""
        years = [r.year for r in self.rebalances]
        if len(years) != len(set(years)):
            raise ValueError("Rebalance years must be unique.")
        return self


class SimulationParameters(BaseModel):
    num_simulations: int = Field(
        ..., gt=0, description="Number of Monte Carlo simulations to run."
    )
    random_seed: int | None = Field(
        default=None,
        description="Optional random seed for deterministic runs. If None, uses entropy.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class Shock(BaseModel):
    """Defines a one-time financial shock."""

    year: int
    asset: str
    magnitude: float
    description: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class Shocks(BaseModel):
    events: list[Shock] = Field(
        default_factory=list, description="List of shock events."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)
