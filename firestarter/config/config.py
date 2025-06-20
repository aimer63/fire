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
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from typing import List
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


class PlannedExtraExpenses(BaseModel):
    """Represents a planned, single-year, extra expense."""

    amount: float = Field(
        ..., description="Real (today's money) amount of the expense."
    )
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when the expense occurs."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeterministicInputs(BaseModel):
    """
    Pydantic model representing the deterministic financial inputs for the simulation.
    These parameters are loaded from the 'deterministic_inputs' section of config.toml.
    """

    initial_investment: float = Field(
        ..., description="Initial investment portfolio value."
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
    planned_extra_expenses: list[PlannedExtraExpenses] = Field(
        default_factory=list,
        description=(
            "List of planned extra expenses. e.g. [{amount = 15000, year = 3}, ...]"
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
    These parameters are loaded from the 'economic_assumptions' section of config.toml.
    """

    stock_mu: float = Field(
        ..., description="Arithmetic mean annual return for stocks."
    )
    stock_sigma: float = Field(
        ..., description="Standard deviation of annual returns for stocks."
    )

    bond_mu: float = Field(..., description="Arithmetic mean annual return for bonds.")
    bond_sigma: float = Field(
        ..., description="Standard deviation of annual returns for bonds."
    )

    str_mu: float = Field(
        ..., description="Arithmetic mean annual return for short-term reserves (STR)."
    )
    str_sigma: float = Field(
        ..., description="Standard deviation of annual returns for STR."
    )

    fun_mu: float = Field(
        ...,
        description="Arithmetic mean annual return for 'fun money' (e.g., crypto/silver).",
    )
    fun_sigma: float = Field(
        ..., description="Standard deviation of annual returns for 'fun money'."
    )

    real_estate_mu: float = Field(
        ...,
        description=(
            "Arithmetic mean annual return for real estate "
            "(capital gains, net of maintenance)."
        ),
    )
    real_estate_sigma: float = Field(
        ..., description="Standard deviation of annual returns for real estate."
    )

    pi_mu: float = Field(..., description="Arithmetic mean of annual inflation rate.")
    pi_sigma: float = Field(
        ..., description="Standard deviation of annual inflation rate."
    )

    correlation_matrix: CorrelationMatrix | None = Field(
        default=None,
        description=(
            "Optional correlation matrix for asset returns and inflation. "
            "If not provided, assets are assumed to be uncorrelated."
        ),
    )

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
            "stocks": self._convert_to_lognormal(self.stock_mu, self.stock_sigma),
            "bonds": self._convert_to_lognormal(self.bond_mu, self.bond_sigma),
            "str": self._convert_to_lognormal(self.str_mu, self.str_sigma),
            "fun": self._convert_to_lognormal(self.fun_mu, self.fun_sigma),
            "real_estate": self._convert_to_lognormal(
                self.real_estate_mu, self.real_estate_sigma
            ),
            "inflation": self._convert_to_lognormal(
                self.pi_mu, self.pi_sigma
            ),  # <-- FIXED HERE
        }

    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioRebalance(BaseModel):
    """
    Represents a single portfolio rebalance event.

    Attributes:
        year (int): The year (0-indexed) when this rebalance occurs.
        stocks (float): Weight for stocks (liquid assets only).
        bonds (float): Weight for bonds (liquid assets only).
        str (float): Weight for short-term reserves (STR, liquid assets only).
        fun (float): Weight for 'fun money' (liquid assets only).
    """

    year: int
    stocks: float
    bonds: float
    str: float
    fun: float


class PortfolioRebalances(BaseModel):
    """
    Contains the list of all scheduled portfolio rebalances.

    Attributes:
        rebalances (List[PortfolioRebalance]): List of rebalance events, each specifying
        the year and weights.
    """

    rebalances: List[PortfolioRebalance] = Field(
        ..., description="List of portfolio rebalances with year and weights."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class SimulationParameters(BaseModel):
    num_simulations: int = Field(
        ..., gt=0, description="Number of Monte Carlo simulations to run."
    )
    random_seed: int | None = Field(
        default=None,
        description="Optional random seed for deterministic runs. If None, uses entropy.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShockEvent(BaseModel):
    year: int = Field(..., description="Year index of the shock (0-indexed).")
    asset: str = Field(..., description="Asset affected by the shock (e.g., 'Stocks').")
    magnitude: float = Field(
        ..., description="Magnitude of the shock (e.g., -0.35 for -35%)."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class Shocks(BaseModel):
    events: list[ShockEvent] = Field(
        default_factory=list, description="List of shock events."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)
