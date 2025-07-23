#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

"""
Configuration models for the FIRE Monte Carlo simulation tool.

This module defines Pydantic models for validating and loading user-supplied
configuration data from TOML files. The models correspond to sections in the
configuration file and ensure that all required parameters for the simulation
are present and correctly typed.

Features:
- Strict type validation for all simulation parameters.
- Cross-section consistency checks (e.g., asset names, withdrawal priorities).

Classes:
    - DeterministicInputs: User-controllable financial plan parameters.
    - Asset: Financial asset class definition.
    - PortfolioRebalance: Portfolio rebalance event.
    - SimulationParameters: Simulation run parameters.
    - Shock: One-time financial shock event.
    - Paths: Output paths for simulation results.
    - Config: Top-level container for the entire simulation configuration.

These models provide type safety and validation for the simulation engine.
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
import numpy as np

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


class SalaryStep(BaseModel):
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when this salary step starts."
    )
    monthly_amount: float = Field(
        ..., ge=0.0, description="Monthly salary amount (today's money) for this step."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


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

    monthly_salary_steps: list[SalaryStep] = Field(
        ...,
        description="List of salary steps, each with a start year and monthly amount.",
    )
    salary_inflation_factor: float = Field(
        ...,
        ge=0.0,
        description=(
            "Quota of inflation that is applied to salary after the last step. "
            "1.0 = tracks inflation, 0.0 = no inflation adjustment, >1.0 = grows faster than inflation."
        ),
    )
    salary_end_year: int = Field(
        ...,
        ge=0,
        description="Year index (0-indexed) when salary income ends (exclusive). Salary stops before this year begins.",
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

    @model_validator(mode="after")
    def validate_salary_steps(self) -> "DeterministicInputs":
        years = [step.year for step in self.monthly_salary_steps]
        if len(set(years)) != len(years):
            raise ValueError("Years in monthly_salary_steps must be unique.")
        if not years:
            raise ValueError("At least one salary step must be provided.")
        if sorted(years) != years:
            raise ValueError(
                "Years in monthly_salary_steps must be sorted in ascending order."
            )
        last_step_year = years[-1]
        if self.salary_end_year < last_step_year:
            raise ValueError(
                "salary_end_year must be >= the last year in monthly_salary_steps."
            )
        return self

    @model_validator(mode="after")
    def validate_bank_bounds(self) -> "DeterministicInputs":
        if self.bank_lower_bound > self.bank_upper_bound:
            raise ValueError(
                "bank_lower_bound must be less than or equal to bank_upper_bound"
            )
        return self


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
    def check_mu_and_sigma(self) -> "Asset":
        if self.mu <= -1.0:
            raise ValueError("mu must be greater than -1.0")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        return self

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
    """Defines a one-time financial shock for a given year."""

    year: int
    description: str | None = None
    impact: dict[str, float] = {}

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def build_impact_dict(cls, values: dict) -> dict:
        """
        Collect all undefined fields into the 'impact' dictionary.
        This allows for a cleaner TOML structure, e.g., `stocks = -0.25`.
        """
        defined_fields = {"year", "description", "impact"}
        impact = values.get("impact", {})

        for key, value in list(values.items()):
            if key not in defined_fields:
                impact[key] = values.pop(key)

        values["impact"] = impact
        return values


class Paths(BaseModel):
    """Defines paths for simulation outputs."""

    output_root: str = Field(
        default="output", description="The root directory for all output files."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class Config(BaseModel):
    """Top-level container for the entire simulation configuration."""

    assets: dict[str, Asset]
    deterministic_inputs: DeterministicInputs
    correlation_matrix: CorrelationMatrix | None = Field(
        default=None,
        description=(
            "Correlation matrix for asset returns and inflation."
            "To get indipendent draws provide the identity matrix."
        ),
    )
    portfolio_rebalances: list[PortfolioRebalance]
    simulation_parameters: SimulationParameters
    shocks: list[Shock] | None = None
    paths: Paths | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_cross_config_consistency(self) -> "Config":
        """
        Performs validation checks that require access to multiple configuration sections.
        """
        # 1. Establish the definitive set of asset names from the top-level assets
        defined_assets = set(self.assets.keys())
        if "inflation" not in defined_assets:
            raise ValueError(
                "An asset named 'inflation' must be defined in the assets section."
            )

        # 1a. Validate that withdrawal_priority values for liquid assets are unique
        priorities = [
            asset.withdrawal_priority
            for asset in self.assets.values()
            if asset.is_liquid and asset.withdrawal_priority is not None
        ]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Withdrawal priorities for liquid assets must be unique")

        # 1b. Validate that all assets in initial_portfolio are declared in assets
        initial_portfolio_assets = set(
            self.deterministic_inputs.initial_portfolio.keys()
        )
        undeclared_assets = initial_portfolio_assets - defined_assets
        if undeclared_assets:
            raise ValueError(
                f"Assets in initial_portfolio not declared in [assets]: {sorted(list(undeclared_assets))}"
            )

        # 2. Validate the correlation matrix asset list
        if self.correlation_matrix:
            matrix_assets = set(self.correlation_matrix.assets_order)
            if defined_assets != matrix_assets:
                missing = defined_assets - matrix_assets
                extra = matrix_assets - defined_assets
                error_msg = "Correlation matrix assets must match defined assets."
                if missing:
                    error_msg += f" Missing: {sorted(list(missing))}."
                if extra:
                    error_msg += f" Extra: {sorted(list(extra))}."
                raise ValueError(error_msg)

        # 3. Validate that all shock events target defined assets
        if self.shocks:
            for shock in self.shocks:
                for asset_name in shock.impact:
                    if asset_name not in defined_assets:
                        raise ValueError(
                            f"Shock in year {shock.year} targets an undefined asset: "
                            f"'{asset_name}'. Valid assets are: {sorted(list(defined_assets))}"
                        )

        # 4. Validate that rebalance years are unique
        rebalance_years = [r.year for r in self.portfolio_rebalances]
        if len(rebalance_years) != len(set(rebalance_years)):
            raise ValueError("Rebalance years must be unique.")

        return self
