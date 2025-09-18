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

from firecast.config.correlation_matrix import CorrelationMatrix


class PlannedContribution(BaseModel):
    """Represents a planned, single-year, contribution to a specific asset."""

    amount: float = Field(
        ..., description="Real (today's money) amount of the contribution."
    )
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when the contribution occurs."
    )
    asset: str | None = Field(
        default=None,
        description="Name of the asset to receive the contribution. If None, "
        "the contribution is allocated according to current portfolio weights.",
    )
    description: str | None = Field(
        default=None, description="Optional description of the contribution."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_asset(self) -> "PlannedContribution":
        if self.asset == "inflation":
            raise ValueError(
                "Planned contribution cannot target the 'inflation' asset."
            )
        return self


class PlannedIlliquidPurchase(BaseModel):
    """Represents a planned purchase of an illiquid asset from liquid assets."""

    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when the purchase occurs."
    )
    amount: float = Field(
        ..., description="Real (today's money) amount of the purchase."
    )
    asset: str = Field(
        ..., description="Name of the illiquid asset to receive the purchase."
    )
    description: str | None = Field(
        default=None, description="Optional description of the purchase."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_asset(self) -> "PlannedIlliquidPurchase":
        if self.asset == "inflation":
            raise ValueError("Illiquid purchase cannot target the 'inflation' asset.")
        return self


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


class IncomeStep(BaseModel):
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when this income step starts."
    )
    monthly_amount: float = Field(
        ..., ge=0.0, description="Monthly income amount (today's money) for this step."
    )
    description: str | None = Field(
        default=None, description="Optional description of the income step."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpenseStep(BaseModel):
    year: int = Field(
        ..., ge=0, description="Year index (0-indexed) when this expense step starts."
    )
    monthly_amount: float = Field(
        ..., ge=0.0, description="Monthly expense amount (today's money) for this step."
    )
    description: str | None = Field(
        default=None, description="Optional description of the expense step."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class DeterministicInputs(BaseModel):
    """
    Pydantic model representing the deterministic financial inputs for the simulation.
    These parameters are loaded from the 'deterministic_inputs' section of config.toml.
    """

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

    monthly_income_steps: list[IncomeStep] = Field(
        default_factory=list,
        description="List of income steps, each with a start year and monthly amount.",
    )
    income_inflation_factor: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Quota of inflation that is applied to income after the last step. "
            "1.0 = tracks inflation, 0.0 = no inflation adjustment, >1.0 = grows faster than inflation."
        ),
    )
    income_end_year: int = Field(
        default=0,
        ge=0,
        description="Year index (0-indexed) when income income ends (exclusive). Income stops before this year begins.",
    )

    monthly_pension: float = Field(
        default=0.0, description="Initial real (today's money) monthly pension."
    )
    pension_inflation_factor: float | None = Field(
        default=None,
        description=(
            "Factor by which pension adjusts to inflation (e.g., 1.0 for full adjustment, "
            "0.6 for 60% adjustment)."
        ),
    )
    pension_start_year: int | None = Field(
        default=None, description="Year index (0-indexed) when pension income starts."
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
    transactions_fee: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional transaction fee applied to all investments and disinvestments. "
            "Format: {min: float, rate: float, max: float}. "
            "Fee is calculated as max(min, amount * rate), capped at max if max > 0. "
            "If omitted or None, no fee is applied."
        ),
    )
    investment_lot_size: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum chunk size for investing excess bank balance. "
            "Only multiples of this amount are invested when bank balance exceeds upper bound. "
            "If 0.0, all excess is invested immediately."
        ),
    )
    planned_illiquid_purchases: list[PlannedIlliquidPurchase] = Field(
        default_factory=list,
        description="List of planned purchases of illiquid assets from liquid assets.",
    )
    monthly_expenses_steps: list[ExpenseStep] = Field(
        default_factory=list,
        description="List of expense steps, each with a start year and monthly amount.",
    )
    planned_extra_expenses: list[PlannedExtraExpense] = Field(
        default_factory=list,
        description=(
            "List of planned extra expenses. e.g. [{amount = 15000, year = 3, description = 'Car'}, ...]"
        ),
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_income_steps(self) -> "DeterministicInputs":
        # If income steps are not provided or empty, skip further checks
        if not self.monthly_income_steps:
            return self
        # If income steps are present, require explicit user specification of inflation factor and end year
        if "income_inflation_factor" not in self.__pydantic_fields_set__:
            raise ValueError(
                "income_inflation_factor must be explicitly specified when monthly_income_steps are provided."
            )
        if "income_end_year" not in self.__pydantic_fields_set__:
            raise ValueError(
                "income_end_year must be explicitly specified when monthly_income_steps are provided."
            )
        years = [step.year for step in self.monthly_income_steps]
        if len(set(years)) != len(years):
            raise ValueError("Years in monthly_income_steps must be unique.")
        if sorted(years) != years:
            raise ValueError(
                "Years in monthly_income_steps must be sorted in ascending order."
            )
        last_step_year = years[-1]
        if self.income_end_year <= last_step_year:
            raise ValueError(
                "income_end_year must be > the year of the last IncomeStep."
            )
        if (
            self.pension_start_year is not None
            and self.pension_start_year < self.income_end_year
        ):
            raise ValueError("pension_start_year must be >= income_end_year.")
        return self

    def validate_pension_fields(self) -> "DeterministicInputs":
        if self.monthly_pension > 0.0:
            if self.pension_inflation_factor is None:
                raise ValueError(
                    "pension_inflation_factor must be provided when monthly_pension > 0."
                )
            if self.pension_start_year is None:
                raise ValueError(
                    "pension_start_year must be provided when monthly_pension > 0."
                )
        return self

    def validate_expense_steps(self) -> "DeterministicInputs":
        # If expense steps are not provided or empty, treat as zero expenses
        if not self.monthly_expenses_steps:
            return self
        years = [step.year for step in self.monthly_expenses_steps]
        if len(set(years)) != len(years):
            raise ValueError("Years in monthly_expenses_steps must be unique.")
        if sorted(years) != years:
            raise ValueError(
                "Years in monthly_expenses_steps must be sorted in ascending order."
            )
        if years and years[0] != 0:
            raise ValueError(
                "The first step in monthly_expenses_steps must start at year 0."
            )
        return self

    @model_validator(mode="after")
    def validate_bank_bounds(self) -> "DeterministicInputs":
        if self.bank_lower_bound > self.bank_upper_bound:
            raise ValueError(
                "bank_lower_bound must be less than or equal to bank_upper_bound"
            )
        if self.initial_bank_balance < self.bank_lower_bound:
            raise ValueError(
                "initial_bank_balance must be greater than or equal to bank_lower_bound"
            )
        return self

    @model_validator(mode="after")
    def validate_transactions_fee(self) -> "DeterministicInputs":
        if self.transactions_fee is not None:
            fee = self.transactions_fee
            required_keys = {"min", "rate", "max"}
            missing = required_keys - set(fee.keys())
            if missing:
                raise ValueError(
                    f"transactions_fee must contain keys: {sorted(list(missing))}"
                )
            if not isinstance(fee["min"], (int, float)) or fee["min"] < 0:
                raise ValueError(
                    "transactions_fee['min'] must be a non-negative number."
                )
            if not isinstance(fee["rate"], (int, float)) or fee["rate"] < 0:
                raise ValueError(
                    "transactions_fee['rate'] must be a non-negative number."
                )
            if not isinstance(fee["max"], (int, float)) or fee["max"] < 0:
                raise ValueError(
                    "transactions_fee['max'] must be a non-negative number (0 means no cap)."
                )
            if fee["max"] < fee["min"]:
                raise ValueError(
                    "transactions_fee['max'] must be greater than or equal to 'min'."
                )
        return self


class Asset(BaseModel):
    """Represents a single financial asset class."""

    mu: float = Field(..., description="Expected annual arithmetic mean return.")
    sigma: float = Field(
        ..., description="Expected annual standard deviation of returns."
    )
    withdrawal_priority: int | None = Field(
        default=None,
        description="Order for selling to cover cash shortfalls (lower is sold first). Required for liquid assets.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_mu_and_sigma(self) -> "Asset":
        if self.mu <= -1.0:
            raise ValueError("mu must be greater than -1.0")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        return self


class PortfolioRebalance(BaseModel):
    """
    Represents a single portfolio rebalance event.

    Attributes:
        year (int): The year (0-indexed) when this rebalance first occurs.
        period (int): The period in years for periodic rebalancing. If 0, rebalance is applied only once at 'year'.
        weights (dict[str, float]): A dictionary mapping liquid asset names to their
                                    target weights, which must sum to 1.0.
        description (str | None): Optional description of the rebalance event.
    """

    year: int
    period: int = Field(
        default=0,
        ge=0,
        description="Period in years for periodic rebalancing. 0 means rebalance is applied only once at 'year'.",
    )
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
        # Establish the definitive set of asset names from the top-level assets
        defined_assets = set(self.assets.keys())
        if "inflation" not in defined_assets:
            raise ValueError(
                "An asset named 'inflation' must be defined in the assets section."
            )

        # Validate withdrawal_priority for all assets and check uniqueness (excluding illiquid assets)
        priorities = []
        for name, asset in self.assets.items():
            if name == "inflation":
                if asset.withdrawal_priority is not None:
                    raise ValueError(
                        "withdrawal_priority must be None for the 'inflation' asset."
                    )
            else:
                if asset.withdrawal_priority is not None:
                    priorities.append(asset.withdrawal_priority)
        if len(priorities) != len(set(priorities)):
            raise ValueError(
                "withdrawal_priority values for assets must be unique (excluding illiquid assets)."
            )

        # Validate planned contributions reference valid assets (if asset is specified)
        for contrib in self.deterministic_inputs.planned_contributions:
            if contrib.asset is not None and contrib.asset not in self.assets:
                raise ValueError(
                    f"Planned contribution targets unknown asset '{contrib.asset}'."
                )

        # Validate planned illiquid purchases reference valid illiquid assets
        for purchase in self.deterministic_inputs.planned_illiquid_purchases:
            if purchase.asset not in self.assets:
                raise ValueError(
                    f"Planned illiquid purchase targets unknown asset '{purchase.asset}'."
                )
            if self.assets[purchase.asset].withdrawal_priority is not None:
                raise ValueError(
                    f"Planned illiquid purchase asset '{purchase.asset}' must be illiquid."
                )
            if purchase.asset == "inflation":
                raise ValueError(
                    "Planned illiquid purchase cannot target the 'inflation' asset."
                )

        # Validate the correlation matrix asset list
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

        # Validate that all shock events target defined assets
        if self.shocks:
            for shock in self.shocks:
                for asset_name in shock.impact:
                    if asset_name not in defined_assets:
                        raise ValueError(
                            f"Shock in year {shock.year} targets an undefined asset: "
                            f"'{asset_name}'. Valid assets are: {sorted(list(defined_assets))}"
                        )

        rebalance_years = [r.year for r in self.portfolio_rebalances]
        # Enforce presence of a rebalance at year 0
        if 0 not in rebalance_years:
            raise ValueError(
                "A portfolio rebalance at year 0 is required to set initial target weights."
            )

        # Validate that rebalance years are unique
        if len(rebalance_years) != len(set(rebalance_years)):
            raise ValueError("Rebalance years must be unique.")

        # Validate that rebalance weights only reference liquid assets and sum to 1.0
        liquid_assets = {
            k for k, v in self.assets.items() if v.withdrawal_priority is not None
        }
        for rebalance in self.portfolio_rebalances:
            non_liquid = set(rebalance.weights.keys()) - liquid_assets
            if non_liquid:
                raise ValueError(
                    f"Rebalance at year {rebalance.year} weights reference non-liquid assets: {sorted(list(non_liquid))}."
                )

        return self
