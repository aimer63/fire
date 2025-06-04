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
from pydantic import BaseModel, Field


class DeterministicInputs(BaseModel):
    """
    Pydantic model representing the deterministic financial inputs for the simulation.
    These parameters are loaded from the 'deterministic_inputs' section of config.toml.
    """

    i0: float = Field(..., description="Initial investment portfolio value.")
    b0: float = Field(..., description="Initial bank account balance.")

    real_bank_lower_bound: float = Field(
        ...,
        description=(
            "Minimum desired bank balance in real (today's money) terms. "
            "If balance drops below this, funds are transferred from investment."
        ),
    )
    real_bank_upper_bound: float = Field(
        ...,
        description=(
            "Maximum desired bank balance in real (today's money) terms. "
            "If balance exceeds this, excess funds are transferred to investment."
        ),
    )

    t_ret_years: int = Field(
        ..., description="Total number of years the retirement simulation will run."
    )

    s_real_monthly: float = Field(..., description="Initial real (today's money) monthly salary.")
    salary_inflation_adjustment_factor: float = Field(
        ...,
        description=(
            "Factor by which salary adjusts to inflation (1.0 = tracks inflation, "
            "1.01 = 1% above inflation)."
        ),
    )
    y_s_start_idx: int = Field(
        ...,
        description=(
            "Year index (0-indexed) when salary income starts. " "E.g., 0 for immediate start."
        ),
    )
    y_s_end_idx: int = Field(
        ...,
        description=(
            "Year index (0-indexed) when salary income ends (exclusive). "
            "E.g., 5 for 5 years of salary, meaning salary ends *before* Year 5 begins."
        ),
    )

    p_real_monthly: float = Field(..., description="Initial real (today's money) monthly pension.")
    pension_inflation_adjustment_factor: float = Field(
        ...,
        description=(
            "Factor by which pension adjusts to inflation (e.g., 1.0 for full adjustment, "
            "0.6 for 60% adjustment)."
        ),
    )
    y_p_start_idx: int = Field(
        ..., description="Year index (0-indexed) when pension income starts."
    )

    c_real_monthly_initial: float = Field(
        ...,
        description=(
            "Fixed initial real (today's money) monthly contribution to invested assets. "
            "Models regular external contributions (e.g., a monthly savings plan)."
        ),
    )
    c_planned: list[tuple[float, int]] = Field(
        default_factory=list,
        description=(
            "List of planned one-time contributions to investments: "
            "[[amount_in_real_terms, year_index]]. "
            "Year index is 0-indexed."
        ),
    )

    ter_annual_percentage: float = Field(
        ...,
        description="Total Expense Ratio (TER) as an annual percentage of investment assets.",
    )
    x_real_monthly_initial: float = Field(
        ...,
        description="Initial real (today's money) fixed monthly expenses for living costs.",
    )
    x_planned_extra: list[tuple[float, int]] = Field(
        default_factory=list,
        description=(
            "List of planned one-time extra expenses: "
            "[[amount_in_real_terms, year_index]]. "
            "Year index is 0-indexed."
        ),
    )

    h0_real_cost: float = Field(
        ...,
        description=(
            "Initial real (today's money) cost of a house "
            "to be purchased at rebalancing_year_idx."
        ),
    )

    class Config:
        """Pydantic configuration for the DeterministicInputs model."""

        # Ensures no unexpected fields are present in the config.toml section.
        extra = "forbid"


class EconomicAssumptions(BaseModel):
    """
    Pydantic model representing the economic assumptions for the simulation.
    These parameters are loaded from the 'economic_assumptions' section of config.toml.
    """

    stock_mu: float = Field(..., description="Arithmetic mean annual return for stocks.")
    stock_sigma: float = Field(..., description="Standard deviation of annual returns for stocks.")

    bond_mu: float = Field(..., description="Arithmetic mean annual return for bonds.")
    bond_sigma: float = Field(..., description="Standard deviation of annual returns for bonds.")

    str_mu: float = Field(
        ..., description="Arithmetic mean annual return for short-term reserves (STR)."
    )
    str_sigma: float = Field(..., description="Standard deviation of annual returns for STR.")

    fun_mu: float = Field(
        ..., description="Arithmetic mean annual return for 'fun money' (e.g., crypto/silver)."
    )
    fun_sigma: float = Field(
        ..., description="Standard deviation of annual returns for 'fun money'."
    )

    real_estate_mu: float = Field(
        ...,
        description=(
            "Arithmetic mean annual return for real estate " "(capital gains, net of maintenance)."
        ),
    )
    real_estate_sigma: float = Field(
        ..., description="Standard deviation of annual returns for real estate."
    )

    mu_pi: float = Field(..., description="Arithmetic mean of annual inflation rate.")
    sigma_pi: float = Field(..., description="Standard deviation of annual inflation rate.")

    class Config:
        """Pydantic configuration for the EconomicAssumptions model."""

        extra = "forbid"  # Ensures no unexpected fields are present in the config.toml section.
