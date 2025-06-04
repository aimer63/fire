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

    s_real_monthly: float = Field(
        ..., description="Initial real (today's money) monthly salary."
    )
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
            "Year index (0-indexed) when salary income starts. "
            "E.g., 0 for immediate start."
        ),
    )
    y_s_end_idx: int = Field(
        ...,
        description=(
            "Year index (0-indexed) when salary income ends (exclusive). "
            "E.g., 5 for 5 years of salary, meaning salary ends *before* Year 5 begins."
        ),
    )

    p_real_monthly: float = Field(
        ..., description="Initial real (today's money) monthly pension."
    )
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
        description="Initial real (today's money) cost of a house to be purchased at rebalancing_year_idx.",
    )

    class Config:
        """Pydantic configuration for the DeterministicInputs model."""

        extra = "forbid"  # Ensures no unexpected fields are present in the config.toml section.
