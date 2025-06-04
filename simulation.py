"""
simulation.py

This module contains the core logic for running Monte Carlo simulations
of a Personal Financial Independence / Early Retirement (FIRE) plan.

It simulates the evolution of investment portfolios and bank accounts over
a specified retirement period, accounting for various factors such as:
- Asset returns (stocks, bonds, short-term reserves, alternative assets, real estate)
- Inflation
- Monthly expenses and income (salary, pension)
- Planned contributions and extra expenses
- Portfolio rebalancing strategies
- Potential market shocks

The module aims to determine the success rate of a FIRE plan and provide
detailed historical data for analysis of wealth evolution and asset allocation.
"""

import numpy as np
from typing import Any, TypedDict
from numpy.typing import NDArray

# Import helper functions
from helpers import annual_to_monthly_compounded_rate, calculate_initial_asset_values

# Import the DeterministicInputs Pydantic model
from config import DeterministicInputs


# Define a TypedDict for the return value of run_single_fire_simulation
class SimulationRunResult(TypedDict):
    """
    Represents the complete result set from a single Monte Carlo simulation run.
    """

    success: bool
    months_lasted: int
    final_investment: float
    final_bank_balance: float
    annual_inflations_seq: NDArray[np.float64]
    nominal_wealth_history: NDArray[np.float64]
    bank_balance_history: NDArray[np.float64]
    pre_rebalancing_allocations_nominal: dict[str, float]
    pre_rebalancing_allocations_real: dict[str, float]
    rebalancing_allocations_nominal: dict[str, float]
    rebalancing_allocations_real: dict[str, float]
    final_allocations_nominal: dict[str, float]
    final_allocations_real: dict[str, float]


def run_single_fire_simulation(
    det_inputs: DeterministicInputs,
    mu_log_inflation: float,
    sigma_log_inflation: float,
    rebalancing_trigger_year_idx: int,
    phase1_stocks_weight: float,
    phase1_bonds_weight: float,
    phase1_str_weight: float,
    phase1_fun_weight: float,
    phase1_real_estate_weight: float,
    phase2_stocks_weight: float,
    phase2_bonds_weight: float,
    phase2_str_weight: float,
    phase2_fun_weight: float,
    phase2_real_estate_weight: float,
    mu_log_stocks: float,
    sigma_log_stocks: float,
    mu_log_bonds: float,
    sigma_log_bonds: float,
    mu_log_str: float,
    sigma_log_str: float,
    mu_log_fun: float,
    sigma_log_fun: float,
    mu_log_real_estate: float,
    sigma_log_real_estate: float,
    shock_events: list[dict[str, Any]],
) -> SimulationRunResult:
    """
    Runs a single Monte Carlo simulation for a Financial Independence/Retirement plan.
    Simulates wealth evolution, income, expenses, and asset allocations over time.

    Args:
        initial_investment (float): Initial investment value.
        initial_bank_balance (float): Initial bank account balance.
        total_retirement_months (int): Total simulation duration in months.
        total_retirement_years (int): Total simulation duration in years.
        initial_real_monthly_expenses (float): Initial real monthly expenses.
        planned_contributions (list[tuple[float, int]]): List of (amount, year_idx) for planned contributions.
        planned_extra_expenses (list[tuple[float, int]]): List of (amount, year_idx) for planned extra
            expenses.
        initial_real_monthly_pension (float): Initial real monthly pension.
        pension_inflation_adjustment_factor (float): Factor for pension inflation
            adjustment.
        pension_start_year_idx (int): Year index when pension starts.
        initial_real_monthly_salary (float): Initial real monthly salary.
        salary_inflation_adjustment_factor (float): Factor for salary inflation
            adjustment.
        salary_start_year_idx (int): Year index when salary starts.
        salary_end_year_idx (int): Year index when salary ends (exclusive).
        mu_log_inflation (float): Log-normal mean for inflation.
        sigma_log_inflation (float): Log-normal std dev for inflation.
        rebalancing_trigger_year_idx (int): Year index for portfolio rebalancing.
        phase1_stocks_weight (float): Phase 1 stock weight.
        phase1_bonds_weight (float): Phase 1 bond weight.
        phase1_str_weight (float): Phase 1 short-term reserve weight.
        phase1_fun_weight (float): Phase 1 alternative asset weight.
        phase1_real_estate_weight (float): Phase 1 real estate weight.
        phase2_stocks_weight (float): Phase 2 stock weight.
        phase2_bonds_weight (float): Phase 2 bond weight.
        phase2_str_weight (float): Phase 2 short-term reserve weight.
        phase2_fun_weight (float): Phase 2 alternative asset weight.
        phase2_real_estate_weight (float): Phase 2 real estate weight.
        mu_log_stocks (float): Log-normal mean for stock returns.
        sigma_log_stocks (float): Log-normal std dev for stock returns.
        mu_log_bonds (float): Log-normal mean for bond returns.
        sigma_log_bonds (float): Log-normal std dev for bond returns.
        mu_log_str (float): Log-normal mean for STR returns.
        sigma_log_str (float): Log-normal std dev for STR returns.
        mu_log_fun (float): Log-normal mean for alternative asset returns.
        sigma_log_fun (float): Log-normal std dev for alternative asset returns.
        mu_log_real_estate (float): Log-normal mean for real estate returns.
        sigma_log_real_estate (float): Log-normal std dev for real estate returns.
        real_bank_lower_bound (float): Lower threshold for bank balance (real).
        real_bank_upper_bound (float): Upper threshold for bank balance (real).
        initial_real_monthly_contribution (float): Initial real monthly contribution to
            investments.
        initial_real_house_cost (float): Initial real cost of house to be purchased.
        ter_annual_percentage (float): Total Expense Ratio (TER) for investments.
        shock_events (list[dict[str, Any]]): List of dicts, each defining a market shock.

    Returns:
        SimulationRunResult: A TypedDict containing the results of the simulation.
    """
    current_bank_balance: float = det_inputs.b0
    (
        current_stocks_value,
        current_bonds_value,
        current_str_value,
        current_fun_value,
        current_real_estate_value,
    ) = calculate_initial_asset_values(
        det_inputs.i0,
        phase1_stocks_weight,
        phase1_bonds_weight,
        phase1_str_weight,
        phase1_fun_weight,
        phase1_real_estate_weight,
    )

    # Initialize current weights for all assets (Phase 1 weights initially)
    current_weights_stocks: float = phase1_stocks_weight
    current_weights_bonds: float = phase1_bonds_weight
    current_weights_str: float = phase1_str_weight
    current_weights_fun: float = phase1_fun_weight
    current_weights_real_estate: float = phase1_real_estate_weight

    # Initialize normalized weights for investment (from Phase 1 liquid weights)
    liquid_weights_sum: float = (
        phase1_stocks_weight
        + phase1_bonds_weight
        + phase1_str_weight
        + phase1_fun_weight
    )

    normalized_weights_stocks: float
    normalized_weights_bonds: float
    normalized_weights_str: float
    normalized_weights_fun: float

    if liquid_weights_sum > 0.0:
        normalized_weights_stocks = current_weights_stocks / liquid_weights_sum
        normalized_weights_bonds = current_weights_bonds / liquid_weights_sum
        normalized_weights_str = current_weights_str / liquid_weights_sum
        normalized_weights_fun = current_weights_fun / liquid_weights_sum
    else:
        normalized_weights_stocks = 0.0
        normalized_weights_bonds = 0.0
        normalized_weights_str = 0.0
        normalized_weights_fun = 0.0

    real_bank_lower_bound: float = det_inputs.real_bank_lower_bound
    real_bank_upper_bound: float = det_inputs.real_bank_upper_bound
    total_retirement_years = det_inputs.t_ret_years
    total_retirement_months: int = total_retirement_years * 12
    nominal_wealth_history: NDArray[np.float64] = np.zeros(
        total_retirement_months, dtype=np.float64
    )
    bank_balance_history: NDArray[np.float64] = np.zeros(
        total_retirement_months, dtype=np.float64
    )

    initial_real_monthly_expenses: float = det_inputs.x_real_monthly_initial
    planned_contributions: list[tuple[float, int]] = det_inputs.c_planned

    planned_extra_expenses: list[tuple[float, int]] = det_inputs.x_planned_extra

    initial_real_monthly_contribution: float = det_inputs.c_real_monthly_initial
    # Pydantic already handles the type conversion for planned_contributions
    planned_contributions: list[tuple[float, int]] = det_inputs.c_planned
    ter_annual_percentage: float = det_inputs.ter_annual_percentage

    initial_real_house_cost: float = det_inputs.h0_real_cost

    initial_real_monthly_pension: float = det_inputs.p_real_monthly
    pension_inflation_adjustment_factor: float = (
        det_inputs.pension_inflation_adjustment_factor
    )
    pension_start_year_idx: int = det_inputs.y_p_start_idx

    initial_real_monthly_salary: float = det_inputs.s_real_monthly
    salary_inflation_adjustment_factor: float = (
        det_inputs.salary_inflation_adjustment_factor
    )
    salary_start_year_idx: int = det_inputs.y_s_start_idx
    salary_end_year_idx: int = det_inputs.y_s_end_idx

    success: bool = True
    months_lasted: int = 0

    # Variables to store rebalancing and final allocations
    pre_rebalancing_allocations_nominal: dict[str, float] = {}
    pre_rebalancing_allocations_real: dict[str, float] = {}
    rebalancing_allocations_nominal: dict[str, float] = {}
    rebalancing_allocations_real: dict[str, float] = {}
    final_allocations_nominal: dict[str, float] = {}
    final_allocations_real: dict[str, float] = {}

    # Pre-generate full sequence of annual returns for each asset and inflation
    # for the entire retirement duration
    annual_inflations_sequence: NDArray[np.float64] = (
        np.random.lognormal(
            mu_log_inflation, sigma_log_inflation, total_retirement_years
        ).astype(np.float64)
        - 1.0
    )
    annual_stocks_returns_sequence: NDArray[np.float64] = (
        np.random.lognormal(
            mu_log_stocks, sigma_log_stocks, total_retirement_years
        ).astype(np.float64)
        - 1.0
    )
    annual_bonds_returns_sequence: NDArray[np.float64] = (
        np.random.lognormal(
            mu_log_bonds, sigma_log_bonds, total_retirement_years
        ).astype(np.float64)
        - 1.0
    )
    annual_str_returns_sequence: NDArray[np.float64] = (
        np.random.lognormal(mu_log_str, sigma_log_str, total_retirement_years).astype(
            np.float64
        )
        - 1.0
    )
    annual_fun_returns_sequence: NDArray[np.float64] = (
        np.random.lognormal(mu_log_fun, sigma_log_fun, total_retirement_years).astype(
            np.float64
        )
        - 1.0
    )
    annual_real_estate_returns_sequence: NDArray[np.float64] = (
        np.random.lognormal(
            mu_log_real_estate, sigma_log_real_estate, total_retirement_years
        ).astype(np.float64)
        - 1.0
    )

    # Apply SHOCK_EVENTS to the pre-generated annual sequences
    for shock in shock_events:
        shock_year: int = shock["year"]
        shock_asset: str = shock["asset"]
        shock_magnitude: float = shock["magnitude"]

        if 0 <= shock_year < total_retirement_years:
            if shock_asset == "Stocks":
                annual_stocks_returns_sequence[shock_year] = shock_magnitude
            elif shock_asset == "Bonds":
                annual_bonds_returns_sequence[shock_year] = shock_magnitude
            elif shock_asset == "STR":
                annual_str_returns_sequence[shock_year] = shock_magnitude
            elif shock_asset == "Fun":
                annual_fun_returns_sequence[shock_year] = shock_magnitude
            elif shock_asset == "Real Estate":
                annual_real_estate_returns_sequence[shock_year] = shock_magnitude
            elif shock_asset == "Inflation":
                annual_inflations_sequence[shock_year] = shock_magnitude

    # --- OPTIMIZATION 1: Pre-calculate cumulative inflation factors ---
    cumulative_inflation_factors_annual: NDArray[np.float64] = np.ones(
        total_retirement_years + 1, dtype=np.float64
    )
    for year_idx in range(total_retirement_years):
        cumulative_inflation_factors_annual[year_idx + 1] = float(
            cumulative_inflation_factors_annual[year_idx]
            * (1.0 + annual_inflations_sequence[year_idx])
        )

    # --- OPTIMIZATION 2: Pre-calculate all monthly returns for all assets ---
    monthly_returns_lookup: dict[str, NDArray[np.float64]] = {
        "Stocks": np.zeros(total_retirement_months, dtype=np.float64),
        "Bonds": np.zeros(total_retirement_months, dtype=np.float64),
        "STR": np.zeros(total_retirement_months, dtype=np.float64),
        "Fun": np.zeros(total_retirement_months, dtype=np.float64),
        "Real Estate": np.zeros(total_retirement_months, dtype=np.float64),
    }

    for year_idx in range(total_retirement_years):
        monthly_stocks_rate: float = annual_to_monthly_compounded_rate(
            annual_stocks_returns_sequence[year_idx]
        )
        monthly_bonds_rate: float = annual_to_monthly_compounded_rate(
            annual_bonds_returns_sequence[year_idx]
        )
        monthly_str_rate: float = annual_to_monthly_compounded_rate(
            annual_str_returns_sequence[year_idx]
        )
        monthly_fun_rate: float = annual_to_monthly_compounded_rate(
            annual_fun_returns_sequence[year_idx]
        )
        monthly_real_estate_rate: float = annual_to_monthly_compounded_rate(
            annual_real_estate_returns_sequence[year_idx]
        )

        start_month: int = year_idx * 12
        end_month: int = min((year_idx + 1) * 12, total_retirement_months)

        monthly_returns_lookup["Stocks"][start_month:end_month] = monthly_stocks_rate
        monthly_returns_lookup["Bonds"][start_month:end_month] = monthly_bonds_rate
        monthly_returns_lookup["STR"][start_month:end_month] = monthly_str_rate
        monthly_returns_lookup["Fun"][start_month:end_month] = monthly_fun_rate
        monthly_returns_lookup["Real Estate"][start_month:end_month] = (
            monthly_real_estate_rate
        )

    # --- OPTIMIZATION 3: Pre-calculate nominal planned contributions/expenses ---
    nominal_planned_contributions_amounts: list[tuple[float, int]] = []
    # Note: `planned_contributions` itself is `list[tuple[float, int]]`
    for real_amount, year_idx in planned_contributions:
        nominal_contribution_amount: float = float(
            real_amount * cumulative_inflation_factors_annual[year_idx]
        )
        nominal_planned_contributions_amounts.append(
            (nominal_contribution_amount, year_idx)
        )

    nominal_planned_extra_expenses_amounts: list[tuple[float, int]] = []
    # Note: `planned_extra_expenses` itself is `list[tuple[float, int]]`
    local_planned_extra_expenses: list[tuple[float, int]] = list(planned_extra_expenses)
    for real_amount, year_idx in local_planned_extra_expenses:
        nominal_extra_expense_amount: float = float(
            real_amount * cumulative_inflation_factors_annual[year_idx]
        )
        nominal_planned_extra_expenses_amounts.append(
            (nominal_extra_expense_amount, year_idx)
        )

    # --- OPTIMIZATION 4: Pre-calculate nominal pension/salary per year ---
    nominal_pension_annual_sequence: NDArray[np.float64] = np.zeros(
        total_retirement_years, dtype=np.float64
    )
    nominal_salary_annual_sequence: NDArray[np.float64] = np.zeros(
        total_retirement_years, dtype=np.float64
    )

    for year_idx in range(total_retirement_years):
        if year_idx >= pension_start_year_idx:
            # Calculate the pension adjustment factor
            pension_factor: float
            if year_idx > pension_start_year_idx:
                pension_adjusted_inflations: NDArray[np.float64] = (
                    annual_inflations_sequence[pension_start_year_idx:year_idx]
                    * pension_inflation_adjustment_factor
                )
                pension_factor = float(np.prod(1.0 + pension_adjusted_inflations))
            else:
                pension_factor = 1.0

            # This uses the nominal monthly pension amount based on inflation up to pension_start_year_idx
            # and then adjusted for subsequent inflation and adjustment factor.
            # This logic appears correct for monthly income already calculated into an "annual sequence"
            # where each element is the monthly nominal value for that year.
            nominal_pension_annual_sequence[year_idx] = float(
                initial_real_monthly_pension
                * cumulative_inflation_factors_annual[pension_start_year_idx]
                * pension_factor
            )

        if salary_start_year_idx <= year_idx < salary_end_year_idx:
            # Calculate the salary adjustment factor
            salary_factor: float
            if year_idx > salary_start_year_idx:
                salary_adjusted_inflations: NDArray[np.float64] = (
                    annual_inflations_sequence[salary_start_year_idx:year_idx]
                    * salary_inflation_adjustment_factor
                )
                salary_factor = float(np.prod(1.0 + salary_adjusted_inflations))
            else:
                salary_factor = 1.0

            # This uses the nominal monthly salary amount based on inflation up to salary_start_year_idx
            # and then adjusted for subsequent inflation and adjustment factor.
            # This logic appears correct for monthly income already calculated into an "annual sequence"
            # where each element is the monthly nominal value for that year.
            nominal_salary_annual_sequence[year_idx] = float(
                initial_real_monthly_salary
                * cumulative_inflation_factors_annual[salary_start_year_idx]
                * salary_factor
            )

    ter_monthly_factor: float = ter_annual_percentage / 12.0

    # Simulation loop
    for current_month_idx in range(total_retirement_months):
        months_lasted += 1
        current_year_idx: int = current_month_idx // 12
        month_in_year_idx: int = current_month_idx % 12

        # --- OPTIMIZATION: Determine current weights once per year ---
        if month_in_year_idx == 0:
            if current_year_idx < rebalancing_trigger_year_idx:
                current_weights_stocks = phase1_stocks_weight
                current_weights_bonds = phase1_bonds_weight
                current_weights_str = phase1_str_weight
                current_weights_fun = phase1_fun_weight
                current_weights_real_estate = phase1_real_estate_weight
            else:
                current_weights_stocks = phase2_stocks_weight
                current_weights_bonds = phase2_bonds_weight
                current_weights_str = phase2_str_weight
                current_weights_fun = phase2_fun_weight
                current_weights_real_estate = phase2_real_estate_weight

            liquid_weights_sum = (
                current_weights_stocks
                + current_weights_bonds
                + current_weights_str
                + current_weights_fun
            )
            normalized_weights_stocks = (
                current_weights_stocks / liquid_weights_sum
                if liquid_weights_sum > 0.0
                else 0.0
            )
            normalized_weights_bonds = (
                current_weights_bonds / liquid_weights_sum
                if liquid_weights_sum > 0.0
                else 0.0
            )
            normalized_weights_str = (
                current_weights_str / liquid_weights_sum
                if liquid_weights_sum > 0.0
                else 0.0
            )
            normalized_weights_fun = (
                current_weights_fun / liquid_weights_sum
                if liquid_weights_sum > 0.0
                else 0.0
            )

        # 1. Add pension if applicable
        nominal_pension_monthly: float = 0.0
        if current_year_idx >= pension_start_year_idx:
            # Using the pre-calculated nominal monthly amount directly (no / 12.0)
            nominal_pension_monthly = nominal_pension_annual_sequence[current_year_idx]
            current_bank_balance += nominal_pension_monthly

        # 2. Add monthly salary if applicable
        nominal_salary_monthly: float = 0.0
        if salary_start_year_idx <= current_year_idx < salary_end_year_idx:
            # Using the pre-calculated nominal monthly amount directly (no / 12.0)
            nominal_salary_monthly = nominal_salary_annual_sequence[current_year_idx]
            current_bank_balance += nominal_salary_monthly

        # --- Check and Top-Up Bank Account if below REAL_BANK_LOWER_BOUND ---
        monthly_inflation_rate_this_year: float = annual_to_monthly_compounded_rate(
            annual_inflations_sequence[current_year_idx]
        )
        cumulative_inflation_factor_up_to_current_month: float = float(
            cumulative_inflation_factors_annual[current_year_idx]
            * ((1.0 + monthly_inflation_rate_this_year) ** (month_in_year_idx + 1.0))
        )

        current_real_bank_balance: float = (
            current_bank_balance / cumulative_inflation_factor_up_to_current_month
        )

        if current_real_bank_balance < real_bank_lower_bound:
            real_shortfall_to_cover: float = (
                real_bank_lower_bound - current_real_bank_balance
            )
            nominal_top_up_amount: float = float(
                real_shortfall_to_cover
                * cumulative_inflation_factor_up_to_current_month
            )

            amount_to_liquidate_for_top_up: float = nominal_top_up_amount

            # Withdraw from STR (priority 1)
            if amount_to_liquidate_for_top_up > 0.0:
                if current_str_value >= amount_to_liquidate_for_top_up:
                    current_str_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0.0
                else:
                    current_bank_balance += current_str_value
                    amount_to_liquidate_for_top_up -= current_str_value
                    current_str_value = 0.0

            # Withdraw from Bonds (priority 2)
            if amount_to_liquidate_for_top_up > 0.0:
                if current_bonds_value >= amount_to_liquidate_for_top_up:
                    current_bonds_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0.0
                else:
                    current_bank_balance += current_bonds_value
                    amount_to_liquidate_for_top_up -= current_bonds_value
                    current_bonds_value = 0.0

            # Withdraw from Stocks (priority 3)
            if amount_to_liquidate_for_top_up > 0.0:
                if current_stocks_value >= amount_to_liquidate_for_top_up:
                    current_stocks_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0.0
                else:
                    current_bank_balance += current_stocks_value
                    amount_to_liquidate_for_top_up -= current_stocks_value
                    current_stocks_value = 0.0

            # Withdraw from Fun Money (priority 4)
            if amount_to_liquidate_for_top_up > 0.0:
                if current_fun_value >= amount_to_liquidate_for_top_up:
                    current_fun_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0.0
                else:
                    current_bank_balance += current_fun_value
                    amount_to_liquidate_for_top_up -= current_fun_value
                    current_fun_value = 0.0

            # CRITICAL FAILURE CHECK 1: If still needed after all liquid assets
            #  (excluding Real Estate)
            if amount_to_liquidate_for_top_up > 0.0:
                success = False
                break  # Simulation failed, exit loop

        # Re-calculate current_real_bank_balance in case it was topped up
        current_real_bank_balance = (
            current_bank_balance / cumulative_inflation_factor_up_to_current_month
        )

        if current_real_bank_balance > real_bank_upper_bound:
            real_excess_to_invest: float = (
                current_real_bank_balance - real_bank_upper_bound
            )
            nominal_excess_to_invest: float = float(
                real_excess_to_invest * cumulative_inflation_factor_up_to_current_month
            )

            if liquid_weights_sum > 0.0:
                current_stocks_value += (
                    nominal_excess_to_invest * normalized_weights_stocks
                )
                current_bonds_value += (
                    nominal_excess_to_invest * normalized_weights_bonds
                )
                current_str_value += nominal_excess_to_invest * normalized_weights_str
                current_fun_value += nominal_excess_to_invest * normalized_weights_fun
                current_bank_balance -= nominal_excess_to_invest

        # 2. Handle planned contributions
        planned_contribution_applied_for_year: bool = False
        for i, (nominal_contribution_amount, contribution_year_idx) in enumerate(
            planned_contributions
        ):
            if (
                current_year_idx == contribution_year_idx
                and month_in_year_idx == 0
                and not planned_contribution_applied_for_year
            ):
                current_stocks_value += (
                    nominal_contribution_amount * current_weights_stocks
                )
                current_bonds_value += (
                    nominal_contribution_amount * current_weights_bonds
                )
                current_str_value += nominal_contribution_amount * current_weights_str
                current_fun_value += nominal_contribution_amount * current_weights_fun
                current_real_estate_value += (
                    nominal_contribution_amount * current_weights_real_estate
                )
                planned_contribution_applied_for_year = True

        # Apply initial_real_monthly_contribution every month
        if initial_real_monthly_contribution > 0.0:
            nominal_monthly_contribution: float = float(
                initial_real_monthly_contribution
                * cumulative_inflation_factors_annual[current_year_idx]
            )
            current_stocks_value += (
                nominal_monthly_contribution * current_weights_stocks
            )
            current_bonds_value += nominal_monthly_contribution * current_weights_bonds
            current_str_value += nominal_monthly_contribution * current_weights_str
            current_fun_value += nominal_monthly_contribution * current_weights_fun
            current_real_estate_value += (
                nominal_monthly_contribution * current_weights_real_estate
            )

        # 3. Calculate nominal monthly withdrawal amount (includes initial_real_monthly_expenses)
        nominal_monthly_expenses: float = float(
            initial_real_monthly_expenses
            * cumulative_inflation_factors_annual[current_year_idx]
        )

        # Add planned extra expenses for this month/year
        extra_expense_for_this_month: float = 0.0
        expenses_to_remove_indices: list[int] = []

        for i, (nominal_extra_expense_amount, expense_year_idx) in enumerate(
            local_planned_extra_expenses
        ):
            if current_year_idx == expense_year_idx and month_in_year_idx == 0:
                extra_expense_for_this_month += nominal_extra_expense_amount
                expenses_to_remove_indices.append(i)

        for idx in sorted(expenses_to_remove_indices, reverse=True):
            local_planned_extra_expenses.pop(idx)

        # 4. Process monthly withdrawals (Prioritized)
        withdrawal_needed: float = (
            nominal_monthly_expenses + extra_expense_for_this_month
        )

        if current_bank_balance >= withdrawal_needed:
            current_bank_balance -= withdrawal_needed
            withdrawal_needed = 0.0
        else:
            withdrawal_needed -= current_bank_balance
            current_bank_balance = 0.0

        if withdrawal_needed > 0.0:
            if current_str_value >= withdrawal_needed:
                current_str_value -= withdrawal_needed
                withdrawal_needed = 0.0
            else:
                withdrawal_needed -= current_str_value
                current_str_value = 0.0

        if withdrawal_needed > 0.0:
            if current_bonds_value >= withdrawal_needed:
                current_bonds_value -= withdrawal_needed
                withdrawal_needed = 0.0
            else:
                withdrawal_needed -= current_bonds_value
                current_bonds_value = 0.0

        if withdrawal_needed > 0.0:
            if current_stocks_value >= withdrawal_needed:
                current_stocks_value -= withdrawal_needed
                withdrawal_needed = 0.0
            else:
                withdrawal_needed -= current_stocks_value
                current_stocks_value = 0.0

        if withdrawal_needed > 0.0:
            if current_fun_value >= withdrawal_needed:
                current_fun_value -= withdrawal_needed
                withdrawal_needed = 0.0
            else:
                withdrawal_needed -= current_fun_value
                current_fun_value = 0.0

        # CRITICAL FAILURE CHECK 2: If still needed after all liquid assets (excluding Real Estate)
        if withdrawal_needed > 0.0:
            success = False
            break  # Exit simulation early if plan fails

        # 5. Apply monthly returns to investments (at end of month)
        current_stocks_value *= (
            1.0 + monthly_returns_lookup["Stocks"][current_month_idx]
        )
        current_bonds_value *= 1.0 + monthly_returns_lookup["Bonds"][current_month_idx]
        current_str_value *= 1.0 + monthly_returns_lookup["STR"][current_month_idx]
        current_fun_value *= 1.0 + monthly_returns_lookup["Fun"][current_month_idx]
        current_real_estate_value *= (
            1.0 + monthly_returns_lookup["Real Estate"][current_month_idx]
        )

        # Apply TER
        current_stocks_value *= 1.0 - ter_monthly_factor
        current_bonds_value *= 1.0 - ter_monthly_factor
        current_str_value *= 1.0 - ter_monthly_factor
        current_fun_value *= 1.0 - ter_monthly_factor

        # Ensure no asset values drop below zero due to fees
        current_stocks_value = max(0.0, current_stocks_value)
        current_bonds_value = max(0.0, current_bonds_value)
        current_str_value = max(0.0, current_str_value)
        current_fun_value = max(0.0, current_fun_value)

        # 6. Rebalance at the start of REBALANCING_TRIGGER_YEAR_IDX
        if (
            current_year_idx == rebalancing_trigger_year_idx
            and month_in_year_idx == 0
            and rebalancing_trigger_year_idx > 0
        ):
            total_investment_value_pre_rebalance: float = (
                current_stocks_value
                + current_bonds_value
                + current_str_value
                + current_fun_value
                + current_real_estate_value
            )

            cumulative_inflation_rebalance_year: np.float64 = (
                cumulative_inflation_factors_annual[rebalancing_trigger_year_idx]
            )

            pre_rebalancing_allocations_nominal = {
                "Stocks": current_stocks_value,
                "Bonds": current_bonds_value,
                "STR": current_str_value,
                "Fun": current_fun_value,
                "Real Estate": current_real_estate_value,
            }
            pre_rebalancing_allocations_real = {
                "Stocks": float(
                    current_stocks_value / cumulative_inflation_rebalance_year
                ),
                "Bonds": float(
                    current_bonds_value / cumulative_inflation_rebalance_year
                ),
                "STR": float(current_str_value / cumulative_inflation_rebalance_year),
                "Fun": float(current_fun_value / cumulative_inflation_rebalance_year),
                "Real Estate": float(
                    current_real_estate_value / cumulative_inflation_rebalance_year
                ),
            }

            # --- START HOUSE PURCHASE LOGIC ---
            if initial_real_house_cost > 0.0:
                nominal_house_cost: float = float(
                    initial_real_house_cost
                    * cumulative_inflation_factors_annual[rebalancing_trigger_year_idx]
                )
                liquid_assets_pre_house: float = (
                    current_str_value
                    + current_bonds_value
                    + current_stocks_value
                    + current_fun_value
                )

                if liquid_assets_pre_house < nominal_house_cost:
                    success = False
                    break  # Exit loop

                remaining_to_buy: float = nominal_house_cost

                if current_str_value >= remaining_to_buy:
                    current_str_value -= remaining_to_buy
                    remaining_to_buy = 0.0
                else:
                    remaining_to_buy -= current_str_value
                    current_str_value = 0.0

                if remaining_to_buy > 0.0:
                    if current_bonds_value >= remaining_to_buy:
                        current_bonds_value -= remaining_to_buy
                        remaining_to_buy = 0.0
                    else:
                        remaining_to_buy -= current_bonds_value
                        current_bonds_value = 0.0

                if remaining_to_buy > 0.0:
                    if current_stocks_value >= remaining_to_buy:
                        current_stocks_value -= remaining_to_buy
                        remaining_to_buy = 0.0
                    else:
                        remaining_to_buy -= current_stocks_value
                        current_stocks_value = 0.0

                if remaining_to_buy > 0.0:
                    if current_fun_value >= remaining_to_buy:
                        current_fun_value -= remaining_to_buy
                        remaining_to_buy = 0.0
                    else:
                        remaining_to_buy -= current_fun_value
                        current_fun_value = 0.0

                current_real_estate_value += nominal_house_cost

            # --- END HOUSE PURCHASE LOGIC ---

            # Recalculate total_investment_value_pre_rebalance AFTER potential house purchase
            total_investment_value_pre_rebalance = (
                current_stocks_value
                + current_bonds_value
                + current_str_value
                + current_fun_value
                + current_real_estate_value
            )

            # --- START CONDITIONAL REBALANCING LOGIC ---
            if initial_real_house_cost > 0.0:
                liquid_portfolio_value_for_rebalance: float = (
                    total_investment_value_pre_rebalance - current_real_estate_value
                )
                sum_liquid_p2_weights: float = (
                    phase2_stocks_weight
                    + phase2_bonds_weight
                    + phase2_str_weight
                    + phase2_fun_weight
                )

                if sum_liquid_p2_weights == 0.0:
                    current_stocks_value = 0.0
                    current_bonds_value = 0.0
                    current_str_value = 0.0
                    current_fun_value = 0.0
                else:
                    normalized_weights_phase2_stocks: float = (
                        phase2_stocks_weight / sum_liquid_p2_weights
                    )
                    normalized_weights_phase2_bonds: float = (
                        phase2_bonds_weight / sum_liquid_p2_weights
                    )
                    normalized_weights_phase2_str: float = (
                        phase2_str_weight / sum_liquid_p2_weights
                    )
                    normalized_weights_phase2_fun: float = (
                        phase2_fun_weight / sum_liquid_p2_weights
                    )

                    current_stocks_value = (
                        liquid_portfolio_value_for_rebalance
                        * normalized_weights_phase2_stocks
                    )
                    current_bonds_value = (
                        liquid_portfolio_value_for_rebalance
                        * normalized_weights_phase2_bonds
                    )
                    current_str_value = (
                        liquid_portfolio_value_for_rebalance
                        * normalized_weights_phase2_str
                    )
                    current_fun_value = (
                        liquid_portfolio_value_for_rebalance
                        * normalized_weights_phase2_fun
                    )

                rebalancing_allocations_nominal = {
                    "Stocks": current_stocks_value,
                    "Bonds": current_bonds_value,
                    "STR": current_str_value,
                    "Fun": current_fun_value,
                    "Real Estate": current_real_estate_value,
                }
            else:
                rebalancing_allocations_nominal = {
                    "Stocks": total_investment_value_pre_rebalance
                    * phase2_stocks_weight,
                    "Bonds": total_investment_value_pre_rebalance * phase2_bonds_weight,
                    "STR": total_investment_value_pre_rebalance * phase2_str_weight,
                    "Fun": total_investment_value_pre_rebalance * phase2_fun_weight,
                    "Real Estate": total_investment_value_pre_rebalance
                    * phase2_real_estate_weight,
                }

                current_stocks_value = rebalancing_allocations_nominal["Stocks"]
                current_bonds_value = rebalancing_allocations_nominal["Bonds"]
                current_str_value = rebalancing_allocations_nominal["STR"]
                current_fun_value = rebalancing_allocations_nominal["Fun"]
                current_real_estate_value = rebalancing_allocations_nominal[
                    "Real Estate"
                ]

            # --- END CONDITIONAL REBALANCING LOGIC ---

            inflation_factor: np.float64 = cumulative_inflation_rebalance_year
            rebalancing_allocations_real = {
                "Stocks": float(
                    rebalancing_allocations_nominal["Stocks"] / inflation_factor
                ),
                "Bonds": float(
                    rebalancing_allocations_nominal["Bonds"] / inflation_factor
                ),
                "STR": float(rebalancing_allocations_nominal["STR"] / inflation_factor),
                "Fun": float(rebalancing_allocations_nominal["Fun"] / inflation_factor),
                "Real Estate": float(
                    rebalancing_allocations_nominal["Real Estate"] / inflation_factor
                ),
            }

        # Record nominal wealth at the end of the month (only for successful months)
        nominal_wealth_history[current_month_idx] = float(
            current_bank_balance
            + current_stocks_value
            + current_bonds_value
            + current_str_value
            + current_fun_value
            + current_real_estate_value
        )
        bank_balance_history[current_month_idx] = current_bank_balance

    # --- MODIFIED: Handle history for failed simulations by filling with last value ---
    if not success:
        # Fill the remaining part of the history arrays with NaN for failed simulations.
        # This prevents misleading averages or plots for failed scenarios.
        if months_lasted > 0:  # Ensure at least one month was processed before failure
            # Fill from the failing month's index onwards
            nominal_wealth_history[months_lasted - 1 :] = np.nan
            bank_balance_history[months_lasted - 1 :] = np.nan
        else:  # Simulation failed in the very first month (months_lasted = 0)
            # The arrays would already be all zeros from initialization.
            # Fill entire array with NaN if it failed before any values were recorded.
            nominal_wealth_history[:] = np.nan
            bank_balance_history[:] = np.nan

    final_investment_value: float = (
        current_stocks_value
        + current_bonds_value
        + current_str_value
        + current_fun_value
        + current_real_estate_value
    )
    final_bank_balance: float = current_bank_balance

    final_allocations_nominal: dict[str, float] = {
        "Stocks": current_stocks_value,
        "Bonds": current_bonds_value,
        "STR": current_str_value,
        "Fun": current_fun_value,
        "Real Estate": current_real_estate_value,
    }

    cumulative_inflation_end_of_sim: np.float64 = cumulative_inflation_factors_annual[
        total_retirement_years
    ]
    final_allocations_real: dict[str, float] = {
        "Stocks": float(current_stocks_value / cumulative_inflation_end_of_sim),
        "Bonds": float(current_bonds_value / cumulative_inflation_end_of_sim),
        "STR": float(current_str_value / cumulative_inflation_end_of_sim),
        "Fun": float(current_fun_value / cumulative_inflation_end_of_sim),
        "Real Estate": float(
            current_real_estate_value / cumulative_inflation_end_of_sim
        ),
    }

    return SimulationRunResult(
        success=success,
        months_lasted=months_lasted,
        final_investment=final_investment_value,
        final_bank_balance=final_bank_balance,
        annual_inflations_seq=annual_inflations_sequence,
        nominal_wealth_history=nominal_wealth_history,
        bank_balance_history=bank_balance_history,
        pre_rebalancing_allocations_nominal=pre_rebalancing_allocations_nominal,
        pre_rebalancing_allocations_real=pre_rebalancing_allocations_real,
        rebalancing_allocations_nominal=rebalancing_allocations_nominal,
        rebalancing_allocations_real=rebalancing_allocations_real,
        final_allocations_nominal=final_allocations_nominal,
        final_allocations_real=final_allocations_real,
    )
