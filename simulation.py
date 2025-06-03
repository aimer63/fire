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

# Import helper functions
from helpers import annual_to_monthly_compounded_rate, calculate_initial_asset_values


def run_single_fire_simulation(
    i0,
    b0,
    t_ret_months,
    t_ret_years,
    x_real_monthly_initial,
    c_planned,
    x_planned_extra,
    p_real_monthly,
    pension_inflation_adjustment_factor,
    y_p_start_idx,
    s_real_monthly,
    salary_inflation_adjustment_factor,
    y_s_start_idx,
    y_s_end_idx,
    mu_log_pi,
    sigma_log_pi,
    rebalancing_year_idx,
    w_p1_stocks,
    w_p1_bonds,
    w_p1_str,
    w_p1_fun,
    w_p1_real_estate,
    w_p2_stocks,
    w_p2_bonds,
    w_p2_str,
    w_p2_fun,
    w_p2_real_estate,
    mu_log_stocks,
    sigma_log_stocks,
    mu_log_bonds,
    sigma_log_bonds,
    mu_log_str,
    sigma_log_str,
    mu_log_fun,
    sigma_log_fun,
    mu_log_real_estate,
    sigma_log_real_estate,
    real_bank_lower_bound,
    real_bank_upper_bound,
    c_real_monthly_initial,
    h0_real_cost,
    ter_annual_percentage,
    shock_events,
):
    """
    Runs a single Monte Carlo simulation for a Financial Independence/Retirement plan.
    Simulates wealth evolution, income, expenses, and asset allocations over time.

    Args:
        i0 (float): Initial investment value.
        b0 (float): Initial bank account balance.
        t_ret_months (int): Total simulation duration in months.
        t_ret_years (int): Total simulation duration in years.
        x_real_monthly_initial (float): Initial real monthly expenses.
        c_planned (list): List of [amount, year_idx] for planned contributions.
        x_planned_extra (list): List of [amount, year_idx] for planned extra
            expenses.
        p_real_monthly (float): Initial real monthly pension.
        pension_inflation_adjustment_factor (float): Factor for pension inflation
            adjustment.
        y_p_start_idx (int): Year index when pension starts.
        s_real_monthly (float): Initial real monthly salary.
        salary_inflation_adjustment_factor (float): Factor for salary inflation
            adjustment.
        y_s_start_idx (int): Year index when salary starts.
        y_s_end_idx (int): Year index when salary ends (exclusive).
        mu_log_pi (float): Log-normal mean for inflation.
        sigma_log_pi (float): Log-normal std dev for inflation.
        rebalancing_year_idx (int): Year index for portfolio rebalancing.
        w_p1_stocks (float): Phase 1 stock weight.
        w_p1_bonds (float): Phase 1 bond weight.
        w_p1_str (float): Phase 1 short-term reserve weight.
        w_p1_fun (float): Phase 1 alternative asset weight.
        w_p1_real_estate (float): Phase 1 real estate weight.
        w_p2_stocks (float): Phase 2 stock weight.
        w_p2_bonds (float): Phase 2 bond weight.
        w_p2_str (float): Phase 2 short-term reserve weight.
        w_p2_fun (float): Phase 2 alternative asset weight.
        w_p2_real_estate (float): Phase 2 real estate weight.
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
        c_real_monthly_initial (float): Initial real monthly contribution to
            investments.
        h0_real_cost (float): Initial real cost of house to be purchased.
        ter_annual_percentage (float): Total Expense Ratio (TER) for investments.
        shock_events (list): List of dicts, each defining a market shock.

    Returns:
        tuple: A tuple containing the results of the simulation:
            - success (bool): True if the simulation lasted for t_ret_months.
            - months_lasted (int): Number of months the simulation lasted.
            - final_investment (float): Final nominal investment value.
            - final_bank_balance (float): Final nominal bank balance.
            - annual_inflations_seq (list): Sequence of annual inflation rates.
            - nominal_wealth_history (np.array): Total nominal wealth history.
            - bank_balance_history (np.array): Nominal bank balance history.
            - pre_rebalancing_allocations_nominal (dict): Nominal allocation
              before rebalancing (at rebalancing_year_idx).
            - pre_rebalancing_allocations_real (dict): Real allocation
              before rebalancing (at rebalancing_year_idx).
            - rebalancing_allocations_nominal (dict): Nominal allocation
              after rebalancing (at rebalancing_year_idx).
            - rebalancing_allocations_real (dict): Real allocation
              after rebalancing (at rebalancing_year_idx).
            - final_allocations_nominal (dict): Final nominal asset allocations.
            - final_allocations_real (dict): Final real asset allocations.
    """
    current_bank_balance = b0
    (
        current_stocks_value,
        current_bonds_value,
        current_str_value,
        current_fun_value,
        current_real_estate_value,
    ) = calculate_initial_asset_values(
        i0, w_p1_stocks, w_p1_bonds, w_p1_str, w_p1_fun, w_p1_real_estate
    )

    # Initialize current weights for all assets (Phase 1 weights initially)
    current_weights_stocks = w_p1_stocks
    current_weights_bonds = w_p1_bonds
    current_weights_str = w_p1_str
    current_weights_fun = w_p1_fun
    current_weights_real_estate = w_p1_real_estate

    # Initialize normalized weights for investment (from Phase 1 liquid weights)
    liquid_weights_sum = w_p1_stocks + w_p1_bonds + w_p1_str + w_p1_fun

    if liquid_weights_sum > 0:
        normalized_weights_stocks = w_p1_stocks / liquid_weights_sum
        normalized_weights_bonds = w_p1_bonds / liquid_weights_sum
        normalized_weights_str = w_p1_str / liquid_weights_sum
        normalized_weights_fun = w_p1_fun / liquid_weights_sum
    else:
        normalized_weights_stocks = 0.0
        normalized_weights_bonds = 0.0
        normalized_weights_str = 0.0
        normalized_weights_fun = 0.0

    nominal_wealth_history = np.zeros(t_ret_months)
    bank_balance_history = np.zeros(t_ret_months)

    success = True
    months_lasted = 0

    # Variables to store rebalancing and final allocations
    pre_rebalancing_allocations_nominal = {}
    pre_rebalancing_allocations_real = {}
    rebalancing_allocations_nominal = {}
    rebalancing_allocations_real = {}
    final_allocations_nominal = {}
    final_allocations_real = {}

    # Pre-generate full sequence of annual returns for each asset and inflation
    # for the entire retirement duration
    annual_inflations_sequence = (
        np.random.lognormal(mu_log_pi, sigma_log_pi, t_ret_years) - 1
    )
    annual_stocks_returns_sequence = (
        np.random.lognormal(mu_log_stocks, sigma_log_stocks, t_ret_years) - 1
    )
    annual_bonds_returns_sequence = (
        np.random.lognormal(mu_log_bonds, sigma_log_bonds, t_ret_years) - 1
    )
    annual_str_returns_sequence = (
        np.random.lognormal(mu_log_str, sigma_log_str, t_ret_years) - 1
    )
    annual_fun_returns_sequence = (
        np.random.lognormal(mu_log_fun, sigma_log_fun, t_ret_years) - 1
    )
    annual_real_estate_returns_sequence = (
        np.random.lognormal(mu_log_real_estate, sigma_log_real_estate, t_ret_years) - 1
    )

    # Apply SHOCK_EVENTS to the pre-generated annual sequences
    for shock in shock_events:
        shock_year = shock["year"]
        shock_asset = shock["asset"]
        shock_magnitude = shock["magnitude"]

        if 0 <= shock_year < t_ret_years:
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
    cumulative_inflation_factors_annual = np.ones(t_ret_years + 1)
    for year_idx in range(t_ret_years):
        cumulative_inflation_factors_annual[year_idx + 1] = (
            cumulative_inflation_factors_annual[year_idx]
            * (1 + annual_inflations_sequence[year_idx])
        )

    # --- OPTIMIZATION 2: Pre-calculate all monthly returns for all assets ---
    monthly_returns_lookup = {
        "Stocks": np.zeros(t_ret_months),
        "Bonds": np.zeros(t_ret_months),
        "STR": np.zeros(t_ret_months),
        "Fun": np.zeros(t_ret_months),
        "Real Estate": np.zeros(t_ret_months),
    }

    for year_idx in range(t_ret_years):
        monthly_stocks_rate = annual_to_monthly_compounded_rate(
            annual_stocks_returns_sequence[year_idx]
        )
        monthly_bonds_rate = annual_to_monthly_compounded_rate(
            annual_bonds_returns_sequence[year_idx]
        )
        monthly_str_rate = annual_to_monthly_compounded_rate(
            annual_str_returns_sequence[year_idx]
        )
        monthly_fun_rate = annual_to_monthly_compounded_rate(
            annual_fun_returns_sequence[year_idx]
        )
        monthly_real_estate_rate = annual_to_monthly_compounded_rate(
            annual_real_estate_returns_sequence[year_idx]
        )

        start_month = year_idx * 12
        end_month = min((year_idx + 1) * 12, t_ret_months)

        monthly_returns_lookup["Stocks"][start_month:end_month] = monthly_stocks_rate
        monthly_returns_lookup["Bonds"][start_month:end_month] = monthly_bonds_rate
        monthly_returns_lookup["STR"][start_month:end_month] = monthly_str_rate
        monthly_returns_lookup["Fun"][start_month:end_month] = monthly_fun_rate
        monthly_returns_lookup["Real Estate"][start_month:end_month] = (
            monthly_real_estate_rate
        )

    # --- OPTIMIZATION 3: Pre-calculate nominal planned contributions/expenses ---
    nominal_planned_contributions_amounts = []
    for real_amount, year_idx in c_planned:
        nominal_contribution_amount = (
            real_amount * cumulative_inflation_factors_annual[year_idx]
        )
        nominal_planned_contributions_amounts.append(
            (nominal_contribution_amount, year_idx)
        )

    nominal_planned_extra_expenses_amounts = []
    local_planned_extra_expenses = list(x_planned_extra)
    for real_amount, year_idx in local_planned_extra_expenses:
        nominal_extra_expense_amount = (
            real_amount * cumulative_inflation_factors_annual[year_idx]
        )
        nominal_planned_extra_expenses_amounts.append(
            (nominal_extra_expense_amount, year_idx)
        )

    # --- OPTIMIZATION 4: Pre-calculate nominal pension/salary per year ---
    nominal_pension_annual_sequence = np.zeros(t_ret_years)
    nominal_salary_annual_sequence = np.zeros(t_ret_years)

    for year_idx in range(t_ret_years):
        if year_idx >= y_p_start_idx:
            # Calculate the pension adjustment factor
            if year_idx > y_p_start_idx:
                pension_adjusted_inflations = (
                    annual_inflations_sequence[y_p_start_idx:year_idx]
                    * pension_inflation_adjustment_factor
                )
                pension_factor = np.prod(1 + pension_adjusted_inflations)
            else:
                pension_factor = 1.0

            nominal_pension_annual_sequence[year_idx] = (
                p_real_monthly
                * cumulative_inflation_factors_annual[y_p_start_idx]
                * pension_factor
            )

        if y_s_start_idx <= year_idx < y_s_end_idx:
            # Calculate the salary adjustment factor
            if year_idx > y_s_start_idx:
                salary_adjusted_inflations = (
                    annual_inflations_sequence[y_s_start_idx:year_idx]
                    * salary_inflation_adjustment_factor
                )
                salary_factor = np.prod(1 + salary_adjusted_inflations)
            else:
                salary_factor = 1.0

            nominal_salary_annual_sequence[year_idx] = (
                s_real_monthly
                * cumulative_inflation_factors_annual[y_s_start_idx]
                * salary_factor
            )

    ter_monthly_factor = ter_annual_percentage / 12.0

    # Simulation loop
    for current_month_idx in range(t_ret_months):
        months_lasted += 1  # Incremented at start of month
        current_year_idx = current_month_idx // 12
        month_in_year_idx = current_month_idx % 12

        # --- OPTIMIZATION: Determine current weights once per year ---
        if month_in_year_idx == 0:
            if current_year_idx < rebalancing_year_idx:
                current_weights_stocks = w_p1_stocks
                current_weights_bonds = w_p1_bonds
                current_weights_str = w_p1_str
                current_weights_fun = w_p1_fun
                current_weights_real_estate = w_p1_real_estate
            else:
                current_weights_stocks = w_p2_stocks
                current_weights_bonds = w_p2_bonds
                current_weights_str = w_p2_str
                current_weights_fun = w_p2_fun
                current_weights_real_estate = w_p2_real_estate

            liquid_weights_sum = (
                current_weights_stocks
                + current_weights_bonds
                + current_weights_str
                + current_weights_fun
            )
            normalized_weights_stocks = (
                current_weights_stocks / liquid_weights_sum
                if liquid_weights_sum > 0
                else 0
            )
            normalized_weights_bonds = (
                current_weights_bonds / liquid_weights_sum
                if liquid_weights_sum > 0
                else 0
            )
            normalized_weights_str = (
                current_weights_str / liquid_weights_sum
                if liquid_weights_sum > 0
                else 0
            )
            normalized_weights_fun = (
                current_weights_fun / liquid_weights_sum
                if liquid_weights_sum > 0
                else 0
            )

        # 1. Add pension if applicable
        nominal_pension_monthly = 0
        if current_year_idx >= y_p_start_idx:
            nominal_pension_monthly = nominal_pension_annual_sequence[current_year_idx]
            current_bank_balance += nominal_pension_monthly

        # 2. Add monthly salary if applicable
        nominal_salary_monthly = 0
        if y_s_start_idx <= current_year_idx < y_s_end_idx:
            nominal_salary_monthly = nominal_salary_annual_sequence[current_year_idx]
            current_bank_balance += nominal_salary_monthly

        # --- Check and Top-Up Bank Account if below REAL_BANK_LOWER_BOUND ---
        monthly_inflation_rate_this_year = annual_to_monthly_compounded_rate(
            annual_inflations_sequence[current_year_idx]
        )
        cumulative_inflation_factor_up_to_current_month = (
            cumulative_inflation_factors_annual[current_year_idx]
            * ((1 + monthly_inflation_rate_this_year) ** (month_in_year_idx + 1))
        )

        current_real_bank_balance = (
            current_bank_balance / cumulative_inflation_factor_up_to_current_month
        )

        if current_real_bank_balance < real_bank_lower_bound:
            real_shortfall_to_cover = real_bank_lower_bound - current_real_bank_balance
            nominal_top_up_amount = (
                real_shortfall_to_cover
                * cumulative_inflation_factor_up_to_current_month
            )

            amount_to_liquidate_for_top_up = nominal_top_up_amount

            # Withdraw from STR (priority 1)
            if amount_to_liquidate_for_top_up > 0:
                if current_str_value >= amount_to_liquidate_for_top_up:
                    current_str_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_bank_balance += current_str_value
                    amount_to_liquidate_for_top_up -= current_str_value
                    current_str_value = 0

            # Withdraw from Bonds (priority 2)
            if amount_to_liquidate_for_top_up > 0:
                if current_bonds_value >= amount_to_liquidate_for_top_up:
                    current_bonds_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_bank_balance += current_bonds_value
                    amount_to_liquidate_for_top_up -= current_bonds_value
                    current_bonds_value = 0

            # Withdraw from Stocks (priority 3)
            if amount_to_liquidate_for_top_up > 0:
                if current_stocks_value >= amount_to_liquidate_for_top_up:
                    current_stocks_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_bank_balance += current_stocks_value
                    amount_to_liquidate_for_top_up -= current_stocks_value
                    current_stocks_value = 0

            # Withdraw from Fun Money (priority 4)
            if amount_to_liquidate_for_top_up > 0:
                if current_fun_value >= amount_to_liquidate_for_top_up:
                    current_fun_value -= amount_to_liquidate_for_top_up
                    current_bank_balance += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_bank_balance += current_fun_value
                    amount_to_liquidate_for_top_up -= current_fun_value
                    current_fun_value = 0

            # CRITICAL FAILURE CHECK 1: If still needed after all liquid assets
            #  (excluding Real Estate)
            if amount_to_liquidate_for_top_up > 0:
                success = False
                break  # Simulation failed, exit loop

        # Re-calculate current_real_bank_balance in case it was topped up
        current_real_bank_balance = (
            current_bank_balance / cumulative_inflation_factor_up_to_current_month
        )

        if current_real_bank_balance > real_bank_upper_bound:
            real_excess_to_invest = current_real_bank_balance - real_bank_upper_bound
            nominal_excess_to_invest = (
                real_excess_to_invest * cumulative_inflation_factor_up_to_current_month
            )

            if liquid_weights_sum > 0:
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
        planned_contribution_applied_for_year = False
        for i, (nominal_contribution_amount, contribution_year_idx) in enumerate(
            c_planned
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

        # Apply C_REAL_MONTHLY_INITIAL every month
        if c_real_monthly_initial > 0:
            nominal_monthly_contribution = (
                c_real_monthly_initial
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

        # 3. Calculate nominal monthly withdrawal amount (includes X_REAL_MONTHLY_INITIAL)
        nominal_monthly_expenses = (
            x_real_monthly_initial
            * cumulative_inflation_factors_annual[current_year_idx]
        )

        # Add planned extra expenses for this month/year
        extra_expense_for_this_month = 0
        expenses_to_remove_indices = []

        for i, (nominal_extra_expense_amount, expense_year_idx) in enumerate(
            local_planned_extra_expenses
        ):
            if current_year_idx == expense_year_idx and month_in_year_idx == 0:
                extra_expense_for_this_month += nominal_extra_expense_amount
                expenses_to_remove_indices.append(i)

        for idx in sorted(expenses_to_remove_indices, reverse=True):
            local_planned_extra_expenses.pop(idx)

        # 4. Process monthly withdrawals (Prioritized)
        withdrawal_needed = nominal_monthly_expenses + extra_expense_for_this_month

        if current_bank_balance >= withdrawal_needed:
            current_bank_balance -= withdrawal_needed
            withdrawal_needed = 0
        else:
            withdrawal_needed -= current_bank_balance
            current_bank_balance = 0

        if withdrawal_needed > 0:
            if current_str_value >= withdrawal_needed:
                current_str_value -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_str_value
                current_str_value = 0

        if withdrawal_needed > 0:
            if current_bonds_value >= withdrawal_needed:
                current_bonds_value -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_bonds_value
                current_bonds_value = 0

        if withdrawal_needed > 0:
            if current_stocks_value >= withdrawal_needed:
                current_stocks_value -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_stocks_value
                current_stocks_value = 0

        if withdrawal_needed > 0:
            if current_fun_value >= withdrawal_needed:
                current_fun_value -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_fun_value
                current_fun_value = 0

        # CRITICAL FAILURE CHECK 2: If still needed after all liquid assets (excluding Real Estate)
        if withdrawal_needed > 0:
            success = False
            break  # Exit simulation early if plan fails

        # 5. Apply monthly returns to investments (at end of month)
        current_stocks_value *= 1 + monthly_returns_lookup["Stocks"][current_month_idx]
        current_bonds_value *= 1 + monthly_returns_lookup["Bonds"][current_month_idx]
        current_str_value *= 1 + monthly_returns_lookup["STR"][current_month_idx]
        current_fun_value *= 1 + monthly_returns_lookup["Fun"][current_month_idx]
        current_real_estate_value *= (
            1 + monthly_returns_lookup["Real Estate"][current_month_idx]
        )

        # Apply TER
        current_stocks_value *= 1 - ter_monthly_factor
        current_bonds_value *= 1 - ter_monthly_factor
        current_str_value *= 1 - ter_monthly_factor
        current_fun_value *= 1 - ter_monthly_factor

        # Ensure no asset values drop below zero due to fees
        current_stocks_value = max(0, current_stocks_value)
        current_bonds_value = max(0, current_bonds_value)
        current_str_value = max(0, current_str_value)
        current_fun_value = max(0, current_fun_value)

        # 6. Rebalance at the start of REBALANCING_YEAR_IDX
        if (
            current_year_idx == rebalancing_year_idx
            and month_in_year_idx == 0
            and rebalancing_year_idx > 0
        ):
            total_investment_value_pre_rebalance = (
                current_stocks_value
                + current_bonds_value
                + current_str_value
                + current_fun_value
                + current_real_estate_value
            )

            cumulative_inflation_rebalance_year = cumulative_inflation_factors_annual[
                rebalancing_year_idx
            ]

            pre_rebalancing_allocations_nominal = {
                "Stocks": current_stocks_value,
                "Bonds": current_bonds_value,
                "STR": current_str_value,
                "Fun": current_fun_value,
                "Real Estate": current_real_estate_value,
            }
            pre_rebalancing_allocations_real = {
                "Stocks": current_stocks_value / cumulative_inflation_rebalance_year,
                "Bonds": current_bonds_value / cumulative_inflation_rebalance_year,
                "STR": current_str_value / cumulative_inflation_rebalance_year,
                "Fun": current_fun_value / cumulative_inflation_rebalance_year,
                "Real Estate": current_real_estate_value
                / cumulative_inflation_rebalance_year,
            }

            # --- START HOUSE PURCHASE LOGIC ---
            if h0_real_cost > 0:
                nominal_house_cost = (
                    h0_real_cost
                    * cumulative_inflation_factors_annual[rebalancing_year_idx]
                )
                liquid_assets_pre_house = (
                    current_str_value
                    + current_bonds_value
                    + current_stocks_value
                    + current_fun_value
                )

                if liquid_assets_pre_house < nominal_house_cost:
                    success = False
                    break  # Exit loop

                remaining_to_buy = nominal_house_cost

                if current_str_value >= remaining_to_buy:
                    current_str_value -= remaining_to_buy
                    remaining_to_buy = 0
                else:
                    remaining_to_buy -= current_str_value
                    current_str_value = 0

                if remaining_to_buy > 0:
                    if current_bonds_value >= remaining_to_buy:
                        current_bonds_value -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_bonds_value
                        current_bonds_value = 0

                if remaining_to_buy > 0:
                    if current_stocks_value >= remaining_to_buy:
                        current_stocks_value -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_stocks_value
                        current_stocks_value = 0

                if remaining_to_buy > 0:
                    if current_fun_value >= remaining_to_buy:
                        current_fun_value -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_fun_value
                        current_fun_value = 0

                current_real_estate_value += nominal_house_cost

            # --- END HOUSE PURCHASE LOGIC ---

            total_investment_value_pre_rebalance = (
                current_stocks_value
                + current_bonds_value
                + current_str_value
                + current_fun_value
                + current_real_estate_value
            )

            # --- START CONDITIONAL REBALANCING LOGIC ---
            if h0_real_cost > 0:
                liquid_portfolio_value_for_rebalance = (
                    total_investment_value_pre_rebalance - current_real_estate_value
                )
                sum_liquid_p2_weights = w_p2_stocks + w_p2_bonds + w_p2_str + w_p2_fun

                if sum_liquid_p2_weights == 0:
                    current_stocks_value = 0.0
                    current_bonds_value = 0.0
                    current_str_value = 0.0
                    current_fun_value = 0.0
                else:
                    normalized_weights_phase2_stocks = (
                        w_p2_stocks / sum_liquid_p2_weights
                    )
                    normalized_weights_phase2_bonds = w_p2_bonds / sum_liquid_p2_weights
                    normalized_weights_phase2_str = w_p2_str / sum_liquid_p2_weights
                    normalized_weights_phase2_fun = w_p2_fun / sum_liquid_p2_weights

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
                    "Stocks": total_investment_value_pre_rebalance * w_p2_stocks,
                    "Bonds": total_investment_value_pre_rebalance * w_p2_bonds,
                    "STR": total_investment_value_pre_rebalance * w_p2_str,
                    "Fun": total_investment_value_pre_rebalance * w_p2_fun,
                    "Real Estate": total_investment_value_pre_rebalance
                    * w_p2_real_estate,
                }

                current_stocks_value = rebalancing_allocations_nominal["Stocks"]
                current_bonds_value = rebalancing_allocations_nominal["Bonds"]
                current_str_value = rebalancing_allocations_nominal["STR"]
                current_fun_value = rebalancing_allocations_nominal["Fun"]
                current_real_estate_value = rebalancing_allocations_nominal[
                    "Real Estate"
                ]

            # --- END CONDITIONAL REBALANCING LOGIC ---

            inflation_factor = cumulative_inflation_rebalance_year
            rebalancing_allocations_real = {
                "Stocks": rebalancing_allocations_nominal["Stocks"] / inflation_factor,
                "Bonds": rebalancing_allocations_nominal["Bonds"] / inflation_factor,
                "STR": rebalancing_allocations_nominal["STR"] / inflation_factor,
                "Fun": rebalancing_allocations_nominal["Fun"] / inflation_factor,
                "Real Estate": rebalancing_allocations_nominal["Real Estate"]
                / inflation_factor,
            }

        # Record nominal wealth at the end of the month (only for successful months)
        nominal_wealth_history[current_month_idx] = (
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

    final_investment_value = (
        current_stocks_value
        + current_bonds_value
        + current_str_value
        + current_fun_value
        + current_real_estate_value
    )
    final_bank_balance = current_bank_balance

    final_allocations_nominal = {
        "Stocks": current_stocks_value,
        "Bonds": current_bonds_value,
        "STR": current_str_value,
        "Fun": current_fun_value,
        "Real Estate": current_real_estate_value,
    }

    cumulative_inflation_end_of_sim = cumulative_inflation_factors_annual[t_ret_years]
    final_allocations_real = {
        "Stocks": current_stocks_value / cumulative_inflation_end_of_sim,
        "Bonds": current_bonds_value / cumulative_inflation_end_of_sim,
        "STR": current_str_value / cumulative_inflation_end_of_sim,
        "Fun": current_fun_value / cumulative_inflation_end_of_sim,
        "Real Estate": current_real_estate_value / cumulative_inflation_end_of_sim,
    }

    return (
        success,
        months_lasted,
        final_investment_value,
        final_bank_balance,
        annual_inflations_sequence,
        nominal_wealth_history,
        bank_balance_history,
        pre_rebalancing_allocations_nominal,
        pre_rebalancing_allocations_real,
        rebalancing_allocations_nominal,
        rebalancing_allocations_real,
        final_allocations_nominal,
        final_allocations_real,
    )
