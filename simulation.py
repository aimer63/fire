# simulation.py

import numpy as np

# Import helper functions from the helpers module
from helpers import annual_to_monthly_compounded_rate, inflate_amount_over_years

def run_single_fire_simulation(
    b0,
    initial_stocks_value, initial_bonds_value, initial_str_value, initial_fun_value, initial_real_estate_value,
    T_ret_months, T_ret_years,
    X_real_monthly_initial,
    C_planned,
    X_planned_extra,
    P_real_monthly, PENSION_INFLATION_ADJUSTMENT_FACTOR, Y_P_start_idx,
    mu_pi, sigma_pi,
    REBALANCING_YEAR_IDX,
    W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE,
    W_P2_STOCKS, W_P2_BONDS, W_P2_STR, W_P2_FUN, W_P2_REAL_ESTATE,
    mu_log_stocks, sigma_log_stocks,
    mu_log_bonds, sigma_log_bonds,
    mu_log_str, sigma_log_str,
    mu_log_fun, sigma_log_fun,
    mu_log_real_estate, sigma_log_real_estate,
    real_bank_lower_bound,
    C_real_monthly_initial,
):
    """
    Runs a single Monte Carlo simulation of a financial independence retirement plan.

    Args:
        b0 (float): Initial bank account balance (nominal).
        initial_stocks_value (float): Initial value of stocks.
        initial_bonds_value (float): Initial value of bonds.
        initial_str_value (float): Initial value of short-term rate (cash equivalent).
        initial_fun_value (float): Initial value of 'fun money' speculative assets.
        initial_real_estate_value (float): Initial value of real estate.
        T_ret_months (int): Total retirement duration in months.
        T_ret_years (int): Total retirement duration in years.
        X_real_monthly_initial (float): Initial monthly expenditure in real terms.
        C_planned (list of tuples): List of planned contributions (real_amount, year_idx).
        X_planned_extra (list of tuples): List of planned extra expenses (real_amount, year_idx).
        P_real_monthly (float): Monthly pension income in real terms.
        PENSION_INFLATION_ADJUSTMENT_FACTOR (float): Factor by which pension adjusts to inflation.
        Y_P_start_idx (int): Year index when pension starts (0-indexed).
        mu_pi (float): Mean of annual inflation rate.
        sigma_pi (float): Standard deviation of annual inflation rate.
        REBALANCING_YEAR_IDX (int): Year index when portfolio rebalances to Phase 2 weights.
        W_P1_STOCKS (float): Phase 1 stock weight.
        W_P1_BONDS (float): Phase 1 bond weight.
        W_P1_STR (float): Phase 1 short-term rate weight.
        W_P1_FUN (float): Phase 1 fun money weight.
        W_P1_REAL_ESTATE (float): Phase 1 real estate weight.
        W_P2_STOCKS (float): Phase 2 stock weight.
        W_P2_BONDS (float): Phase 2 bond weight.
        W_P2_STR (float): Phase 2 short-term rate weight.
        W_P2_FUN (float): Phase 2 fun money weight.
        W_P2_REAL_ESTATE (float): Phase 2 real estate weight.
        mu_log_stocks (float): Log-normal mean for stocks.
        sigma_log_stocks (float): Log-normal sigma for stocks.
        mu_log_bonds (float): Log-normal mean for bonds.
        sigma_log_bonds (float): Log-normal sigma for bonds.
        mu_log_str (float): Log-normal mean for STR.
        sigma_log_str (float): Log-normal sigma for STR.
        mu_log_fun (float): Log-normal mean for fun money.
        sigma_log_fun (float): Log-normal sigma for fun money.
        mu_log_real_estate (float): Log-normal mean for real estate.
        sigma_log_real_estate (float): Log-normal sigma for real estate.
        real_bank_lower_bound (float): Minimum desired real bank balance.
        C_real_monthly_initial (float): Initial monthly contribution in real terms.

    Returns:
        tuple: A tuple containing simulation results:
            (success, months_lasted, final_investment, final_bank_balance,
             annual_inflations_seq, nominal_wealth_history, bank_balance_history,
             pre_rebalancing_allocations_nominal, pre_rebalancing_allocations_real,
             rebalancing_allocations_nominal, rebalancing_allocations_real,
             final_allocations_nominal, final_allocations_real)
    """

    # Initialize balances and history
    current_b = b0
    current_stocks = initial_stocks_value
    current_bonds = initial_bonds_value
    current_str = initial_str_value
    current_fun = initial_fun_value
    current_real_estate = initial_real_estate_value

    nominal_wealth_history = []
    bank_balance_history = [] 
    annual_inflations_seq = np.zeros(T_ret_years) # Placeholder for actual sequence

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
    # for the entire retirement duration of this single simulation trial
    annual_inflations_seq = np.random.lognormal(
        np.log(1 + mu_pi) - 0.5 * sigma_pi**2, sigma_pi, T_ret_years
    ) - 1

    # Generate annual returns for each asset class
    annual_stocks_returns_seq = np.random.lognormal(
        mu_log_stocks, sigma_log_stocks, T_ret_years
    ) - 1
    annual_bonds_returns_seq = np.random.lognormal(
        mu_log_bonds, sigma_log_bonds, T_ret_years
    ) - 1
    annual_str_returns_seq = np.random.lognormal(
        mu_log_str, sigma_log_str, T_ret_years
    ) - 1
    annual_fun_returns_seq = np.random.lognormal(
        mu_log_fun, sigma_log_fun, T_ret_years
    ) - 1
    annual_real_estate_returns_seq = np.random.lognormal(
        mu_log_real_estate, sigma_log_real_estate, T_ret_years
    ) - 1

    # Pre-calculate nominal planned contributions and pension start year based on this trial's inflation
    nominal_c_planned_amounts = []
    for real_amount, year_idx in C_planned:
        nominal_c_amount = inflate_amount_over_years(
            real_amount, year_idx, annual_inflations_seq
        )
        nominal_c_planned_amounts.append((nominal_c_amount, year_idx))

    # Pre-calculate nominal planned extra expenses based on this trial's inflation
    nominal_x_planned_extra_amounts = []
    # Create a local mutable copy to safely remove items after they're applied
    # This ensures the original X_planned_extra list passed from the calling scope isn't modified
    local_x_planned_extra = list(X_planned_extra) 
    for real_amount, year_idx in local_x_planned_extra: # Iterate over the local copy
        nominal_x_amount = inflate_amount_over_years(
            real_amount, year_idx, annual_inflations_seq
        )
        nominal_x_planned_extra_amounts.append((nominal_x_amount, year_idx))

    nominal_pension_start_amount = inflate_amount_over_years(
        P_real_monthly, Y_P_start_idx, annual_inflations_seq
    )

    # Simulation loop
    for current_month_idx in range(T_ret_months):
        months_lasted += 1
        current_year_idx = current_month_idx // 12
        month_in_year_idx = current_month_idx % 12

        # 1. Add pension if applicable (starts at the beginning of the pension year)
        nominal_pension_monthly = 0
        if current_year_idx >= Y_P_start_idx:
            # Adjust pension by its specific inflation factor from start year
            pension_inflation_factor = np.prod(1 + (annual_inflations_seq[Y_P_start_idx:current_year_idx] * PENSION_INFLATION_ADJUSTMENT_FACTOR)) if current_year_idx > Y_P_start_idx else 1
            
            nominal_pension_monthly = nominal_pension_start_amount * pension_inflation_factor
            current_b += nominal_pension_monthly # Pension directly to bank account

        # --- Check and Top-Up Bank Account if below Real Lower Bound ---
        # Calculate current cumulative inflation factor for real value assessment
        monthly_inflation_rate_this_year = annual_to_monthly_compounded_rate(annual_inflations_seq[current_year_idx])
        cumulative_inflation_factor_up_to_current_month = np.prod(1 + annual_inflations_seq[:current_year_idx]) * ((1 + monthly_inflation_rate_this_year)**(month_in_year_idx + 1))
        
        real_current_b = current_b / cumulative_inflation_factor_up_to_current_month

        if real_current_b < real_bank_lower_bound:
            real_shortfall_to_cover = real_bank_lower_bound - real_current_b
            nominal_top_up_amount = real_shortfall_to_cover * cumulative_inflation_factor_up_to_current_month

            amount_to_liquidate_for_top_up = nominal_top_up_amount

            # Withdraw from STR (priority 1)
            if amount_to_liquidate_for_top_up > 0:
                if current_str >= amount_to_liquidate_for_top_up:
                    current_str -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_str # Take all of STR
                    amount_to_liquidate_for_top_up -= current_str
                    current_str = 0

            # Withdraw from Bonds (priority 2)
            if amount_to_liquidate_for_top_up > 0:
                if current_bonds >= amount_to_liquidate_for_top_up:
                    current_bonds -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_bonds # Take all of Bonds
                    amount_to_liquidate_for_top_up -= current_bonds
                    current_bonds = 0

            # Withdraw from Stocks (priority 3)
            if amount_to_liquidate_for_top_up > 0:
                if current_stocks >= amount_to_liquidate_for_top_up:
                    current_stocks -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_stocks # Take all of Stocks
                    amount_to_liquidate_for_top_up -= current_stocks
                    current_stocks = 0

            # Withdraw from Fun Money (priority 4)
            if amount_to_liquidate_for_top_up > 0:
                if current_fun >= amount_to_liquidate_for_top_up:
                    current_fun -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_fun # Take all of Fun Money
                    amount_to_liquidate_for_top_up -= current_fun
                    current_fun = 0

            # CRITICAL FAILURE CHECK: If still needed after all liquid assets (excluding Real Estate)
            if amount_to_liquidate_for_top_up > 0:
                success = False
                break # Simulation failed, exit loop

        # 2. Handle planned contributions
        for nominal_c_amount, c_year_idx in nominal_c_planned_amounts:
            if current_year_idx == c_year_idx and month_in_year_idx == 0: # Apply contribution at start of the year
                # Determine current portfolio weights for contribution allocation
                if current_year_idx < REBALANCING_YEAR_IDX:
                    current_W_STOCKS = W_P1_STOCKS
                    current_W_BONDS = W_P1_BONDS
                    current_W_STR = W_P1_STR
                    current_W_FUN = W_P1_FUN
                    current_W_REAL_ESTATE = W_P1_REAL_ESTATE
                else:
                    current_W_STOCKS = W_P2_STOCKS
                    current_W_BONDS = W_P2_BONDS
                    current_W_STR = W_P2_STR
                    current_W_FUN = W_P2_FUN
                    current_W_REAL_ESTATE = W_P2_REAL_ESTATE
                
                # Allocate contribution across assets based on weights
                current_stocks += nominal_c_amount * current_W_STOCKS
                current_bonds += nominal_c_amount * current_W_BONDS
                current_str += nominal_c_amount * current_W_STR
                current_fun += nominal_c_amount * current_W_FUN
                current_real_estate += nominal_c_amount * current_W_REAL_ESTATE

        # NEW LOGIC FOR C_real_monthly_initial (Fixed Monthly Contribution)
        # Apply C_real_monthly_initial every month
        if C_real_monthly_initial > 0: # Only if a positive contribution is set
            # Determine current portfolio weights for allocation
            if current_year_idx < REBALANCING_YEAR_IDX:
                current_W_STOCKS = W_P1_STOCKS
                current_W_BONDS = W_P1_BONDS
                current_W_STR = W_P1_STR
                current_W_FUN = W_P1_FUN
                current_W_REAL_ESTATE = W_P1_REAL_ESTATE
            else:
                current_W_STOCKS = W_P2_STOCKS
                current_W_BONDS = W_P2_BONDS
                current_W_STR = W_P2_STR
                current_W_FUN = W_P2_FUN
                current_W_REAL_ESTATE = W_P2_REAL_ESTATE

            # Inflate the real monthly contribution to its nominal value for the current year
            nominal_c_monthly = inflate_amount_over_years(
                C_real_monthly_initial,
                current_year_idx,
                annual_inflations_seq # Use this trial's inflation sequence
            )
            
            # Allocate the nominal monthly contribution across assets based on weights
            current_stocks += nominal_c_monthly * current_W_STOCKS
            current_bonds += nominal_c_monthly * current_W_BONDS
            current_str += nominal_c_monthly * current_W_STR
            current_fun += nominal_c_monthly * current_W_FUN
            current_real_estate += nominal_c_monthly * current_W_REAL_ESTATE

        # 3. Calculate nominal monthly withdrawal amount (includes X_real_monthly_initial)
        # Inflate the initial real withdrawal to current month's nominal value
        nominal_x_monthly = inflate_amount_over_years(
            X_real_monthly_initial,
            current_year_idx,
            annual_inflations_seq # Use this trial's inflation sequence
        )
        
        # Add planned extra expenses for this month/year
        extra_expense_for_this_month = 0
        expenses_to_remove_indices = [] 

        # Iterate over the pre-calculated nominal planned extra expenses
        # Note: We iterate over a copy or pre-calculated list to avoid issues when popping
        # `nominal_x_planned_extra_amounts` is already a pre-calculated list for this simulation run.
        for i, (nominal_x_amount, x_year_idx) in enumerate(nominal_x_planned_extra_amounts):
            if current_year_idx == x_year_idx and month_in_year_idx == 0: # Apply expense at start of the year
                extra_expense_for_this_month += nominal_x_amount
                expenses_to_remove_indices.append(i) # Mark for removal

        # Remove applied expenses from the list to prevent them from being applied again
        # Iterate in reverse to avoid index shifting issues when popping
        for idx in sorted(expenses_to_remove_indices, reverse=True):
            nominal_x_planned_extra_amounts.pop(idx) # This modifies the list for the current simulation run

        # 4. Process monthly withdrawals (Prioritized)
        # Sum normal monthly expense and any extra planned expenses for this month
        withdrawal_needed = nominal_x_monthly + extra_expense_for_this_month
        
        # Withdraw from bank account
        if current_b >= withdrawal_needed:
            current_b -= withdrawal_needed
            withdrawal_needed = 0
        else:
            withdrawal_needed -= current_b
            current_b = 0

        # Withdraw from STR
        if withdrawal_needed > 0:
            if current_str >= withdrawal_needed:
                current_str -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_str
                current_str = 0
        
        # Withdraw from Bonds
        if withdrawal_needed > 0:
            if current_bonds >= withdrawal_needed:
                current_bonds -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_bonds
                current_bonds = 0

        # Withdraw from Stocks
        if withdrawal_needed > 0:
            if current_stocks >= withdrawal_needed:
                current_stocks -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_stocks
                current_stocks = 0

        # Withdraw from Fun Money
        if withdrawal_needed > 0:
            if current_fun >= withdrawal_needed:
                current_fun -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_fun
                current_fun = 0

        # Check for failure (if Real Estate would be needed)
        if withdrawal_needed > 0:
            success = False
            break # Exit simulation early if plan fails

        # 5. Apply monthly returns to investments (at end of month)
        # Calculate monthly returns for this year for each asset
        monthly_stocks_return = annual_to_monthly_compounded_rate(annual_stocks_returns_seq[current_year_idx])
        monthly_bonds_return = annual_to_monthly_compounded_rate(annual_bonds_returns_seq[current_year_idx])
        monthly_str_return = annual_to_monthly_compounded_rate(annual_str_returns_seq[current_year_idx])
        monthly_fun_return = annual_to_monthly_compounded_rate(annual_fun_returns_seq[current_year_idx])
        monthly_real_estate_return = annual_to_monthly_compounded_rate(annual_real_estate_returns_seq[current_year_idx])

        # Apply returns
        current_stocks *= (1 + monthly_stocks_return)
        current_bonds *= (1 + monthly_bonds_return)
        current_str *= (1 + monthly_str_return)
        current_fun *= (1 + monthly_fun_return)
        current_real_estate *= (1 + monthly_real_estate_return)
        
        # 6. Rebalance at the start of REBALANCING_YEAR_IDX
        if current_year_idx == REBALANCING_YEAR_IDX and month_in_year_idx == 0 and REBALANCING_YEAR_IDX > 0: # Ensure rebalancing only happens once at the start of the year
            total_investment_value_pre_rebalance = current_stocks + current_bonds + current_str + current_fun + current_real_estate
            
            # Calculate cumulative inflation up to rebalancing year to get real values
            cumulative_inflation_rebalance_year = np.prod(1 + annual_inflations_seq[:REBALANCING_YEAR_IDX]) if REBALANCING_YEAR_IDX > 0 else 1.0

            # Record nominal and real portfolio allocation *just before* rebalancing
            pre_rebalancing_allocations_nominal = {
                'Stocks': current_stocks,
                'Bonds': current_bonds,
                'STR': current_str,
                'Fun': current_fun,
                'Real Estate': current_real_estate
            }
            pre_rebalancing_allocations_real = {
                'Stocks': current_stocks / cumulative_inflation_rebalance_year,
                'Bonds': current_bonds / cumulative_inflation_rebalance_year,
                'STR': current_str / cumulative_inflation_rebalance_year,
                'Fun': current_fun / cumulative_inflation_rebalance_year,
                'Real Estate': current_real_estate / cumulative_inflation_rebalance_year
            }


            # Record nominal portfolio allocation just after rebalancing
            rebalancing_allocations_nominal = {
                'Stocks': total_investment_value_pre_rebalance * W_P2_STOCKS,
                'Bonds': total_investment_value_pre_rebalance * W_P2_BONDS,
                'STR': total_investment_value_pre_rebalance * W_P2_STR,
                'Fun': total_investment_value_pre_rebalance * W_P2_FUN,
                'Real Estate': total_investment_value_pre_rebalance * W_P2_REAL_ESTATE
            }

            rebalancing_allocations_real = {
                'Stocks': rebalancing_allocations_nominal['Stocks'] / cumulative_inflation_rebalance_year,
                'Bonds': rebalancing_allocations_nominal['Bonds'] / cumulative_inflation_rebalance_year,
                'STR': rebalancing_allocations_nominal['STR'] / cumulative_inflation_rebalance_year,
                'Fun': rebalancing_allocations_nominal['Fun'] / cumulative_inflation_rebalance_year,
                'Real Estate': rebalancing_allocations_nominal['Real Estate'] / cumulative_inflation_rebalance_year
            }

            current_stocks = total_investment_value_pre_rebalance * W_P2_STOCKS
            current_bonds = total_investment_value_pre_rebalance * W_P2_BONDS
            current_str = total_investment_value_pre_rebalance * W_P2_STR
            current_fun = total_investment_value_pre_rebalance * W_P2_FUN
            current_real_estate = total_investment_value_pre_rebalance * W_P2_REAL_ESTATE

        # Record nominal wealth at the end of the month
        nominal_wealth_history.append(current_b + current_stocks + current_bonds + current_str + current_fun + current_real_estate)
        bank_balance_history.append(current_b)

    # Final results for this simulation
    final_investment = current_stocks + current_bonds + current_str + current_fun + current_real_estate
    final_bank_balance = current_b

    # Record nominal and real portfolio allocation at the end of the simulation
    final_allocations_nominal = {
        'Stocks': current_stocks,
        'Bonds': current_bonds,
        'STR': current_str,
        'Fun': current_fun,
        'Real Estate': current_real_estate
    }

    # Calculate cumulative inflation for the entire simulation duration
    cumulative_inflation_end_of_sim = np.prod(1 + annual_inflations_seq) if T_ret_years > 0 else 1.0
    final_allocations_real = {
        'Stocks': current_stocks / cumulative_inflation_end_of_sim,
        'Bonds': current_bonds / cumulative_inflation_end_of_sim,
        'STR': current_str / cumulative_inflation_end_of_sim,
        'Fun': current_fun / cumulative_inflation_end_of_sim,
        'Real Estate': current_real_estate / cumulative_inflation_end_of_sim
    }

    return (success, months_lasted, final_investment, final_bank_balance, annual_inflations_seq, 
            nominal_wealth_history, bank_balance_history,
            pre_rebalancing_allocations_nominal, pre_rebalancing_allocations_real,
            rebalancing_allocations_nominal, rebalancing_allocations_real, 
            final_allocations_nominal, final_allocations_real)