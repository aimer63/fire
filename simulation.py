# simulation.py

import numpy as np

# Import helper functions
from helpers import annual_to_monthly_compounded_rate, inflate_amount_over_years 

def run_single_fire_simulation(
    b0,
    initial_stocks_value, initial_bonds_value, initial_str_value, initial_fun_value, initial_real_estate_value,
    T_ret_months, T_ret_years,
    X_real_monthly_initial,
    C_planned,
    X_planned_extra,
    P_real_monthly, PENSION_INFLATION_ADJUSTMENT_FACTOR, Y_P_start_idx,
    S_real_monthly, SALARY_INFLATION_ADJUSTMENT_FACTOR, Y_S_start_idx, Y_S_end_idx,
    mu_log_pi, sigma_log_pi,
    REBALANCING_YEAR_IDX,
    W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE,
    W_P2_STOCKS, W_P2_BONDS, W_P2_STR, W_P2_FUN, W_P2_REAL_ESTATE,
    mu_log_stocks, sigma_log_stocks,
    mu_log_bonds, sigma_log_bonds,
    mu_log_str, sigma_log_str,
    mu_log_fun, sigma_log_fun,
    mu_log_real_estate, sigma_log_real_estate,
    real_bank_lower_bound,
    real_bank_upper_bound,
    C_real_monthly_initial,
    H0_real_cost,
    TER_ANNUAL_PERCENTAGE,
    shock_events,
):
    """
    Runs a single Monte Carlo simulation of a financial independence retirement plan.
    """

    # Initialize balances and pre-allocate history arrays for performance
    current_b = b0
    current_stocks = initial_stocks_value
    current_bonds = initial_bonds_value
    current_str = initial_str_value
    current_fun = initial_fun_value
    current_real_estate = initial_real_estate_value

    nominal_wealth_history = np.zeros(T_ret_months) # Pre-allocated
    bank_balance_history = np.zeros(T_ret_months)   # Pre-allocated

    success = True
    months_lasted = 0

    # Variables to store rebalancing and final allocations
    pre_rebalancing_allocations_nominal = {}
    pre_rebalancing_allocations_real = {}
    rebalancing_allocations_nominal = {}
    rebalancing_allocations_real = {}
    final_allocations_nominal = {}
    final_allocations_real = {}

    # Pre-generate full sequence of annual returns for each asset and inflation for the entire retirement duration
    annual_inflations_seq = np.random.lognormal(mu_log_pi, sigma_log_pi, T_ret_years) - 1
    annual_stocks_returns_seq = np.random.lognormal(mu_log_stocks, sigma_log_stocks, T_ret_years) - 1
    annual_bonds_returns_seq = np.random.lognormal(mu_log_bonds, sigma_log_bonds, T_ret_years) - 1
    annual_str_returns_seq = np.random.lognormal(mu_log_str, sigma_log_str, T_ret_years) - 1
    annual_fun_returns_seq = np.random.lognormal(mu_log_fun, sigma_log_fun, T_ret_years) - 1
    annual_real_estate_returns_seq = np.random.lognormal(mu_log_real_estate, sigma_log_real_estate, T_ret_years) - 1

    # Apply shock events to the pre-generated annual sequences
    for shock in shock_events:
        shock_year = shock['year']
        shock_asset = shock['asset']
        shock_magnitude = shock['magnitude']

        if 0 <= shock_year < T_ret_years:
            if shock_asset == 'Stocks':
                annual_stocks_returns_seq[shock_year] = shock_magnitude
            elif shock_asset == 'Bonds':
                annual_bonds_returns_seq[shock_year] = shock_magnitude
            elif shock_asset == 'STR':
                annual_str_returns_seq[shock_year] = shock_magnitude
            elif shock_asset == 'Fun':
                annual_fun_returns_seq[shock_year] = shock_magnitude
            elif shock_asset == 'Real Estate':
                annual_real_estate_returns_seq[shock_year] = shock_magnitude
            elif shock_asset == 'Inflation':
                annual_inflations_seq[shock_year] = shock_magnitude


    # --- OPTIMIZATION 1: Pre-calculate cumulative inflation factors ---
    cumulative_inflation_factors_annual = np.ones(T_ret_years + 1)
    for y in range(T_ret_years):
        cumulative_inflation_factors_annual[y+1] = cumulative_inflation_factors_annual[y] * (1 + annual_inflations_seq[y])

    # --- OPTIMIZATION 2: Pre-calculate all monthly returns for all assets ---
    monthly_returns_lookup = {
        'Stocks': np.zeros(T_ret_months),
        'Bonds': np.zeros(T_ret_months),
        'STR': np.zeros(T_ret_months),
        'Fun': np.zeros(T_ret_months),
        'Real Estate': np.zeros(T_ret_months)
    }
    
    for y_idx in range(T_ret_years):
        m_stocks_rate = annual_to_monthly_compounded_rate(annual_stocks_returns_seq[y_idx])
        m_bonds_rate = annual_to_monthly_compounded_rate(annual_bonds_returns_seq[y_idx])
        m_str_rate = annual_to_monthly_compounded_rate(annual_str_returns_seq[y_idx])
        m_fun_rate = annual_to_monthly_compounded_rate(annual_fun_returns_seq[y_idx])
        m_real_estate_rate = annual_to_monthly_compounded_rate(annual_real_estate_returns_seq[y_idx])
        
        start_month = y_idx * 12
        end_month = min((y_idx + 1) * 12, T_ret_months)

        monthly_returns_lookup['Stocks'][start_month:end_month] = m_stocks_rate
        monthly_returns_lookup['Bonds'][start_month:end_month] = m_bonds_rate
        monthly_returns_lookup['STR'][start_month:end_month] = m_str_rate
        monthly_returns_lookup['Fun'][start_month:end_month] = m_fun_rate
        monthly_returns_lookup['Real Estate'][start_month:end_month] = m_real_estate_rate


    # --- OPTIMIZATION 3: Pre-calculate nominal planned contributions/expenses ---
    nominal_c_planned_amounts = []
    for real_amount, year_idx in C_planned:
        nominal_c_amount = real_amount * cumulative_inflation_factors_annual[year_idx]
        nominal_c_planned_amounts.append((nominal_c_amount, year_idx))

    nominal_x_planned_extra_amounts = []
    local_x_planned_extra = list(X_planned_extra)
    for real_amount, year_idx in local_x_planned_extra:
        nominal_x_amount = real_amount * cumulative_inflation_factors_annual[year_idx]
        nominal_x_planned_extra_amounts.append((nominal_x_amount, year_idx))

    # --- OPTIMIZATION 4: Pre-calculate nominal pension/salary per year ---
    nominal_pension_annual_seq = np.zeros(T_ret_years)
    nominal_salary_annual_seq = np.zeros(T_ret_years)

    for y_idx in range(T_ret_years):
        if y_idx >= Y_P_start_idx:
            factor = np.prod(1 + (annual_inflations_seq[Y_P_start_idx:y_idx] * PENSION_INFLATION_ADJUSTMENT_FACTOR)) if y_idx > Y_P_start_idx else 1.0
            nominal_pension_annual_seq[y_idx] = P_real_monthly * cumulative_inflation_factors_annual[Y_P_start_idx] * factor

        if Y_S_start_idx <= y_idx < Y_S_end_idx:
            factor = np.prod(1 + (annual_inflations_seq[Y_S_start_idx:y_idx] * SALARY_INFLATION_ADJUSTMENT_FACTOR)) if y_idx > Y_S_start_idx else 1.0
            nominal_salary_annual_seq[y_idx] = S_real_monthly * cumulative_inflation_factors_annual[Y_S_start_idx] * factor
            
    ter_monthly_factor = TER_ANNUAL_PERCENTAGE / 12.0

    # Simulation loop
    for current_month_idx in range(T_ret_months):
        months_lasted += 1 # Incremented at start of month
        current_year_idx = current_month_idx // 12
        month_in_year_idx = current_month_idx % 12

        # --- OPTIMIZATION: Determine current weights once per year ---
        if month_in_year_idx == 0:
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

            liquid_weights_sum = current_W_STOCKS + current_W_BONDS + current_W_STR + current_W_FUN
            norm_W_STOCKS = current_W_STOCKS / liquid_weights_sum if liquid_weights_sum > 0 else 0
            norm_W_BONDS = current_W_BONDS / liquid_weights_sum if liquid_weights_sum > 0 else 0
            norm_W_STR = current_W_STR / liquid_weights_sum if liquid_weights_sum > 0 else 0
            norm_W_FUN = current_W_FUN / liquid_weights_sum if liquid_weights_sum > 0 else 0


        # 1. Add pension if applicable
        nominal_pension_monthly = 0
        if current_year_idx >= Y_P_start_idx:
            nominal_pension_monthly = nominal_pension_annual_seq[current_year_idx]
            current_b += nominal_pension_monthly

        # 2. Add monthly salary if applicable
        nominal_salary_monthly = 0
        if Y_S_start_idx <= current_year_idx < Y_S_end_idx:
            nominal_salary_monthly = nominal_salary_annual_seq[current_year_idx]
            current_b += nominal_salary_monthly


        # --- Check and Top-Up Bank Account if below Real Lower Bound ---
        monthly_inflation_rate_this_year = annual_to_monthly_compounded_rate(annual_inflations_seq[current_year_idx])
        cumulative_inflation_factor_up_to_current_month = cumulative_inflation_factors_annual[current_year_idx] * ((1 + monthly_inflation_rate_this_year)**(month_in_year_idx + 1))
        
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
                    current_b += current_str
                    amount_to_liquidate_for_top_up -= current_str
                    current_str = 0

            # Withdraw from Bonds (priority 2)
            if amount_to_liquidate_for_top_up > 0:
                if current_bonds >= amount_to_liquidate_for_top_up:
                    current_bonds -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_bonds
                    amount_to_liquidate_for_top_up -= current_bonds
                    current_bonds = 0

            # Withdraw from Stocks (priority 3)
            if amount_to_liquidate_for_top_up > 0:
                if current_stocks >= amount_to_liquidate_for_top_up:
                    current_stocks -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_stocks
                    amount_to_liquidate_for_top_up -= current_stocks
                    current_stocks = 0

            # Withdraw from Fun Money (priority 4)
            if amount_to_liquidate_for_top_up > 0:
                if current_fun >= amount_to_liquidate_for_top_up:
                    current_fun -= amount_to_liquidate_for_top_up
                    current_b += amount_to_liquidate_for_top_up
                    amount_to_liquidate_for_top_up = 0
                else:
                    current_b += current_fun
                    amount_to_liquidate_for_top_up -= current_fun
                    current_fun = 0

            # CRITICAL FAILURE CHECK 1: If still needed after all liquid assets (excluding Real Estate)
            if amount_to_liquidate_for_top_up > 0:
                success = False
                # REMOVED: Recording nominal_wealth_history[current_month_idx] here.
                # It will now be handled by the post-loop fill using the last successful month's value.
                break # Simulation failed, exit loop

        # Re-calculate real_current_b in case it was topped up
        real_current_b = current_b / cumulative_inflation_factor_up_to_current_month

        if real_current_b > real_bank_upper_bound:
            real_excess_to_invest = real_current_b - real_bank_upper_bound
            nominal_excess_to_invest = real_excess_to_invest * cumulative_inflation_factor_up_to_current_month
            
            if liquid_weights_sum > 0:
                current_stocks += nominal_excess_to_invest * norm_W_STOCKS
                current_bonds += nominal_excess_to_invest * norm_W_BONDS
                current_str += nominal_excess_to_invest * norm_W_STR
                current_fun += nominal_excess_to_invest * norm_W_FUN
                current_b -= nominal_excess_to_invest

        # 2. Handle planned contributions
        c_planned_applied_for_year = False
        for i, (nominal_c_amount, c_year_idx) in enumerate(nominal_c_planned_amounts):
            if current_year_idx == c_year_idx and month_in_year_idx == 0 and not c_planned_applied_for_year:
                current_stocks += nominal_c_amount * current_W_STOCKS
                current_bonds += nominal_c_amount * current_W_BONDS
                current_str += nominal_c_amount * current_W_STR
                current_fun += nominal_c_amount * current_W_FUN
                current_real_estate += nominal_c_amount * current_W_REAL_ESTATE
                c_planned_applied_for_year = True

        # Apply C_real_monthly_initial every month
        if C_real_monthly_initial > 0:
            nominal_c_monthly = C_real_monthly_initial * cumulative_inflation_factors_annual[current_year_idx]
            current_stocks += nominal_c_monthly * current_W_STOCKS
            current_bonds += nominal_c_monthly * current_W_BONDS
            current_str += nominal_c_monthly * current_W_STR
            current_fun += nominal_c_monthly * current_W_FUN
            current_real_estate += nominal_c_monthly * current_W_REAL_ESTATE

        # 3. Calculate nominal monthly withdrawal amount (includes X_real_monthly_initial)
        nominal_x_monthly = X_real_monthly_initial * cumulative_inflation_factors_annual[current_year_idx]
        
        # Add planned extra expenses for this month/year
        extra_expense_for_this_month = 0
        expenses_to_remove_indices = [] 

        for i, (nominal_x_amount, x_year_idx) in enumerate(nominal_x_planned_extra_amounts):
            if current_year_idx == x_year_idx and month_in_year_idx == 0:
                extra_expense_for_this_month += nominal_x_amount
                expenses_to_remove_indices.append(i)

        for idx in sorted(expenses_to_remove_indices, reverse=True):
            nominal_x_planned_extra_amounts.pop(idx) 

        # 4. Process monthly withdrawals (Prioritized)
        withdrawal_needed = nominal_x_monthly + extra_expense_for_this_month
        
        if current_b >= withdrawal_needed:
            current_b -= withdrawal_needed
            withdrawal_needed = 0
        else:
            withdrawal_needed -= current_b
            current_b = 0

        if withdrawal_needed > 0:
            if current_str >= withdrawal_needed:
                current_str -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_str
                current_str = 0
        
        if withdrawal_needed > 0:
            if current_bonds >= withdrawal_needed:
                current_bonds -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_bonds
                current_bonds = 0

        if withdrawal_needed > 0:
            if current_stocks >= withdrawal_needed:
                current_stocks -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_stocks
                current_stocks = 0

        if withdrawal_needed > 0:
            if current_fun >= withdrawal_needed:
                current_fun -= withdrawal_needed
                withdrawal_needed = 0
            else:
                withdrawal_needed -= current_fun
                current_fun = 0

        # CRITICAL FAILURE CHECK 2: If still needed after all liquid assets (excluding Real Estate)
        if withdrawal_needed > 0:
            success = False
            # REMOVED: Recording nominal_wealth_history[current_month_idx] here.
            break # Exit simulation early if plan fails

        # 5. Apply monthly returns to investments (at end of month)
        current_stocks *= (1 + monthly_returns_lookup['Stocks'][current_month_idx])
        current_bonds *= (1 + monthly_returns_lookup['Bonds'][current_month_idx])
        current_str *= (1 + monthly_returns_lookup['STR'][current_month_idx])
        current_fun *= (1 + monthly_returns_lookup['Fun'][current_month_idx])
        current_real_estate *= (1 + monthly_returns_lookup['Real Estate'][current_month_idx])

        # Apply TER
        current_stocks *= (1 - ter_monthly_factor)
        current_bonds *= (1 - ter_monthly_factor)
        current_str *= (1 - ter_monthly_factor)
        current_fun *= (1 - ter_monthly_factor)

        # Ensure no asset values drop below zero due to fees
        current_stocks = max(0, current_stocks)
        current_bonds = max(0, current_bonds)
        current_str = max(0, current_str)
        current_fun = max(0, current_fun)

        # 6. Rebalance at the start of REBALANCING_YEAR_IDX
        if current_year_idx == REBALANCING_YEAR_IDX and month_in_year_idx == 0 and REBALANCING_YEAR_IDX > 0:
            total_investment_value_pre_rebalance = current_stocks + current_bonds + current_str + current_fun + current_real_estate
            
            cumulative_inflation_rebalance_year = cumulative_inflation_factors_annual[REBALANCING_YEAR_IDX]

            pre_rebalancing_allocations_nominal = {
                'Stocks': current_stocks, 'Bonds': current_bonds, 'STR': current_str,
                'Fun': current_fun, 'Real Estate': current_real_estate
            }
            pre_rebalancing_allocations_real = {
                'Stocks': current_stocks / cumulative_inflation_rebalance_year,
                'Bonds': current_bonds / cumulative_inflation_rebalance_year,
                'STR': current_str / cumulative_inflation_rebalance_year,
                'Fun': current_fun / cumulative_inflation_rebalance_year,
                'Real Estate': current_real_estate / cumulative_inflation_rebalance_year
            }

            # --- START HOUSE PURCHASE LOGIC ---
            if H0_real_cost > 0:
                nominal_house_cost = H0_real_cost * cumulative_inflation_factors_annual[REBALANCING_YEAR_IDX]
                liquid_assets_pre_house = current_str + current_bonds + current_stocks + current_fun

                if liquid_assets_pre_house < nominal_house_cost:
                    success = False
                    # REMOVED: Recording nominal_wealth_history[current_month_idx] here.
                    break # Exit loop
                
                remaining_to_buy = nominal_house_cost
                
                if current_str >= remaining_to_buy:
                    current_str -= remaining_to_buy
                    remaining_to_buy = 0
                else:
                    remaining_to_buy -= current_str
                    current_str = 0
                
                if remaining_to_buy > 0:
                    if current_bonds >= remaining_to_buy:
                        current_bonds -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_bonds
                        current_bonds = 0
                        
                if remaining_to_buy > 0:
                    if current_stocks >= remaining_to_buy:
                        current_stocks -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_stocks
                        current_stocks = 0
                        
                if remaining_to_buy > 0:
                    if current_fun >= remaining_to_buy:
                        current_fun -= remaining_to_buy
                        remaining_to_buy = 0
                    else:
                        remaining_to_buy -= current_fun
                        current_fun = 0
                
                current_real_estate += nominal_house_cost

            # --- END HOUSE PURCHASE LOGIC ---

            total_investment_value_pre_rebalance = current_stocks + current_bonds + current_str + current_fun + current_real_estate

            # --- START CONDITIONAL REBALANCING LOGIC ---
            if H0_real_cost > 0:
                liquid_portfolio_value_for_rebalance = total_investment_value_pre_rebalance - current_real_estate
                sum_liquid_p2_weights = W_P2_STOCKS + W_P2_BONDS + W_P2_STR + W_P2_FUN
                
                if sum_liquid_p2_weights == 0:
                    current_stocks = 0.0
                    current_bonds = 0.0
                    current_str = 0.0
                    current_fun = 0.0
                else:
                    norm_W_P2_STOCKS = W_P2_STOCKS / sum_liquid_p2_weights
                    norm_W_P2_BONDS = W_P2_BONDS / sum_liquid_p2_weights
                    norm_W_P2_STR = W_P2_STR / sum_liquid_p2_weights
                    norm_W_P2_FUN = W_P2_FUN / sum_liquid_p2_weights

                    current_stocks = liquid_portfolio_value_for_rebalance * norm_W_P2_STOCKS
                    current_bonds = liquid_portfolio_value_for_rebalance * norm_W_P2_BONDS
                    current_str = liquid_portfolio_value_for_rebalance * norm_W_P2_STR
                    current_fun = liquid_portfolio_value_for_rebalance * norm_W_P2_FUN
                
                rebalancing_allocations_nominal = {
                    'Stocks': current_stocks, 'Bonds': current_bonds, 'STR': current_str,
                    'Fun': current_fun, 'Real Estate': current_real_estate
                }
            else:
                rebalancing_allocations_nominal = {
                    'Stocks': total_investment_value_pre_rebalance * W_P2_STOCKS,
                    'Bonds': total_investment_value_pre_rebalance * W_P2_BONDS,
                    'STR': total_investment_value_pre_rebalance * W_P2_STR,
                    'Fun': total_investment_value_pre_rebalance * W_P2_FUN,
                    'Real Estate': total_investment_value_pre_rebalance * W_P2_REAL_ESTATE
                }

                current_stocks = rebalancing_allocations_nominal['Stocks']
                current_bonds = rebalancing_allocations_nominal['Bonds']
                current_str = rebalancing_allocations_nominal['STR']
                current_fun = rebalancing_allocations_nominal['Fun']
                current_real_estate = rebalancing_allocations_nominal['Real Estate']

            # --- END CONDITIONAL REBALANCING LOGIC ---

            rebalancing_allocations_real = {
                'Stocks': rebalancing_allocations_nominal['Stocks'] / cumulative_inflation_rebalance_year,
                'Bonds': rebalancing_allocations_nominal['Bonds'] / cumulative_inflation_rebalance_year,
                'STR': rebalancing_allocations_nominal['STR'] / cumulative_inflation_rebalance_year,
                'Fun': rebalancing_allocations_nominal['Fun'] / cumulative_inflation_rebalance_year,
                'Real Estate': rebalancing_allocations_nominal['Real Estate'] / cumulative_inflation_rebalance_year
            }


        # Record nominal wealth at the end of the month (only for successful months)
        nominal_wealth_history[current_month_idx] = current_b + current_stocks + current_bonds + current_str + current_fun + current_real_estate
        bank_balance_history[current_month_idx] = current_b

    # --- MODIFIED: Handle history for failed simulations by filling with last value ---
    if not success:
        # Fill the remaining part of the history arrays with the value from the LAST *SUCCESSFUL* month.
        if months_lasted > 1: # Ensures there was at least one successful month before failure
            # If current_month_idx was the failing month, then months_lasted = current_month_idx + 1.
            # So, (months_lasted - 1) is current_month_idx (the failing month's index)
            # And (months_lasted - 2) is current_month_idx - 1 (the last successful month's index).
            #last_recorded_wealth = nominal_wealth_history[months_lasted - 2]
            last_recorded_bank_balance = bank_balance_history[months_lasted - 2]
            
            # Fill from the failing month's index (months_lasted - 1) onwards
            nominal_wealth_history[months_lasted - 1:] = np.nan # last_recorded_wealth
            bank_balance_history[months_lasted - 1:] = np.nan # last_recorded_bank_balance
        elif months_lasted == 1: # Simulation failed in the very first month (current_month_idx = 0)
            # In this case, nominal_wealth_history[0] would be the initial wealth if it was recorded before failure,
            # but if it failed immediately, it might still be 0 or initial.
            # We'll just leave it as it is (likely initial 0 or initial wealth if recorded).
            # If nominal_wealth_history[0] was recorded successfully before failure, then it would be used.
            # If it failed before recording, it would be 0, and filling with 0 is appropriate.
            pass # The array would already be all zeros from initialization if no values were recorded.
    
    final_investment = current_stocks + current_bonds + current_str + current_fun + current_real_estate
    final_bank_balance = current_b

    final_allocations_nominal = {
        'Stocks': current_stocks, 'Bonds': current_bonds, 'STR': current_str,
        'Fun': current_fun, 'Real Estate': current_real_estate
    }

    cumulative_inflation_end_of_sim = cumulative_inflation_factors_annual[T_ret_years]
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