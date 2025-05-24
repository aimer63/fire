# main.py

import sys
import os
import tomllib
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

# Import helper functions
from helpers import calculate_log_normal_params, calculate_initial_asset_values

# Import the main simulation function
from simulation import run_single_fire_simulation

# Import the new analysis module
import analysis

# Import plotting functions (import display_all_plots_in_one_window as well)
from plots import (
    plot_retirement_duration_distribution,
    plot_final_wealth_distribution_nominal,
    plot_final_wealth_distribution_real,
    plot_wealth_evolution_samples_real,
    plot_wealth_evolution_samples_nominal,
    plot_bank_account_trajectories_real,
    plot_bank_account_trajectories_nominal,
)

def main():
    """
    Main function to orchestrate the Monte Carlo retirement simulation,
    analysis, and plotting.
    """
    # --- 1-5. Config Loading, Parameter Assignment, Derived Calculations, and Assertions ---
    # (Keep all this part as it was in the previous main.py update)
    # ... (Your existing code for sections 1 through 5, including assertions and parameter prints) ...

    config_file_path = 'config.toml'
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        sys.exit(1)

    try:
        with open(config_file_path, 'rb') as f:
            config_data = tomllib.load(f)
    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1)

    print("Configuration file parsed successfully. Extracting parameters...")

    det_inputs = config_data['deterministic_inputs']
    I0 = det_inputs['I0']
    b0 = det_inputs['b0']
    REAL_BANK_LOWER_BOUND_EUROS = det_inputs['REAL_BANK_LOWER_BOUND_EUROS']
    REAL_BANK_UPPER_BOUND_EUROS = det_inputs['REAL_BANK_UPPER_BOUND_EUROS']
    T_ret_years = det_inputs['T_ret_years']
    T_ret_months = T_ret_years * 12

    X_real_monthly_initial = det_inputs['X_real_monthly_initial']
    X_planned_extra = [tuple(item) for item in det_inputs['X_planned_extra']]

    C_real_monthly_initial = det_inputs['C_real_monthly_initial']
    C_planned = [tuple(item) for item in det_inputs['C_planned']] 
    
    H0_real_cost = det_inputs['H0_real_cost']

    P_real_monthly = det_inputs['P_real_monthly']
    PENSION_INFLATION_ADJUSTMENT_FACTOR = det_inputs['PENSION_INFLATION_ADJUSTMENT_FACTOR']
    Y_P_start_idx = det_inputs['Y_P_start_idx']
    
    S_real_monthly = det_inputs['S_real_monthly']
    SALARY_INFLATION_ADJUSTMENT_FACTOR = det_inputs['SALARY_INFLATION_ADJUSTMENT_FACTOR']
    Y_S_start_idx = det_inputs['Y_S_start_idx']
    Y_S_end_idx = det_inputs['Y_S_end_idx']

    eco_assumptions = config_data['economic_assumptions']
    STOCK_MU = eco_assumptions['STOCK_MU']
    STOCK_SIGMA = eco_assumptions['STOCK_SIGMA']
    BOND_MU = eco_assumptions['BOND_MU']
    BOND_SIGMA = eco_assumptions['BOND_SIGMA']
    STR_MU = eco_assumptions['STR_MU']
    STR_SIGMA = eco_assumptions['STR_SIGMA']
    FUN_MU = eco_assumptions['FUN_MU']
    FUN_SIGMA = eco_assumptions['FUN_SIGMA']
    REAL_ESTATE_MU = eco_assumptions['REAL_ESTATE_MU']
    REAL_ESTATE_SIGMA = eco_assumptions['REAL_ESTATE_SIGMA']
    mu_pi = eco_assumptions['mu_pi']
    sigma_pi = eco_assumptions['sigma_pi']

    port_allocs = config_data['portfolio_allocations']
    REBALANCING_YEAR_IDX = port_allocs['REBALANCING_YEAR_IDX']
    W_P1_STOCKS = port_allocs['W_P1_STOCKS']
    W_P1_BONDS = port_allocs['W_P1_BONDS']
    W_P1_STR = port_allocs['W_P1_STR']
    W_P1_FUN = port_allocs['W_P1_FUN']
    W_P1_REAL_ESTATE = port_allocs['W_P1_REAL_ESTATE']
    W_P2_STOCKS = port_allocs['W_P2_STOCKS']
    W_P2_BONDS = port_allocs['W_P2_BONDS']
    W_P2_STR = port_allocs['W_P2_STR']
    W_P2_FUN = port_allocs['W_P2_FUN']
    W_P2_REAL_ESTATE = port_allocs['W_P2_REAL_ESTATE']

    P1_sum = W_P1_STOCKS + W_P1_BONDS + W_P1_STR + W_P1_FUN + W_P1_REAL_ESTATE
    P2_sum = W_P2_STOCKS + W_P2_BONDS + W_P2_STR + W_P2_FUN + W_P2_REAL_ESTATE

    assert np.isclose(P1_sum, 1.0), f"Error: Phase 1 portfolio weights sum to {P1_sum:.4f}, but should sum to 1.0."
    assert np.isclose(P2_sum, 1.0), f"Error: Phase 2 portfolio weights sum to {P2_sum:.4f}, but should sum to 1.0."
    print("Portfolio weights (W_P1, W_P2) successfully validated: sum to 1.0.")

    assert REAL_BANK_UPPER_BOUND_EUROS >= REAL_BANK_LOWER_BOUND_EUROS, \
        f"Error: REAL_BANK_UPPER_BOUND_EUROS ({REAL_BANK_UPPER_BOUND_EUROS:,.2f}) must be greater than or equal to REAL_BANK_LOWER_BOUND_EUROS ({REAL_BANK_LOWER_BOUND_EUROS:,.2f})."
    print("Bank account bounds successfully validated: Upper bound >= Lower bound.")


    sim_params = config_data['simulation_parameters']
    num_simulations = sim_params['num_simulations']
    # random_seed = sim_params['random_seed']
    # np.random.seed(random_seed)

    (
        mu_log_stocks, sigma_log_stocks,
        mu_log_bonds, sigma_log_bonds,
        mu_log_str, sigma_log_str,
        mu_log_fun, sigma_log_fun,
        mu_log_real_estate, sigma_log_real_estate
    ) = calculate_log_normal_params(
        STOCK_MU, STOCK_SIGMA,
        BOND_MU, BOND_SIGMA,
        STR_MU, STR_SIGMA,
        FUN_MU, FUN_SIGMA,
        REAL_ESTATE_MU, REAL_ESTATE_SIGMA
    )

    (
        initial_stocks_value, initial_bonds_value, initial_str_value,
        initial_fun_value, initial_real_estate_value
    ) = calculate_initial_asset_values(
        I0,
        W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE
    )

    print("All parameters successfully extracted and assigned to Python variables, including derived ones.")

    # --- Print all parameters for verification ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"I0: {I0:,.2f}")
    print(f"b0: {b0:,.2f}")
    print(f"REAL_BANK_LOWER_BOUND_EUROS: {REAL_BANK_LOWER_BOUND_EUROS:,.2f}")
    print(f"REAL_BANK_UPPER_BOUND_EUROS: {REAL_BANK_UPPER_BOUND_EUROS:,.2f}")
    print(f"T_ret_years: {T_ret_years}")
    print(f"T_ret_months: {T_ret_months}")
    print(f"X_real_monthly_initial: {X_real_monthly_initial:,.2f}")
    print(f"X_planned_extra: {X_planned_extra}")
    print(f"C_planned: {C_planned}")
    print(f"C_real_monthly_initial: {C_real_monthly_initial:,.2f}")    
    print(f"H0_real_cost: {H0_real_cost:,.2f}")
    print(f"P_real_monthly: {P_real_monthly:,.2f}")
    print(f"PENSION_INFLATION_ADJUSTMENT_FACTOR: {PENSION_INFLATION_ADJUSTMENT_FACTOR}")
    print(f"Y_P_start_idx: {Y_P_start_idx}")
    print(f"S_real_monthly: {S_real_monthly:,.2f}")
    print(f"SALARY_INFLATION_ADJUSTMENT_FACTOR: {SALARY_INFLATION_ADJUSTMENT_FACTOR}")
    print(f"Y_S_start_idx: {Y_S_start_idx}")
    print(f"Y_S_end_idx: {Y_S_end_idx}")

    print("\n--- Economic Assumptions ---")
    print(f"STOCK_MU: {STOCK_MU:.4f}, STOCK_SIGMA: {STOCK_SIGMA:.4f}")
    print(f"BOND_MU: {BOND_MU:.4f}, BOND_SIGMA: {BOND_SIGMA:.4f}")
    print(f"STR_MU: {STR_MU:.4f}, STR_SIGMA: {STR_SIGMA:.4f}")
    print(f"FUN_MU: {FUN_MU:.4f}, FUN_SIGMA: {FUN_SIGMA:.4f}")
    print(f"REAL_ESTATE_MU: {REAL_ESTATE_MU:.4f}, REAL_ESTATE_SIGMA: {REAL_ESTATE_SIGMA:.4f}")
    print(f"mu_pi: {mu_pi:.4f}, sigma_pi: {sigma_pi:.4f}")

    print("\n--- Derived Log-Normal Parameters ---")
    print(f"mu_log_stocks: {mu_log_stocks:.6f}, sigma_log_stocks: {sigma_log_stocks:.6f}")
    print(f"mu_log_bonds: {mu_log_bonds:.6f}, sigma_log_bonds: {sigma_log_bonds:.6f}")
    print(f"mu_log_str: {mu_log_str:.6f}, sigma_log_str: {sigma_log_str:.6f}")
    print(f"mu_log_fun: {mu_log_fun:.6f}, sigma_log_fun: {sigma_log_fun:.6f}")
    print(f"mu_log_real_estate: {mu_log_real_estate:.6f}, sigma_log_real_estate: {sigma_log_real_estate:.6f}")

    print("\n--- Portfolio Allocations ---")
    print(f"REBALANCING_YEAR_IDX: {REBALANCING_YEAR_IDX}")
    print(f"W_P1_STOCKS: {W_P1_STOCKS:.4f}, W_P1_BONDS: {W_P1_BONDS:.4f}, W_P1_STR: {W_P1_STR:.4f}, W_P1_FUN: {W_P1_FUN:.4f}, W_P1_REAL_ESTATE: {W_P1_REAL_ESTATE:.4f}")
    print(f"W_P2_STOCKS: {W_P2_STOCKS:.4f}, W_P2_BONDS: {W_P2_BONDS:.4f}, W_P2_STR: {W_P2_STR:.4f}, W_P2_FUN: {W_P2_FUN:.4f}, W_P2_REAL_ESTATE: {W_P2_REAL_ESTATE:.4f}")

    print("\n--- Initial Asset Values ---")
    print(f"initial_stocks_value: {initial_stocks_value:,.2f}")
    print(f"initial_bonds_value: {initial_bonds_value:,.2f}")
    print(f"initial_str_value: {initial_str_value:,.2f}")
    print(f"initial_fun_value: {initial_fun_value:,.2f}")
    print(f"initial_real_estate_value: {initial_real_estate_value:,.2f}")

    print("\n--- Simulation Parameters ---")
    print(f"num_simulations: {num_simulations}")
    # print(f"random_seed: {random_seed}")

    print("\n--- End of Parameter Assignment and Verification ---")

    # --- 6. Run Monte Carlo Simulations ---
    simulation_results = []
    
    spinner = itertools.cycle(['-', '\\', '|', '/'])
    start_time = time.time()

    print(f"\nRunning {num_simulations} Monte Carlo simulations (T={T_ret_years} years)...")
    for i in range(num_simulations):
        result = run_single_fire_simulation(
            b0,
            initial_stocks_value, initial_bonds_value, initial_str_value, initial_fun_value, initial_real_estate_value,
            T_ret_months, T_ret_years,
            X_real_monthly_initial,
            list(C_planned),
            list(X_planned_extra),
            P_real_monthly, PENSION_INFLATION_ADJUSTMENT_FACTOR, Y_P_start_idx,
            S_real_monthly, SALARY_INFLATION_ADJUSTMENT_FACTOR, Y_S_start_idx, Y_S_end_idx,
            mu_pi, sigma_pi,
            REBALANCING_YEAR_IDX,
            W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE,
            W_P2_STOCKS, W_P2_BONDS, W_P2_STR, W_P2_FUN, W_P2_REAL_ESTATE,
            mu_log_stocks, sigma_log_stocks,
            mu_log_bonds, sigma_log_bonds,
            mu_log_str, sigma_log_str,
            mu_log_fun, sigma_log_fun,
            mu_log_real_estate, sigma_log_real_estate,
            REAL_BANK_LOWER_BOUND_EUROS,\
            REAL_BANK_UPPER_BOUND_EUROS,
            C_real_monthly_initial,
            H0_real_cost,
        )
        simulation_results.append(result)
        
        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\r{next(spinner)} Running simulation {i+1}/{num_simulations} | Elapsed: {elapsed_time:.2f}s")
        sys.stdout.flush()
    
    sys.stdout.write('\n') 
    sys.stdout.flush()
    
    print(f"\nMonte Carlo Simulation Complete. Total time elapsed: {elapsed_time:.2f} seconds.")

    # --- 7. Perform Analysis and Prepare Plotting Data ---
    results_df, plot_data = analysis.perform_analysis_and_prepare_plots_data(
        simulation_results, T_ret_years, I0,
        W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE, # Pass initial weights
        REBALANCING_YEAR_IDX, num_simulations, mu_pi
    )

    # --- 8. Generate Plots ---
    print("\n--- Generating Plots ---")

    # Extract data from plot_data dictionary
    failed_sims = plot_data['failed_sims']
    successful_sims = plot_data['successful_sims']
    plot_lines_data = plot_data['plot_lines_data']
    bank_account_plot_indices = plot_data['bank_account_plot_indices']
    
    # Plotting Historical Distributions
    plot_retirement_duration_distribution(failed_sims, T_ret_years)
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(results_df, plot_lines_data, mu_pi)
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(results_df, bank_account_plot_indices, REAL_BANK_LOWER_BOUND_EUROS)
    plot_bank_account_trajectories_nominal(results_df, bank_account_plot_indices, REAL_BANK_LOWER_BOUND_EUROS)
    
    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True) # This will keep all open Matplotlib windows alive until manually closed

if __name__ == "__main__":
    main()