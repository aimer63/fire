# main.py

import sys
import os
import tomllib
import numpy as np
import time # Import for time tracking
import itertools # Import for spinner animation


# Import helper functions from the helpers module
from helpers import calculate_log_normal_params, calculate_initial_asset_values

# Import the main simulation function from the simulation module
from simulation import run_single_fire_simulation

# Import plotting functions from the plots module (these will be defined next)
# from plots import plot_bank_account_evolution_real, plot_bank_account_evolution_nominal

def main():
    """
    Main function to orchestrate the Monte Carlo retirement simulation.
    Reads parameters from a TOML config file, runs simulations, and prepares for plotting.
    """
    # --- 1. Handle Command-Line Arguments for Config File ---
    config_file_path = 'config.toml' # Default config file name
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1] # Allow specifying config file as argument

    # --- 2. Validate Config File Existence ---
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        sys.exit(1) # Exit if the file doesn't exist

    # --- 3. Read and Parse Config File ---
    print(f"Attempting to load configuration from '{config_file_path}'...")
    try:
        # Open the TOML file in binary read mode ('rb') for tomllib
        with open(config_file_path, 'rb') as f:
            config_data = tomllib.load(f) # Parse the TOML data into a Python dictionary
    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1) # Exit if there's any error during file reading or TOML parsing

    print("Configuration file parsed successfully. Extracting parameters...")

    # --- 4. Extract and Assign Parameters to Variables ---

    # --- A. Deterministic Inputs ---
    det_inputs = config_data['deterministic_inputs']
    I0 = det_inputs['I0']
    b0 = det_inputs['b0']
    REAL_BANK_LOWER_BOUND_EUROS = det_inputs['REAL_BANK_LOWER_BOUND_EUROS']
    T_ret_years = det_inputs['T_ret_years']
    T_ret_months = T_ret_years * 12 # Derived parameter from config

    X_real_monthly_initial = det_inputs['X_real_monthly_initial']
    
    # Convert lists of lists from TOML into lists of tuples for C_planned and X_planned_extra
    C_planned = [tuple(item) for item in det_inputs['C_planned']] 
    X_planned_extra = [tuple(item) for item in det_inputs['X_planned_extra']]

    P_real_monthly = det_inputs['P_real_monthly']
    PENSION_INFLATION_ADJUSTMENT_FACTOR = det_inputs['PENSION_INFLATION_ADJUSTMENT_FACTOR']
    Y_P_start_idx = det_inputs['Y_P_start_idx']

    # --- B. Economic Assumptions ---
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

    # --- C. Portfolio Allocations ---
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

    # --- Assertion: Check if portfolio weights sum to 1 ---
    # Using np.isclose for floating-point comparison
    P1_sum = W_P1_STOCKS + W_P1_BONDS + W_P1_STR + W_P1_FUN + W_P1_REAL_ESTATE
    P2_sum = W_P2_STOCKS + W_P2_BONDS + W_P2_STR + W_P2_FUN + W_P2_REAL_ESTATE

    assert np.isclose(P1_sum, 1.0), f"Error: Phase 1 portfolio weights sum to {P1_sum:.4f}, but should sum to 1.0."
    assert np.isclose(P2_sum, 1.0), f"Error: Phase 2 portfolio weights sum to {P2_sum:.4f}, but should sum to 1.0."
    print("Portfolio weights (W_P1, W_P2) successfully validated: sum to 1.0.")

    # --- D. Simulation Parameters ---
    sim_params = config_data['simulation_parameters']
    num_simulations = sim_params['num_simulations']
    random_seed = sim_params['random_seed']
    np.random.seed(random_seed) # Set the random seed for reproducibility of simulation results

    # --- 5. Call helper functions for derived calculations ---
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

    # --- Optional: Print all parameters for verification (you can comment this out later) ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"I0: {I0:,.2f}")
    print(f"b0: {b0:,.2f}")
    print(f"REAL_BANK_LOWER_BOUND_EUROS: {REAL_BANK_LOWER_BOUND_EUROS:,.2f}")
    print(f"T_ret_years: {T_ret_years}")
    print(f"T_ret_months: {T_ret_months}")
    print(f"X_real_monthly_initial: {X_real_monthly_initial:,.2f}")
    print(f"C_planned: {C_planned}")
    print(f"X_planned_extra: {X_planned_extra}")
    print(f"P_real_monthly: {P_real_monthly:,.2f}")
    print(f"PENSION_INFLATION_ADJUSTMENT_FACTOR: {PENSION_INFLATION_ADJUSTMENT_FACTOR}")
    print(f"Y_P_start_idx: {Y_P_start_idx}")

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
    print(f"random_seed: {random_seed}")

    print("\n--- End of Parameter Assignment and Verification ---")


# --- 6. Run Monte Carlo Simulations ---
    simulation_results = []
    successful_simulations = 0
    
    # Setup for progress indicator
    spinner = itertools.cycle(['-', '\\', '|', '/']) # Spinner characters
    start_time = time.time() # Record start time

    print(f"\nRunning {num_simulations} Monte Carlo simulations (T={T_ret_years} years)...")
    for i in range(num_simulations):
        # Call the simulation function with all loaded and derived parameters
        # Pass copies of mutable lists (C_planned, X_planned_extra) to ensure
        # each simulation starts with the original state of these inputs.
        result = run_single_fire_simulation(
            b0,
            initial_stocks_value, initial_bonds_value, initial_str_value, initial_fun_value, initial_real_estate_value,
            T_ret_months, T_ret_years,
            X_real_monthly_initial,
            list(C_planned),        # Pass a copy for this simulation
            list(X_planned_extra),  # Pass a copy for this simulation
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
            REAL_BANK_LOWER_BOUND_EUROS
        )
        simulation_results.append(result)
        if result[0]: # result[0] is the 'success' boolean from run_single_fire_simulation
            successful_simulations += 1
        
        # Update progress indicator
        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\r{next(spinner)} Running simulation {i+1}/{num_simulations} | Elapsed: {elapsed_time:.2f}s")
        sys.stdout.flush() # Ensure the output is immediately written to the console
    
    # After the loop, print a final newline to ensure the next print statement appears on a new line
    sys.stdout.write('\n') 
    sys.stdout.flush()

    success_rate = (successful_simulations / num_simulations) * 100
    print(f"\nMonte Carlo Simulation Complete.")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds.")
    print(f"Success Rate: {success_rate:.2f}% out of {num_simulations} trials.")

    # --- 7. Prepare and Call Plotting Functions ---
    print("\nPreparing data for plots...")
    
    # Filter results for successful simulations for plotting (if desired, or use all)
    # For bank balance evolution, we might want to see all runs to understand failures too.
    
    # Extract necessary data for plotting
    # Assuming bank_balance_history is at result index 6
    bank_balance_histories_nominal = [res[6] for res in simulation_results]
    # Assuming annual_inflations_seq is at result index 4
    annual_inflations_sequences = [res[4] for res in simulation_results]

    # Placeholder calls to plotting functions (functions to be defined in plots.py)
    # Uncomment and ensure 'import matplotlib.pyplot as plt' is added to plots.py
    # and 'from plots import ...' is uncommented at the top of main.py once plots.py is ready.

    # plot_bank_account_evolution_real(
    #     bank_balance_histories_nominal,
    #     annual_inflations_sequences,
    #     REAL_BANK_LOWER_BOUND_EUROS,
    #     T_ret_months,
    #     plot_title="Bank Account Balance Evolution (Real Terms)",
    #     filename="bank_account_real_evolution.png",
    #     num_paths_to_plot=min(50, num_simulations) # Plot up to 50 paths
    # )
    
    # plot_bank_account_evolution_nominal(
    #     bank_balance_histories_nominal,
    #     REAL_BANK_LOWER_BOUND_EUROS, # Nominal equivalent of real lower bound will be dynamic, but this is fixed initial real
    #     T_ret_months,
    #     plot_title="Bank Account Balance Evolution (Nominal Terms)",
    #     filename="bank_account_nominal_evolution.png",
    #     num_paths_to_plot=min(50, num_simulations) # Plot up to 50 paths
    # )
    
    # print("Plots will be saved to current directory once 'plots.py' functions are defined.")


if __name__ == "__main__":
    main()