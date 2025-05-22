# main.py

import sys  # For command-line arguments
import os  # For checking file existence
import tomllib  # For parsing TOML files (built-in in Python 3.11+)
import numpy as np  # Needed for np.log and np.random.seed if you keep them here


def main():
    """
    Main function for the application.
    It reads configuration from a TOML file and assigns parameters to variables.
    This version only performs parsing and prints confirmation.
    """
    # --- 1. Handle Command-Line Arguments for Config File ---
    config_file_path = "config.toml"  # Default config file name
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]  # Allow specifying config file as argument

    # --- 2. Validate Config File Existence ---
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        sys.exit(1)  # Exit if the file doesn't exist

    # --- 3. Read and Parse Config File ---
    print(f"Attempting to load configuration from '{config_file_path}'...")
    try:
        # Open the TOML file in binary read mode ('rb') for tomllib
        with open(config_file_path, "rb") as f:
            config_data = tomllib.load(
                f
            )  # Parse the TOML data into a Python dictionary
    except Exception as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1)  # Exit if there's any error during file reading or TOML parsing

    print("Configuration file parsed successfully. Extracting parameters...")

    # --- 4. Extract and Assign Parameters to Variables ---

    # --- A. Deterministic Inputs ---
    det_inputs = config_data["deterministic_inputs"]
    I0 = det_inputs["I0"]
    b0 = det_inputs["b0"]
    REAL_BANK_LOWER_BOUND_EUROS = det_inputs["REAL_BANK_LOWER_BOUND_EUROS"]
    T_ret_years = det_inputs["T_ret_years"]
    T_ret_months = T_ret_years * 12  # Derived parameter from config

    X_real_monthly_initial = det_inputs["X_real_monthly_initial"]

    # Convert lists of lists from TOML into lists of tuples for C_planned and X_planned_extra
    C_planned = [tuple(item) for item in det_inputs["C_planned"]]
    X_planned_extra = [tuple(item) for item in det_inputs["X_planned_extra"]]

    P_real_monthly = det_inputs["P_real_monthly"]
    PENSION_INFLATION_ADJUSTMENT_FACTOR = det_inputs[
        "PENSION_INFLATION_ADJUSTMENT_FACTOR"
    ]
    Y_P_start_idx = det_inputs["Y_P_start_idx"]

    # --- B. Economic Assumptions ---
    eco_assumptions = config_data["economic_assumptions"]
    STOCK_MU = eco_assumptions["STOCK_MU"]
    STOCK_SIGMA = eco_assumptions["STOCK_SIGMA"]
    BOND_MU = eco_assumptions["BOND_MU"]
    BOND_SIGMA = eco_assumptions["BOND_SIGMA"]
    STR_MU = eco_assumptions["STR_MU"]
    STR_SIGMA = eco_assumptions["STR_SIGMA"]
    FUN_MU = eco_assumptions["FUN_MU"]
    FUN_SIGMA = eco_assumptions["FUN_SIGMA"]
    REAL_ESTATE_MU = eco_assumptions["REAL_ESTATE_MU"]
    REAL_ESTATE_SIGMA = eco_assumptions["REAL_ESTATE_SIGMA"]
    mu_pi = eco_assumptions["mu_pi"]
    sigma_pi = eco_assumptions["sigma_pi"]

    # Pre-calculate log-normal distribution parameters for asset returns
    # These calculations use numpy, so make sure numpy is imported.
    mu_log_stocks = np.log(1 + STOCK_MU) - 0.5 * STOCK_SIGMA**2
    sigma_log_stocks = STOCK_SIGMA
    mu_log_bonds = np.log(1 + BOND_MU) - 0.5 * BOND_SIGMA**2
    sigma_log_bonds = BOND_SIGMA
    mu_log_str = np.log(1 + STR_MU) - 0.5 * STR_SIGMA**2
    sigma_log_str = STR_SIGMA
    mu_log_fun = np.log(1 + FUN_MU) - 0.5 * FUN_SIGMA**2
    sigma_log_fun = FUN_SIGMA
    mu_log_real_estate = np.log(1 + REAL_ESTATE_MU) - 0.5 * REAL_ESTATE_SIGMA**2
    sigma_log_real_estate = REAL_ESTATE_SIGMA

    # --- C. Portfolio Allocations ---
    port_allocs = config_data["portfolio_allocations"]
    REBALANCING_YEAR_IDX = port_allocs["REBALANCING_YEAR_IDX"]
    W_P1_STOCKS = port_allocs["W_P1_STOCKS"]
    W_P1_BONDS = port_allocs["W_P1_BONDS"]
    W_P1_STR = port_allocs["W_P1_STR"]
    W_P1_FUN = port_allocs["W_P1_FUN"]
    W_P1_REAL_ESTATE = port_allocs["W_P1_REAL_ESTATE"]
    W_P2_STOCKS = port_allocs["W_P2_STOCKS"]
    W_P2_BONDS = port_allocs["W_P2_BONDS"]
    W_P2_STR = port_allocs["W_P2_STR"]
    W_P2_FUN = port_allocs["W_P2_FUN"]
    W_P2_REAL_ESTATE = port_allocs["W_P2_REAL_ESTATE"]

    # --- D. Simulation Parameters ---
    sim_params = config_data["simulation_parameters"]
    num_simulations = sim_params["num_simulations"]
    random_seed = sim_params["random_seed"]
    np.random.seed(random_seed)  # Set the seed for reproducibility

    # --- Initial asset values calculation (derived from loaded parameters) ---
    initial_stocks_value = I0 * W_P1_STOCKS
    initial_bonds_value = I0 * W_P1_BONDS
    initial_str_value = I0 * W_P1_STR
    initial_fun_value = I0 * W_P1_FUN
    initial_real_estate_value = I0 * W_P1_REAL_ESTATE

    print("All parameters successfully extracted and assigned to Python variables.")
    # Optional: Print a few variables to confirm
    print(f"  I0: {I0}")
    print(f"  T_ret_years: {T_ret_years}")
    print(f"  X_planned_extra: {X_planned_extra}")
    print(f"  STOCK_MU: {STOCK_MU}")
    print(f"  num_simulations: {num_simulations}")
    print(f"  mu_log_stocks: {mu_log_stocks:.4f}")  # Example of a derived parameter


if __name__ == "__main__":
    main()
