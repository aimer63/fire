# main.py
import sys
import os
import tomllib
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from helpers import calculate_log_normal_params, calculate_initial_asset_values

# Import the main simulation function
from simulation import run_single_fire_simulation

# Import the new analysis module
import analysis

# Import plotting functions
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
    config_file_path = 'config.toml'
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        sys.exit(1)

    try:
        with open(config_file_path, 'rb') as f:
            config_data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1)

    print("Configuration file parsed successfully. Extracting parameters...")

    det_inputs = config_data['deterministic_inputs']
    i0 = det_inputs['i0']
    b0 = det_inputs['b0']
    real_bank_lower_bound = det_inputs['real_bank_lower_bound']
    real_bank_upper_bound = det_inputs['real_bank_upper_bound']
    t_ret_years = det_inputs['t_ret_years']
    t_ret_months = t_ret_years * 12

    x_real_monthly_initial = det_inputs['x_real_monthly_initial']
    x_planned_extra = [tuple(item) for item in det_inputs['x_planned_extra']]

    c_real_monthly_initial = det_inputs['c_real_monthly_initial']
    c_planned = [tuple(item) for item in det_inputs['c_planned']]
    ter_annual_percentage = det_inputs['ter_annual_percentage']

    h0_real_cost = det_inputs['h0_real_cost']

    p_real_monthly = det_inputs['p_real_monthly']
    pension_inflation_adjustment_factor = det_inputs['pension_inflation_adjustment_factor']
    y_p_start_idx = det_inputs['y_p_start_idx']

    s_real_monthly = det_inputs['s_real_monthly']
    salary_inflation_adjustment_factor = det_inputs['salary_inflation_adjustment_factor']
    y_s_start_idx = det_inputs['y_s_start_idx']
    y_s_end_idx = det_inputs['y_s_end_idx']

    eco_assumptions = config_data['economic_assumptions']
    stock_mu = eco_assumptions['stock_mu']
    stock_sigma = eco_assumptions['stock_sigma']
    bond_mu = eco_assumptions['bond_mu']
    bond_sigma = eco_assumptions['bond_sigma']
    str_mu = eco_assumptions['str_mu']
    str_sigma = eco_assumptions['str_sigma']
    fun_mu = eco_assumptions['fun_mu']
    fun_sigma = eco_assumptions['fun_sigma']
    real_estate_mu = eco_assumptions['real_estate_mu']
    real_estate_sigma = eco_assumptions['real_estate_sigma']
    mu_pi = eco_assumptions['mu_pi']
    sigma_pi = eco_assumptions['sigma_pi']

    # Load historical shock events
    shocks_config = config_data.get('shocks', {})
    shock_events = shocks_config.get('events', [])

    port_allocs = config_data['portfolio_allocations']
    rebalancing_year_idx = port_allocs['rebalancing_year_idx']
    w_p1_stocks = port_allocs['w_p1_stocks']
    w_p1_bonds = port_allocs['w_p1_bonds']
    w_p1_str = port_allocs['w_p1_str']
    w_p1_fun = port_allocs['w_p1_fun']
    w_p1_real_estate = port_allocs['w_p1_real_estate']
    w_p2_stocks = port_allocs['w_p2_stocks']
    w_p2_bonds = port_allocs['w_p2_bonds']
    w_p2_str = port_allocs['w_p2_str']
    w_p2_fun = port_allocs['w_p2_fun']
    w_p2_real_estate = port_allocs['w_p2_real_estate']

    p1_sum = w_p1_stocks + w_p1_bonds + w_p1_str + w_p1_fun + w_p1_real_estate
    p2_sum = w_p2_stocks + w_p2_bonds + w_p2_str + w_p2_fun + w_p2_real_estate

    assert np.isclose(p1_sum, 1.0), f"Error: Phase 1 portfolio weights sum to {p1_sum:.4f}, but should sum to 1.0."
    assert np.isclose(p2_sum, 1.0), f"Error: Phase 2 portfolio weights sum to {p2_sum:.4f}, but should sum to 1.0."
    print("Portfolio weights (w_p1, w_p2) successfully validated: sum to 1.0.")

    assert real_bank_upper_bound >= real_bank_lower_bound, \
        f"Error: real_bank_upper_bound ({real_bank_upper_bound:,.2f}) must be greater than or equal to real_bank_lower_bound ({real_bank_lower_bound:,.2f})."
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
        mu_log_real_estate, sigma_log_real_estate,
        mu_log_pi, sigma_log_pi,
    ) = calculate_log_normal_params(
        stock_mu, stock_sigma,
        bond_mu, bond_sigma,
        str_mu, str_sigma,
        fun_mu, fun_sigma,
        real_estate_mu, real_estate_sigma,
        mu_pi, sigma_pi,
    )

    (
        initial_stocks_value, initial_bonds_value, initial_str_value,
        initial_fun_value, initial_real_estate_value
    ) = calculate_initial_asset_values(
        i0,
        w_p1_stocks, w_p1_bonds, w_p1_str, w_p1_fun, w_p1_real_estate
    )

    print("All parameters successfully extracted and assigned to Python variables, including derived ones.")

    # --- Print all parameters for verification ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"i0: {i0:,.2f}")
    print(f"b0: {b0:,.2f}")
    print(f"real_bank_lower_bound: {real_bank_lower_bound:,.2f}")
    print(f"real_bank_upper_bound: {real_bank_upper_bound:,.2f}")
    print(f"t_ret_years: {t_ret_years}")
    print(f"t_ret_months: {t_ret_months}")
    print(f"x_real_monthly_initial: {x_real_monthly_initial:,.2f}")
    print(f"x_planned_extra: {x_planned_extra}")
    print(f"c_planned: {c_planned}")
    print(f"c_real_monthly_initial: {c_real_monthly_initial:,.2f}")
    print(f"ter_annual_percentage: {ter_annual_percentage:.4f}")
    print(f"h0_real_cost: {h0_real_cost:,.2f}")
    print(f"p_real_monthly: {p_real_monthly:,.2f}")
    print(f"pension_inflation_adjustment_factor: {pension_inflation_adjustment_factor}")
    print(f"y_p_start_idx: {y_p_start_idx}")
    print(f"s_real_monthly: {s_real_monthly:,.2f}")
    print(f"salary_inflation_adjustment_factor: {salary_inflation_adjustment_factor}")
    print(f"y_s_start_idx: {y_s_start_idx}")
    print(f"y_s_end_idx: {y_s_end_idx}")

    print("\n--- Economic Assumptions ---")
    print(f"stock_mu: {stock_mu:.4f}, stock_sigma: {stock_sigma:.4f}")
    print(f"bond_mu: {bond_mu:.4f}, bond_sigma: {bond_sigma:.4f}")
    print(f"str_mu: {str_mu:.4f}, str_sigma: {str_sigma:.4f}")
    print(f"fun_mu: {fun_mu:.4f}, fun_sigma: {fun_sigma:.4f}")
    print(f"real_estate_mu: {real_estate_mu:.4f}, real_estate_sigma: {real_estate_sigma:.4f}")
    print(f"mu_pi: {mu_pi:.4f}, sigma_pi: {sigma_pi:.4f}")

    print("\n--- Derived Log-Normal Parameters ---")
    print(f"mu_log_stocks: {mu_log_stocks:.6f}, sigma_log_stocks: {sigma_log_stocks:.6f}")
    print(f"mu_log_bonds: {mu_log_bonds:.6f}, sigma_log_bonds: {sigma_log_bonds:.6f}")
    print(f"mu_log_str: {mu_log_str:.6f}, sigma_log_str: {sigma_log_str:.6f}")
    print(f"mu_log_fun: {mu_log_fun:.6f}, sigma_log_fun: {sigma_log_fun:.6f}")
    print(f"mu_log_real_estate: {mu_log_real_estate:.6f}, sigma_log_real_estate: {sigma_log_real_estate:.6f}")
    print(f"mu_log_pi: {mu_log_pi:.6f}, sigma_log_pi: {sigma_log_pi:.6f}")

    print("\n--- Portfolio Allocations ---")
    print(f"rebalancing_year_idx: {rebalancing_year_idx}")
    print(f"w_p1_stocks: {w_p1_stocks:.4f}, w_p1_bonds: {w_p1_bonds:.4f}, w_p1_str: {w_p1_str:.4f}, w_p1_fun: {w_p1_fun:.4f}, w_p1_real_estate: {w_p1_real_estate:.4f}")
    print(f"w_p2_stocks: {w_p2_stocks:.4f}, w_p2_bonds: {w_p2_bonds:.4f}, w_p2_str: {w_p2_str:.4f}, w_p2_fun: {w_p2_fun:.4f}, w_p2_real_estate: {w_p2_real_estate:.4f}")

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

    print(f"\nRunning {num_simulations} Monte Carlo simulations (T={t_ret_years} years)...")
    for i in range(num_simulations):
        result = run_single_fire_simulation(
            i0, b0,
            # initial_stocks_value, initial_bonds_value, initial_str_value, initial_fun_value, initial_real_estate_value,
            t_ret_months, t_ret_years,
            x_real_monthly_initial,
            list(c_planned),
            list(x_planned_extra),
            p_real_monthly, pension_inflation_adjustment_factor, y_p_start_idx,
            s_real_monthly, salary_inflation_adjustment_factor, y_s_start_idx, y_s_end_idx,
            mu_log_pi, sigma_log_pi,
            rebalancing_year_idx,
            w_p1_stocks, w_p1_bonds, w_p1_str, w_p1_fun, w_p1_real_estate,
            w_p2_stocks, w_p2_bonds, w_p2_str, w_p2_fun, w_p2_real_estate,
            mu_log_stocks, sigma_log_stocks,
            mu_log_bonds, sigma_log_bonds,
            mu_log_str, sigma_log_str,
            mu_log_fun, sigma_log_fun,
            mu_log_real_estate, sigma_log_real_estate,
            real_bank_lower_bound,
            real_bank_upper_bound,
            c_real_monthly_initial,
            h0_real_cost,
            ter_annual_percentage,
            shock_events,
        )
        simulation_results.append(result)

        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\r{next(spinner)} Running simulation {i+1}/{num_simulations} | Elapsed: {elapsed_time:.2f}s")
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

    end_simulation_time = time.time()
    total_simulation_elapsed_time = end_simulation_time - start_time

    print(f"\nMonte Carlo Simulation Complete. Total time elapsed: {total_simulation_elapsed_time:.2f} seconds.")

    # --- 7. Perform Analysis and Prepare Plotting Data ---
    results_df, plot_data = analysis.perform_analysis_and_prepare_plots_data(
        simulation_results, t_ret_years, i0,
        w_p1_stocks, w_p1_bonds, w_p1_str, w_p1_fun, w_p1_real_estate,
        rebalancing_year_idx, num_simulations, mu_pi
    )

    # Generate and print the consolidated FIRE plan summary
    initial_total_wealth = i0 + b0
    fire_summary_string = analysis.generate_fire_plan_summary(simulation_results, initial_total_wealth, t_ret_years)

    # Print the consolidated summary, including the total simulation time here
    print(f"\nTotal simulations run: {num_simulations}")
    print(f"Total simulation time: {total_simulation_elapsed_time:.2f} seconds")
    print(fire_summary_string)

    # --- 8. Generate Plots ---
    print("\n--- Generating Plots ---")

    # Extract data from plot_data dictionary
    failed_sims = plot_data['failed_sims']
    successful_sims = plot_data['successful_sims']
    plot_lines_data = plot_data['plot_lines_data']
    bank_account_plot_indices = plot_data['bank_account_plot_indices']

    # Plotting Historical Distributions
    plot_retirement_duration_distribution(failed_sims, t_ret_years)
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(results_df, plot_lines_data, mu_pi)
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(results_df, bank_account_plot_indices, real_bank_lower_bound)
    plot_bank_account_trajectories_nominal(results_df, bank_account_plot_indices, real_bank_lower_bound)

    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True) # This will keep all open Matplotlib windows alive until manually closed

if __name__ == "__main__":
    main()
