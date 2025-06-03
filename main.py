"""
main.py

This is the main entry point for the Personal Financial Independence / Early
Retirement (FIRE) plan Monte Carlo simulation application.

It orchestrates the entire simulation process, including:
- Loading simulation parameters and configurations from 'config.toml'.
- Running multiple iterations of the financial simulation.
- Performing post-simulation analysis on the results.
- Generating various plots to visualize the simulation outcomes.

This script brings together functionalities from the 'helpers', 'simulation',
'analysis', and 'plots' modules to provide a comprehensive tool for
FIRE planning.
"""

import sys
import os
import tomllib
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from numpy.typing import NDArray
import pandas as pd  # Import pandas for DataFrame type hint in analysis output

# Import helper functions
from helpers import calculate_log_normal_params, calculate_initial_asset_values

# Import the main simulation function and its return type
from simulation import run_single_fire_simulation, SimulationRunResult

# Import the new analysis module functions and its plotting data TypedDict
import analysis
from analysis import PlotDataDict  # Import PlotDataDict for explicit typing

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


def main() -> None:
    """
    Main function to orchestrate the Monte Carlo retirement simulation,
    analysis, and plotting.
    """
    # --- 1-5. Config Loading, Parameter Assignment, Derived Calculations, and Assertions ---
    config_file_path: str = "config.toml"
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        sys.exit(1)

    config_data: dict[str, Any]
    try:
        with open(config_file_path, "rb") as f:
            config_data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1)

    print("Configuration file parsed successfully. Extracting parameters...")

    det_inputs: dict[str, Any] = config_data["deterministic_inputs"]
    # Renamed variables to snake_case and assigned types
    initial_investment: float = det_inputs["i0"]
    initial_bank_balance: float = det_inputs["b0"]
    real_bank_lower_bound: float = det_inputs["real_bank_lower_bound"]
    real_bank_upper_bound: float = det_inputs["real_bank_upper_bound"]
    total_retirement_years: int = det_inputs["t_ret_years"]
    total_retirement_months: int = total_retirement_years * 12

    initial_real_monthly_expenses: float = det_inputs["x_real_monthly_initial"]
    # Ensure inner items are tuples of float and int, as expected by simulation.py
    planned_extra_expenses: list[tuple[float, int]] = [
        (float(item[0]), int(item[1]))
        for item in det_inputs["x_planned_extra"]  # Explicitly cast to float and int
    ]

    initial_real_monthly_contribution: float = det_inputs["c_real_monthly_initial"]
    planned_contributions: list[tuple[float, int]] = [
        (float(item[0]), int(item[1]))
        for item in det_inputs["c_planned"]  # Explicitly cast to float and int
    ]
    ter_annual_percentage: float = det_inputs["ter_annual_percentage"]

    initial_real_house_cost: float = det_inputs["h0_real_cost"]

    initial_real_monthly_pension: float = det_inputs["p_real_monthly"]
    pension_inflation_adjustment_factor: float = det_inputs[
        "pension_inflation_adjustment_factor"
    ]
    pension_start_year_idx: int = det_inputs["y_p_start_idx"]

    initial_real_monthly_salary: float = det_inputs["s_real_monthly"]
    salary_inflation_adjustment_factor: float = det_inputs[
        "salary_inflation_adjustment_factor"
    ]
    salary_start_year_idx: int = det_inputs["y_s_start_idx"]
    salary_end_year_idx: int = det_inputs["y_s_end_idx"]

    eco_assumptions: dict[str, Any] = config_data["economic_assumptions"]
    stock_mu: float = eco_assumptions["stock_mu"]
    stock_sigma: float = eco_assumptions["stock_sigma"]
    bond_mu: float = eco_assumptions["bond_mu"]
    bond_sigma: float = eco_assumptions["bond_sigma"]
    str_mu: float = eco_assumptions["str_mu"]
    str_sigma: float = eco_assumptions["str_sigma"]
    fun_mu: float = eco_assumptions["fun_mu"]
    fun_sigma: float = eco_assumptions["fun_sigma"]
    real_estate_mu: float = eco_assumptions["real_estate_mu"]
    real_estate_sigma: float = eco_assumptions["real_estate_sigma"]
    mu_pi: float = eco_assumptions[
        "mu_pi"
    ]  # Used directly in plots.py, so keep this name for now
    sigma_pi: float = eco_assumptions[
        "sigma_pi"
    ]  # Used directly in plots.py, so keep this name for now

    # Load historical shock events
    shocks_config: dict[str, Any] = config_data.get("shocks", {})
    shock_events: list[dict[str, Any]] = shocks_config.get("events", [])

    port_allocs: dict[str, Any] = config_data["portfolio_allocations"]
    rebalancing_trigger_year_idx: int = port_allocs[
        "rebalancing_year_idx"
    ]  # Renamed for clarity and PEP 8
    phase1_stocks_weight: float = port_allocs[
        "w_p1_stocks"
    ]  # Renamed for clarity and PEP 8
    phase1_bonds_weight: float = port_allocs[
        "w_p1_bonds"
    ]  # Renamed for clarity and PEP 8
    phase1_str_weight: float = port_allocs["w_p1_str"]  # Renamed for clarity and PEP 8
    phase1_fun_weight: float = port_allocs["w_p1_fun"]  # Renamed for clarity and PEP 8
    phase1_real_estate_weight: float = port_allocs[
        "w_p1_real_estate"
    ]  # Renamed for clarity and PEP 8
    phase2_stocks_weight: float = port_allocs[
        "w_p2_stocks"
    ]  # Renamed for clarity and PEP 8
    phase2_bonds_weight: float = port_allocs[
        "w_p2_bonds"
    ]  # Renamed for clarity and PEP 8
    phase2_str_weight: float = port_allocs["w_p2_str"]  # Renamed for clarity and PEP 8
    phase2_fun_weight: float = port_allocs["w_p2_fun"]  # Renamed for clarity and PEP 8
    phase2_real_estate_weight: float = port_allocs[
        "w_p2_real_estate"
    ]  # Renamed for clarity and PEP 8

    p1_sum: float = (
        phase1_stocks_weight
        + phase1_bonds_weight
        + phase1_str_weight
        + phase1_fun_weight
        + phase1_real_estate_weight
    )
    p2_sum: float = (
        phase2_stocks_weight
        + phase2_bonds_weight
        + phase2_str_weight
        + phase2_fun_weight
        + phase2_real_estate_weight
    )

    assert np.isclose(p1_sum, 1.0), f"Phase 1 weights sum to {p1_sum:.4f}, not 1.0."
    assert np.isclose(p2_sum, 1.0), (
        f"Phase 2 weights sum to {p2_sum:.4f}, not 1.0."
    )  # Corrected from p1_sum to p2_sum in message
    print("Portfolio weights (w_p1, w_p2) successfully validated: sum to 1.0.")

    assert real_bank_upper_bound >= real_bank_lower_bound, (
        f"Bounds invalid: Upper ({real_bank_upper_bound:,.0f}) "  # Fixed implicit concatenation
        + f"< Lower ({real_bank_lower_bound:,.0f})."
    )
    print("Bank account bounds successfully validated: Upper bound >= Lower bound.")

    sim_params: dict[str, Any] = config_data["simulation_parameters"]
    num_simulations: int = sim_params["num_simulations"]
    # random_seed = sim_params['random_seed'] # Not used
    # np.random.seed(random_seed) # Not used

    (
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
        mu_log_inflation,  # Renamed from mu_log_pi to match simulation.py
        sigma_log_inflation,  # Renamed from sigma_log_pi to match simulation.py
    ) = calculate_log_normal_params(
        stock_mu,
        stock_sigma,
        bond_mu,
        bond_sigma,
        str_mu,
        str_sigma,
        fun_mu,
        fun_sigma,
        real_estate_mu,
        real_estate_sigma,
        mu_pi,  # Keep original name for input to helper function
        sigma_pi,  # Keep original name for input to helper function
    )

    (
        initial_stocks_value,
        initial_bonds_value,
        initial_str_value,
        initial_fun_value,
        initial_real_estate_value,
    ) = calculate_initial_asset_values(
        initial_investment,
        phase1_stocks_weight,
        phase1_bonds_weight,
        phase1_str_weight,
        phase1_fun_weight,
        phase1_real_estate_weight,
    )

    print(
        "All parameters successfully extracted and assigned to Python variables, "  # Fixed implicit concatenation
        + "including derived ones."
    )

    # --- Print all parameters for verification ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"initial_investment: {initial_investment:,.2f}")
    print(f"initial_bank_balance: {initial_bank_balance:,.2f}")
    print(f"real_bank_lower_bound: {real_bank_lower_bound:,.2f}")
    print(f"real_bank_upper_bound: {real_bank_upper_bound:,.2f}")
    print(f"total_retirement_years: {total_retirement_years}")
    print(f"total_retirement_months: {total_retirement_months}")
    print(f"initial_real_monthly_expenses: {initial_real_monthly_expenses:,.2f}")
    print(f"planned_extra_expenses: {planned_extra_expenses}")
    print(f"planned_contributions: {planned_contributions}")
    print(
        f"initial_real_monthly_contribution: {initial_real_monthly_contribution:,.2f}"
    )
    print(f"ter_annual_percentage: {ter_annual_percentage:.4f}")
    print(f"initial_real_house_cost: {initial_real_house_cost:,.2f}")
    print(f"initial_real_monthly_pension: {initial_real_monthly_pension:,.2f}")
    print(f"pension_inflation_adjustment_factor: {pension_inflation_adjustment_factor}")
    print(f"pension_start_year_idx: {pension_start_year_idx}")
    print(f"initial_real_monthly_salary: {initial_real_monthly_salary:,.2f}")
    print(f"salary_inflation_adjustment_factor: {salary_inflation_adjustment_factor}")
    print(f"salary_start_year_idx: {salary_start_year_idx}")
    print(f"salary_end_year_idx: {salary_end_year_idx}")

    print("\n--- Economic Assumptions ---")
    print(f"stock_mu: {stock_mu:.4f}, stock_sigma: {stock_sigma:.4f}")
    print(f"bond_mu: {bond_mu:.4f}, bond_sigma: {bond_sigma:.4f}")
    print(f"str_mu: {str_mu:.4f}, str_sigma: {str_sigma:.4f}")
    print(f"fun_mu: {fun_mu:.4f}, fun_sigma: {fun_sigma:.4f}")
    print(
        f"real_estate_mu: {real_estate_mu:.4f}, real_estate_sigma: {real_estate_sigma:.4f}"
    )
    print(f"mu_pi: {mu_pi:.4f}, sigma_pi: {sigma_pi:.4f}")

    print("\n--- Derived Log-Normal Parameters ---")
    print(
        f"mu_log_stocks: {mu_log_stocks:.6f}, sigma_log_stocks: {sigma_log_stocks:.6f}"
    )
    print(f"mu_log_bonds: {mu_log_bonds:.6f}, sigma_log_bonds: {sigma_log_bonds:.6f}")
    print(f"mu_log_str: {mu_log_str:.6f}, sigma_log_str: {sigma_log_str:.6f}")
    print(f"mu_log_fun: {mu_log_fun:.6f}, sigma_log_fun: {sigma_log_fun:.6f}")
    print(
        f"mu_log_real_estate: {mu_log_real_estate:.6f}, "  # Fixed implicit concatenation
        + f"sigma_log_real_estate: {sigma_log_real_estate:.6f}"
    )
    print(
        f"mu_log_inflation: {mu_log_inflation:.6f}, sigma_log_inflation: {sigma_log_inflation:.6f}"
    )  # Renamed

    print("\n--- Portfolio Allocations ---")
    print(f"rebalancing_trigger_year_idx: {rebalancing_trigger_year_idx}")
    print(
        f"phase1_stocks_weight: {phase1_stocks_weight:.4f}, phase1_bonds_weight: {phase1_bonds_weight:.4f}, "  # Fixed implicit concatenation
        + f"phase1_str_weight: {phase1_str_weight:.4f}, phase1_fun_weight: {phase1_fun_weight:.4f}, "
        + f"phase1_real_estate_weight: {phase1_real_estate_weight:.4f}"
    )
    print(  # Corrected variable names to phase2_*_weight
        f"phase2_stocks_weight: {phase2_stocks_weight:.4f}, phase2_bonds_weight: {phase2_bonds_weight:.4f}, "  # Fixed implicit concatenation
        + f"phase2_str_weight: {phase2_str_weight:.4f}, phase2_fun_weight: {phase2_fun_weight:.4f}, "
        + f"phase2_real_estate_weight: {phase2_real_estate_weight:.4f}"
    )

    print("\n--- Initial Asset Values ---")
    print(f"initial_stocks_value: {initial_stocks_value:,.2f}")
    print(f"initial_bonds_value: {initial_bonds_value:,.2f}")
    print(f"initial_str_value: {initial_str_value:,.2f}")
    print(f"initial_fun_value: {initial_fun_value:,.2f}")
    print(f"initial_real_estate_value: {initial_real_estate_value:,.2f}")

    print("\n--- Simulation Parameters ---")
    print(f"num_simulations: {num_simulations}")
    # print(f"random_seed: {random_seed}") # Not used

    print("\n--- End of Parameter Assignment and Verification ---")

    # --- 6. Run Monte Carlo Simulations ---
    simulation_results: list[SimulationRunResult] = []  # Use TypedDict for list content

    spinner: itertools.cycle[str] = itertools.cycle(["-", "\\", "|", "/"])
    start_time: float = time.time()

    print(
        f"\nRunning {num_simulations} Monte Carlo simulations (T={total_retirement_years} years)..."
    )
    for i in range(num_simulations):
        result: SimulationRunResult = (
            run_single_fire_simulation(  # Use TypedDict for result
                initial_investment,
                initial_bank_balance,
                total_retirement_months,
                total_retirement_years,
                initial_real_monthly_expenses,
                # Ensure these are passed as lists of tuples, matching simulation.py
                list(planned_contributions),
                list(planned_extra_expenses),
                initial_real_monthly_pension,
                pension_inflation_adjustment_factor,
                pension_start_year_idx,
                initial_real_monthly_salary,
                salary_inflation_adjustment_factor,
                salary_start_year_idx,
                salary_end_year_idx,
                mu_log_inflation,  # Renamed
                sigma_log_inflation,  # Renamed
                rebalancing_trigger_year_idx,  # Renamed
                phase1_stocks_weight,
                phase1_bonds_weight,
                phase1_str_weight,
                phase1_fun_weight,
                phase1_real_estate_weight,
                phase2_stocks_weight,
                phase2_bonds_weight,
                phase2_str_weight,
                phase2_fun_weight,
                phase2_real_estate_weight,
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
                initial_real_monthly_contribution,
                initial_real_house_cost,
                ter_annual_percentage,
                shock_events,
            )
        )
        simulation_results.append(result)

        elapsed_time: float = time.time() - start_time
        sys.stdout.write(
            f"\r{next(spinner)} Running sim {i + 1}/{num_simulations} | "  # Fixed implicit concatenation
            + f"Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    end_simulation_time: float = time.time()
    total_simulation_elapsed_time: float = end_simulation_time - start_time

    print(
        f"\nMonte Carlo Simulation Complete. "  # Fixed implicit concatenation
        + f"Total time elapsed: {total_simulation_elapsed_time:.2f} seconds."
    )

    # --- 7. Perform Analysis and Prepare Plotting Data ---
    results_df: pd.DataFrame  # Explicitly type as pandas DataFrame
    plot_data: PlotDataDict  # Explicitly type using the TypedDict
    results_df, plot_data = analysis.perform_analysis_and_prepare_plots_data(
        simulation_results, num_simulations
    )

    # Generate and print the consolidated FIRE plan summary
    initial_total_wealth: float = (
        initial_investment + initial_bank_balance
    )  # Use new variable names
    fire_summary_string: str = analysis.generate_fire_plan_summary(
        simulation_results,
        initial_total_wealth,
        total_retirement_years,  # Use new variable name
    )

    # Print the consolidated summary, including the total simulation time here
    print(f"\nTotal simulations run: {num_simulations}")
    print(f"Total simulation time: {total_simulation_elapsed_time:.2f} seconds")
    print(fire_summary_string)

    # --- 8. Generate Plots ---
    print("\n--- Generating Plots ---")

    # Extract data from plot_data dictionary with explicit types
    failed_sims: pd.DataFrame = plot_data["failed_sims"]
    successful_sims: pd.DataFrame = plot_data["successful_sims"]
    plot_lines_data: list[analysis.PlotLineData] = plot_data["plot_lines_data"]
    bank_account_plot_indices: NDArray[np.intp] = plot_data["bank_account_plot_indices"]

    # Plotting Historical Distributions
    plot_retirement_duration_distribution(
        failed_sims, total_retirement_years
    )  # Use new variable name
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(
        results_df, plot_lines_data, mu_pi
    )  # Keep mu_pi for now as it's used in plots.py
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(
        results_df, bank_account_plot_indices, real_bank_lower_bound
    )
    plot_bank_account_trajectories_nominal(
        results_df, bank_account_plot_indices, real_bank_lower_bound
    )

    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
