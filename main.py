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
import pandas as pd

# Import helper functions
from helpers import calculate_log_normal_params, calculate_initial_asset_values

# Import the main simulation function and its return type
from simulation import run_single_fire_simulation, SimulationRunResult

# Import the new analysis module functions and its plotting data TypedDict
import analysis
from analysis import PlotDataDict

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

# Import the DeterministicInputs Pydantic model
from config import DeterministicInputs


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

    # --- Pydantic: Load and validate deterministic inputs ---
    # This replaces the manual extraction of 'det_inputs' dictionary and individual assignments
    deterministic_inputs: DeterministicInputs = DeterministicInputs(
        **config_data["deterministic_inputs"]
    )

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
    mu_pi: float = eco_assumptions["mu_pi"]
    sigma_pi: float = eco_assumptions["sigma_pi"]

    # Load historical shock events
    shocks_config: dict[str, Any] = config_data.get("shocks", {})
    shock_events: list[dict[str, Any]] = shocks_config.get("events", [])

    port_allocs: dict[str, Any] = config_data["portfolio_allocations"]
    rebalancing_trigger_year_idx: int = port_allocs["rebalancing_year_idx"]
    phase1_stocks_weight: float = port_allocs["w_p1_stocks"]
    phase1_bonds_weight: float = port_allocs["w_p1_bonds"]
    phase1_str_weight: float = port_allocs["w_p1_str"]
    phase1_fun_weight: float = port_allocs["w_p1_fun"]
    phase1_real_estate_weight: float = port_allocs["w_p1_real_estate"]
    phase2_stocks_weight: float = port_allocs["w_p2_stocks"]
    phase2_bonds_weight: float = port_allocs["w_p2_bonds"]
    phase2_str_weight: float = port_allocs["w_p2_str"]
    phase2_fun_weight: float = port_allocs["w_p2_fun"]
    phase2_real_estate_weight: float = port_allocs["w_p2_real_estate"]

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
    assert np.isclose(p2_sum, 1.0), f"Phase 2 weights sum to {p2_sum:.4f}, not 1.0."
    print("Portfolio weights (w_p1, w_p2) successfully validated: sum to 1.0.")

    assert (
        deterministic_inputs.real_bank_upper_bound
        >= deterministic_inputs.real_bank_lower_bound
    ), (
        f"Bounds invalid: Upper ({deterministic_inputs.real_bank_upper_bound:,.0f}) "
        + f"< Lower ({deterministic_inputs.real_bank_lower_bound:,.0f})."
    )
    print("Bank account bounds successfully validated: Upper bound >= Lower bound.")

    sim_params: dict[str, Any] = config_data["simulation_parameters"]
    num_simulations: int = sim_params["num_simulations"]

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
        mu_log_inflation,
        sigma_log_inflation,
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
        mu_pi,
        sigma_pi,
    )

    (
        initial_stocks_value,
        initial_bonds_value,
        initial_str_value,
        initial_fun_value,
        initial_real_estate_value,
    ) = calculate_initial_asset_values(
        deterministic_inputs.i0,
        phase1_stocks_weight,
        phase1_bonds_weight,
        phase1_str_weight,
        phase1_fun_weight,
        phase1_real_estate_weight,
    )

    print(
        "All parameters successfully extracted and assigned to Python variables, "
        + "including derived ones."
    )

    # --- Print all parameters for verification ---
    # These print statements now pull directly from the deterministic_inputs object
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"initial_investment: {deterministic_inputs.i0:,.2f}")
    print(f"initial_bank_balance: {deterministic_inputs.b0:,.2f}")
    print(f"real_bank_lower_bound: {deterministic_inputs.real_bank_lower_bound:,.2f}")
    print(f"real_bank_upper_bound: {deterministic_inputs.real_bank_upper_bound:,.2f}")
    print(f"total_retirement_years: {deterministic_inputs.t_ret_years}")
    print(
        f"total_retirement_months: {deterministic_inputs.t_ret_years * 12}"
    )  # Derived value
    print(
        f"initial_real_monthly_expenses: {deterministic_inputs.x_real_monthly_initial:,.2f}"
    )
    print(f"planned_extra_expenses: {deterministic_inputs.x_planned_extra}")
    print(f"planned_contributions: {deterministic_inputs.c_planned}")
    print(
        f"initial_real_monthly_contribution: {deterministic_inputs.c_real_monthly_initial:,.2f}"
    )
    print(f"ter_annual_percentage: {deterministic_inputs.ter_annual_percentage:.4f}")
    print(f"initial_real_house_cost: {deterministic_inputs.h0_real_cost:,.2f}")
    print(f"initial_real_monthly_pension: {deterministic_inputs.p_real_monthly:,.2f}")
    print(
        f"pension_inflation_adjustment_factor: {deterministic_inputs.pension_inflation_adjustment_factor}"
    )
    print(f"pension_start_year_idx: {deterministic_inputs.y_p_start_idx}")
    print(f"initial_real_monthly_salary: {deterministic_inputs.s_real_monthly:,.2f}")
    print(
        f"salary_inflation_adjustment_factor: {deterministic_inputs.salary_inflation_adjustment_factor}"
    )
    print(f"salary_start_year_idx: {deterministic_inputs.y_s_start_idx}")
    print(f"salary_end_year_idx: {deterministic_inputs.y_s_end_idx}")

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
        f"mu_log_real_estate: {mu_log_real_estate:.6f}, "
        + f"sigma_log_real_estate: {sigma_log_real_estate:.6f}"
    )
    print(
        f"mu_log_inflation: {mu_log_inflation:.6f}, sigma_log_inflation: {sigma_log_inflation:.6f}"
    )

    print("\n--- Portfolio Allocations ---")
    print(f"rebalancing_trigger_year_idx: {rebalancing_trigger_year_idx}")
    print(
        f"phase1_stocks_weight: {phase1_stocks_weight:.4f}, phase1_bonds_weight: {phase1_bonds_weight:.4f}, "
        + f"phase1_str_weight: {phase1_str_weight:.4f}, phase1_fun_weight: {phase1_fun_weight:.4f}, "
        + f"phase1_real_estate_weight: {phase1_real_estate_weight:.4f}"
    )
    print(
        f"phase2_stocks_weight: {phase2_stocks_weight:.4f}, phase2_bonds_weight: {phase2_bonds_weight:.4f}, "
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

    print("\n--- End of Parameter Assignment and Verification ---")

    # --- 6. Run Monte Carlo Simulations ---
    simulation_results: list[SimulationRunResult] = []

    spinner: itertools.cycle[str] = itertools.cycle(["-", "\\", "|", "/"])
    start_time: float = time.time()

    print(
        f"\nRunning {num_simulations} Monte Carlo simulations (T={deterministic_inputs.t_ret_years} years)..."
    )
    for i in range(num_simulations):
        result: SimulationRunResult = run_single_fire_simulation(
            deterministic_inputs,
            mu_log_inflation,
            sigma_log_inflation,
            rebalancing_trigger_year_idx,
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
            shock_events,
        )
        simulation_results.append(result)

        elapsed_time: float = time.time() - start_time
        sys.stdout.write(
            f"\r{next(spinner)} Running sim {i + 1}/{num_simulations} | "
            + f"Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    end_simulation_time: float = time.time()
    total_simulation_elapsed_time: float = end_simulation_time - start_time

    print(
        "\nMonte Carlo Simulation Complete. "
        + f"Total time elapsed: {total_simulation_elapsed_time:.2f} seconds."
    )

    # --- 7. Perform Analysis and Prepare Plotting Data ---
    results_df: pd.DataFrame
    plot_data: PlotDataDict
    results_df, plot_data = analysis.perform_analysis_and_prepare_plots_data(
        simulation_results, num_simulations
    )

    # Generate and print the consolidated FIRE plan summary
    initial_total_wealth: float = deterministic_inputs.i0 + deterministic_inputs.b0
    fire_summary_string: str = analysis.generate_fire_plan_summary(
        simulation_results,
        initial_total_wealth,
        deterministic_inputs.t_ret_years,
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
    plot_retirement_duration_distribution(failed_sims, deterministic_inputs.t_ret_years)
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(results_df, plot_lines_data, mu_pi)
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(
        results_df,
        bank_account_plot_indices,
        deterministic_inputs.real_bank_lower_bound,
    )
    plot_bank_account_trajectories_nominal(
        results_df,
        bank_account_plot_indices,
        deterministic_inputs.real_bank_lower_bound,
    )

    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
