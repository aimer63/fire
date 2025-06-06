# SPDX-FileCopyrightText: 2025 2024 Aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
from typing import Any
import tomllib
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
import pandas as pd

# Import helper functions
from firestarter.core.helpers import calculate_initial_asset_values

# Import the main simulation function and its return type
from firestarter.core.simulation import run_single_fire_simulation, SimulationRunResult

# Import the new analysis module functions and its plotting data TypedDict
import firestarter.analysis.analysis as analysis
from firestarter.analysis.analysis import PlotDataDict

# Import plotting functions
from firestarter.plots.plots import (
    plot_retirement_duration_distribution,
    plot_final_wealth_distribution_nominal,
    plot_final_wealth_distribution_real,
    plot_wealth_evolution_samples_real,
    plot_wealth_evolution_samples_nominal,
    plot_bank_account_trajectories_real,
    plot_bank_account_trajectories_nominal,
)

# Import the DeterministicInputs Pydantic model
from firestarter.config.config import (
    DeterministicInputs,
    EconomicAssumptions,
    PortfolioAllocations,
    SimulationParameters,
    Shocks,
)

# from firestarter.version import __version__
from firestarter.analysis.reporting import generate_markdown_report
import firestarter.plots.plots as plots_module


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

    output_root = config_data.get("paths", {}).get("output_root", "output")
    plots_module.set_output_dir(os.path.join(output_root, "plots"))

    print("Configuration file parsed successfully. Extracting parameters...")

    # --- Pydantic: Load and validate deterministic inputs ---
    det_inputs: DeterministicInputs = DeterministicInputs(**config_data["deterministic_inputs"])

    # --- Pydantic: Load and validate Economic Assumption ---
    econ_assumptions: EconomicAssumptions = EconomicAssumptions(
        **config_data["economic_assumptions"]
    )

    # --- Pydantic: Load and validate Portfolio Allocations ---
    portfolio_allocs: PortfolioAllocations = PortfolioAllocations(
        **config_data["portfolio_allocations"]
    )

    # --- Pydantic: Load and validate Simulation Parameters ---
    sim_params: SimulationParameters = SimulationParameters(**config_data["simulation_parameters"])
    num_simulations: int = sim_params.num_simulations

    # --- Pydantic: Load and validate Shocks ---
    shocks: Shocks = Shocks(**config_data.get("shocks", {}))
    shock_events: list[dict[str, Any]] = [event.dict() for event in shocks.events]

    # Validate portfolio weights
    p1_sum: float = (
        portfolio_allocs.w_p1_stocks
        + portfolio_allocs.w_p1_bonds
        + portfolio_allocs.w_p1_str
        + portfolio_allocs.w_p1_fun
        + portfolio_allocs.w_p1_real_estate
    )
    p2_sum: float = (
        portfolio_allocs.w_p2_stocks
        + portfolio_allocs.w_p2_bonds
        + portfolio_allocs.w_p2_str
        + portfolio_allocs.w_p2_fun
        + portfolio_allocs.w_p2_real_estate
    )

    assert np.isclose(p1_sum, 1.0), f"Phase 1 weights sum to {p1_sum:.4f}, not 1.0."
    assert np.isclose(p2_sum, 1.0), f"Phase 2 weights sum to {p2_sum:.4f}, not 1.0."
    print("Portfolio weights (w_p1, w_p2) successfully validated: sum to 1.0.")

    assert det_inputs.real_bank_upper_bound >= det_inputs.real_bank_lower_bound, (
        f"Bounds invalid: Upper ({det_inputs.real_bank_upper_bound:,.0f}) "
        + f"< Lower ({det_inputs.real_bank_lower_bound:,.0f})."
    )
    print("Bank account bounds successfully validated: Upper bound >= Lower bound.")

    (
        initial_stocks_value,
        initial_bonds_value,
        initial_str_value,
        initial_fun_value,
        initial_real_estate_value,
    ) = calculate_initial_asset_values(
        det_inputs.i0,
        portfolio_allocs.w_p1_stocks,
        portfolio_allocs.w_p1_bonds,
        portfolio_allocs.w_p1_str,
        portfolio_allocs.w_p1_fun,
        portfolio_allocs.w_p1_real_estate,
    )

    print(
        "All parameters successfully extracted and assigned to Python variables, "
        + "including derived ones."
    )

    # --- Print all parameters for verification ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"initial_investment: {det_inputs.i0:,.2f}")
    print(f"initial_bank_balance: {det_inputs.b0:,.2f}")
    print(f"real_bank_lower_bound: {det_inputs.real_bank_lower_bound:,.2f}")
    print(f"real_bank_upper_bound: {det_inputs.real_bank_upper_bound:,.2f}")
    print(f"total_retirement_years: {det_inputs.t_ret_years}")
    print(f"total_retirement_months: {det_inputs.t_ret_years * 12}")  # Derived value
    print(f"initial_real_monthly_expenses: {det_inputs.x_real_monthly_initial:,.2f}")
    print(f"planned_extra_expenses: {det_inputs.x_planned_extra}")
    print(f"planned_contributions: {det_inputs.c_planned}")
    print(f"initial_real_monthly_contribution: {det_inputs.c_real_monthly_initial:,.2f}")
    print(f"ter_annual_percentage: {det_inputs.ter_annual_percentage:.4f}")
    print(f"initial_real_house_cost: {det_inputs.h0_real_cost:,.2f}")
    print(f"initial_real_monthly_pension: {det_inputs.p_real_monthly:,.2f}")
    print(
        "pension_inflation_adjustment_factor: " f"{det_inputs.pension_inflation_adjustment_factor}"
    )
    print(f"pension_start_year_idx: {det_inputs.y_p_start_idx}")
    print(f"initial_real_monthly_salary: {det_inputs.s_real_monthly:,.2f}")
    print("salary_inflation_adjustment_factor: " f"{det_inputs.salary_inflation_adjustment_factor}")
    print(f"salary_start_year_idx: {det_inputs.y_s_start_idx}")
    print(f"salary_end_year_idx: {det_inputs.y_s_end_idx}")

    print("\n--- Economic Assumptions ---")
    print(
        f"stock_mu: {econ_assumptions.stock_mu:.4f}, "
        + f"stock_sigma: {econ_assumptions.stock_sigma:.4f}"
    )
    print(f"bond_mu: {econ_assumptions.bond_mu:.4f}, bond_sigma: {econ_assumptions.bond_sigma:.4f}")
    print(f"str_mu: {econ_assumptions.str_mu:.4f}, str_sigma: {econ_assumptions.str_sigma:.4f}")
    print(f"fun_mu: {econ_assumptions.fun_mu:.4f}, fun_sigma: {econ_assumptions.fun_sigma:.4f}")
    print(
        f"real_estate_mu: {econ_assumptions.real_estate_mu:.4f}, "
        + f"real_estate_sigma: {econ_assumptions.real_estate_sigma:.4f}"
    )
    print(f"mu_pi: {econ_assumptions.mu_pi:.4f}, sigma_pi: {econ_assumptions.sigma_pi:.4f}")

    print("\n--- Derived Log-Normal Parameters ---")
    for asset, (mu_log, sigma_log) in econ_assumptions.lognormal.items():
        print(f"{asset}: mu_log = {mu_log:.6f}, sigma_log = {sigma_log:.6f}")

    print("\n--- Portfolio Allocations ---")
    print(f"rebalancing_trigger_year_idx: {portfolio_allocs.rebalancing_year_idx}")
    print(
        f"phase1_stocks_weight: {portfolio_allocs.w_p1_stocks:.4f}, "
        + f"phase1_bonds_weight: {portfolio_allocs.w_p1_bonds:.4f}, "
        + f"phase1_str_weight: {portfolio_allocs.w_p1_str:.4f}, "
        + f"phase1_fun_weight: {portfolio_allocs.w_p1_fun:.4f}, "
        + f"phase1_real_estate_weight: {portfolio_allocs.w_p1_real_estate:.4f}"
    )
    print(
        f"phase2_stocks_weight: {portfolio_allocs.w_p2_stocks:.4f}, "
        + f"phase2_bonds_weight: {portfolio_allocs.w_p2_bonds:.4f}, "
        + f"phase2_str_weight: {portfolio_allocs.w_p2_str:.4f}, "
        + f"phase2_fun_weight: {portfolio_allocs.w_p2_fun:.4f}, "
        + f"phase2_real_estate_weight: {portfolio_allocs.w_p2_real_estate:.4f}"
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

    spinner = itertools.cycle(["-", "\\", "|", "/"])
    start_time: float = time.time()

    print(
        f"\nRunning {num_simulations} Monte Carlo simulations "
        + f"(T={det_inputs.t_ret_years} years)..."
    )
    for i in range(num_simulations):
        result: SimulationRunResult = run_single_fire_simulation(
            det_inputs,
            econ_assumptions,
            portfolio_allocs,
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
    initial_total_wealth: float = det_inputs.i0 + det_inputs.b0
    fire_summary_string, fire_stats = analysis.generate_fire_plan_summary(
        simulation_results,
        initial_total_wealth,
        det_inputs.t_ret_years,
    )

    # Print the consolidated summary, including the total simulation time here
    print(f"\nTotal simulations run: {num_simulations}")
    print(f"Total simulation time: {total_simulation_elapsed_time:.2f} seconds")
    print(fire_summary_string)

    plots = {
        "Retirement Duration Distribution": os.path.join(
            output_root, "plots", "retirement_duration_distribution.png"
        ),
        "Final Wealth Distribution (Nominal)": os.path.join(
            output_root, "plots", "final_wealth_distribution_nominal.png"
        ),
        "Final Wealth Distribution (Real)": os.path.join(
            output_root, "plots", "final_wealth_distribution_real.png"
        ),
        "Wealth Evolution Samples (Real)": os.path.join(
            output_root, "plots", "wealth_evolution_samples_real.png"
        ),
        "Wealth Evolution Samples (Nominal)": os.path.join(
            output_root, "plots", "wealth_evolution_samples_nominal.png"
        ),
        "Bank Account Trajectories (Real)": os.path.join(
            output_root, "plots", "bank_account_trajectories_real.png"
        ),
        "Bank Account Trajectories (Nominal)": os.path.join(
            output_root, "plots", "bank_account_trajectories_nominal.png"
        ),
    }

    # Pass output_root to generate_markdown_report for the reports subfolder
    report_path = generate_markdown_report(
        config_path=config_file_path,
        fire_stats=fire_stats,
        output_dir=os.path.join(output_root, "reports"),
        plots=plots,
    )

    print(f"\nMarkdown report generated: {report_path}")

    # --- 8. Generate Plots ---
    print("\n--- Generating Plots ---")

    # Extract data from plot_data dictionary with explicit types
    failed_sims: pd.DataFrame = plot_data["failed_sims"]
    successful_sims: pd.DataFrame = plot_data["successful_sims"]
    plot_lines_data: list[analysis.PlotLineData] = plot_data["plot_lines_data"]
    bank_account_plot_indices: NDArray[np.intp] = plot_data["bank_account_plot_indices"]

    # Plotting Historical Distributions
    plot_retirement_duration_distribution(failed_sims, det_inputs.t_ret_years)
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(results_df, plot_lines_data, econ_assumptions.mu_pi)
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(
        results_df,
        bank_account_plot_indices,
        det_inputs.real_bank_lower_bound,
        plot_lines_data,  # Pass plot_lines_data for color/label consistency
    )
    plot_bank_account_trajectories_nominal(
        results_df,
        bank_account_plot_indices,
        plot_lines_data,  # Pass plot_lines_data for color/label consistency
        det_inputs.real_bank_lower_bound,
    )

    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
