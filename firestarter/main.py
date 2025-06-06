# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
main.py

Main entry point for running FIRE Monte Carlo simulations.

- Loads configuration from TOML files.
- Validates parameters using Pydantic models.
- Supports multiple scheduled portfolio rebalances.
- Runs simulations and generates reports and plots.
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
    PortfolioRebalances,
    SimulationParameters,
    Shocks,
)

# from firestarter.version import __version__
from firestarter.analysis.reporting import generate_markdown_report
import firestarter.plots.plots as plots_module


def main() -> None:
    """
    Main workflow for the FIRE simulation tool.

    - Loads and validates configuration.
    - Validates portfolio rebalance weights.
    - Runs Monte Carlo simulations using the current rebalance schedule.
    - Performs analysis and generates reports and plots.
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

    # --- Pydantic: Load and validate Portfolio Rebalances ---
    portfolio_rebalances: PortfolioRebalances = PortfolioRebalances(
        **config_data["portfolio_rebalances"]
    )
    # print(config_data["portfolio_rebalances"])

    # --- Pydantic: Load and validate Simulation Parameters ---
    sim_params: SimulationParameters = SimulationParameters(**config_data["simulation_parameters"])
    num_simulations: int = sim_params.num_simulations

    # --- Pydantic: Load and validate Shocks ---
    shocks: Shocks = Shocks(**config_data.get("shocks", {}))
    shock_events: list[dict[str, Any]] = [event.dict() for event in shocks.events]

    print("Number of rebalances:", len(portfolio_rebalances.rebalances))
    print("All rebalances:", portfolio_rebalances.rebalances)
    print(portfolio_rebalances.rebalances[0].dict())
    # Validate portfolio rebalance weights
    for reb in portfolio_rebalances.rebalances:
        reb_sum = reb.stocks + reb.bonds + reb.str + reb.fun
        assert np.isclose(
            reb_sum, 1.0
        ), f"Rebalance weights for year {reb.year} sum to {reb_sum:.4f}, not 1.0."
    print("All portfolio rebalance weights successfully validated: sum to 1.0 for each rebalance.")

    assert det_inputs.bank_upper_bound >= det_inputs.bank_lower_bound, (
        f"Bounds invalid: Upper ({det_inputs.bank_upper_bound:,.0f}) "
        + f"< Lower ({det_inputs.bank_lower_bound:,.0f})."
    )
    print("Bank account bounds successfully validated: Upper bound >= Lower bound.")

    # Remove all uses of portfolio_allocs and initial asset value calculation based on it.
    # If you need initial asset values, calculate them based on the first rebalance weights:
    first_reb = portfolio_rebalances.rebalances[0]
    (
        initial_stocks_value,
        initial_bonds_value,
        initial_str_value,
        initial_fun_value,
        initial_real_estate_value,
    ) = calculate_initial_asset_values(
        det_inputs.initial_investment,
        first_reb.stocks,
        first_reb.bonds,
        first_reb.str,
        first_reb.fun,
        0.0,  # real estate is always 0 in liquid allocations
    )

    print(
        "All parameters successfully extracted and assigned to Python variables, "
        + "including derived ones."
    )

    # --- Print all parameters for verification ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    print(f"initial_investment: {det_inputs.initial_investment:,.2f}")
    print(f"initial_bank_balance: {det_inputs.initial_bank_balance:,.2f}")
    print(f"bank_lower_bound: {det_inputs.bank_lower_bound:,.2f}")
    print(f"bank_upper_bound: {det_inputs.bank_upper_bound:,.2f}")
    print(f"years_to_simulate: {det_inputs.years_to_simulate}")
    print(f"total_retirement_months: {det_inputs.years_to_simulate * 12}")  # Derived value
    print(f"monthly_expenses: {det_inputs.monthly_expenses:,.2f}")
    print(f"planned_extra_expenses: {det_inputs.planned_extra_expenses}")
    print(f"planned_contributions: {det_inputs.planned_contributions}")
    print(f"monthly_investment_contribution: {det_inputs.monthly_investment_contribution:,.2f}")
    print(f"annual_fund_fee: {det_inputs.annual_fund_fee:.4f}")
    print(f"planned_house_purchase_cost: {det_inputs.planned_house_purchase_cost:,.2f}")
    print(f"monthly_pension: {det_inputs.monthly_pension:,.2f}")
    print("pension_inflation_factor: " f"{det_inputs.pension_inflation_factor}")
    print(f"pension_start_year: {det_inputs.pension_start_year}")
    print(f"monthly_salary: {det_inputs.monthly_salary:,.2f}")
    print("salary_inflation_factor: " f"{det_inputs.salary_inflation_factor}")
    print(f"salary_start_year: {det_inputs.salary_start_year}")
    print(f"salary_end_year: {det_inputs.salary_end_year}")

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
    print(f"pi_mu: {econ_assumptions.pi_mu:.4f}, pi_sigma: {econ_assumptions.pi_sigma:.4f}")

    print("\n--- Derived Log-Normal Parameters ---")
    for asset, (mu_log, sigma_log) in econ_assumptions.lognormal.items():
        print(f"{asset}: mu_log = {mu_log:.6f}, sigma_log = {sigma_log:.6f}")

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
        + f"(T={det_inputs.years_to_simulate} years)..."
    )
    for i in range(num_simulations):
        result: SimulationRunResult = run_single_fire_simulation(
            det_inputs,
            econ_assumptions,
            portfolio_rebalances,
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
    initial_total_wealth: float = det_inputs.initial_investment + det_inputs.initial_bank_balance
    fire_summary_string, fire_stats = analysis.generate_fire_plan_summary(
        simulation_results,
        initial_total_wealth,
        det_inputs.years_to_simulate,
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
    plot_retirement_duration_distribution(failed_sims, det_inputs.years_to_simulate)
    plot_final_wealth_distribution_nominal(successful_sims)
    plot_final_wealth_distribution_real(successful_sims)

    # Plotting Time Evolution Samples
    plot_wealth_evolution_samples_real(results_df, plot_lines_data, econ_assumptions.pi_mu)
    plot_wealth_evolution_samples_nominal(results_df, plot_lines_data)

    # Plotting Bank Account Trajectories
    plot_bank_account_trajectories_real(
        results_df,
        bank_account_plot_indices,
        det_inputs.bank_lower_bound,
        plot_lines_data,  # Pass plot_lines_data for color/label consistency
    )
    plot_bank_account_trajectories_nominal(
        results_df,
        bank_account_plot_indices,
        plot_lines_data,  # Pass plot_lines_data for color/label consistency
        det_inputs.bank_lower_bound,
    )

    print("\nAll requested plots generated and saved to the current directory.")

    print("\nAll plots generated. Displaying interactive windows. Close them to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
