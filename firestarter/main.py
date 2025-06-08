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
from firestarter.core.helpers import calculate_initial_asset_values, format_floats

# Import the legacy analysis module functions and its plotting data TypedDict
import firestarter.analysis.analysis as analysis
from firestarter.analysis.analysis import PlotDataDict

# Import plotting functions
from firestarter.plots.plots import (  # PATCH: use plots_v1 instead of plots
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
import firestarter.plots.plots as plots_module  # PATCH: use plots_v1 for set_output_dir
import pprint

from firestarter.core.simulation import SimulationBuilder


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
    shock_events = shocks.events  # <-- Pass Pydantic objects directly for type safety

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

    initial_assets = {
        "stocks": initial_stocks_value,
        "bonds": initial_bonds_value,
        "str": initial_str_value,
        "fun": initial_fun_value,
        "real_estate": initial_real_estate_value,
    }

    print(
        "All parameters successfully extracted and assigned to Python variables, "
        + "including derived ones."
    )

    # Prepare parameter summary for both console and report
    parameters_summary = {
        "deterministic_inputs": det_inputs.model_dump(),
        "economic_assumptions": econ_assumptions.model_dump(),
        "portfolio_rebalances": portfolio_rebalances.model_dump(),
        "shocks": shocks.model_dump(),
        "initial_assets": initial_assets,
        "simulation_parameters": sim_params.model_dump(),
    }

    # Sort shocks['events'] and portfolio_rebalances['rebalances'] by year, if present
    if "events" in parameters_summary["shocks"]:
        parameters_summary["shocks"]["events"] = sorted(
            parameters_summary["shocks"]["events"], key=lambda e: e["year"]
        )
    if "rebalances" in parameters_summary["portfolio_rebalances"]:
        parameters_summary["portfolio_rebalances"]["rebalances"] = sorted(
            parameters_summary["portfolio_rebalances"]["rebalances"], key=lambda r: r["year"]
        )

    # Format all floats to 4 decimal digits for console and report
    formatted_summary = format_floats(parameters_summary, ndigits=4)

    # --- Print all loaded parameters in a summary section ---
    print("\n--- Loaded Parameters Summary (from config.toml) ---")
    for section, values in formatted_summary.items():
        print(f"{section.replace('_', ' ').title()}:")
        pprint.pprint(values, sort_dicts=False)
        print()
    print("--- End of Parameters Summary ---\n")

    # --- 6. Run Monte Carlo Simulations ---
    simulation_results = []

    spinner = itertools.cycle(["-", "\\", "|", "/"])
    start_time = time.time()

    print(
        f"\nRunning {num_simulations} Monte Carlo simulations "
        + f"(T={det_inputs.years_to_simulate} years)..."
    )
    for i in range(num_simulations):
        builder = SimulationBuilder.new()
        simulation = (
            builder.set_det_inputs(det_inputs)
            .set_econ_assumptions(econ_assumptions)
            .set_portfolio_rebalances(portfolio_rebalances)
            .set_shock_events(shock_events)
            .set_initial_assets(initial_assets)
            .build()
        )
        simulation.init()
        result = simulation.run()
        simulation_results.append(result)

        elapsed_time = time.time() - start_time
        sys.stdout.write(
            f"\r{next(spinner)} Running sim {i + 1}/{num_simulations} | "
            + f"Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    end_simulation_time = time.time()
    total_simulation_elapsed_time = end_simulation_time - start_time

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

    # Pass formatted_summary to generate_markdown_report
    report_path = generate_markdown_report(
        config_path=config_file_path,
        fire_stats=fire_stats,
        output_dir=os.path.join(output_root, "reports"),
        plots=plots,
        parameters_summary=formatted_summary,
    )

    print(f"\nMarkdown report generated: {report_path}")

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
