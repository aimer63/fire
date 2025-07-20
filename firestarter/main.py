#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
main.py

Main entry point for running FIRE Monte Carlo simulations.

- Loads configuration from TOML files.
- Validates parameters using Pydantic models.
- Runs simulations and generates reports and plots.
"""

import sys
import os
from typing import Any
import time
import shutil
from tqdm import tqdm
import tomllib
import numpy as np

# Import the DeterministicInputs Pydantic model
from firestarter.config.config import (
    Config,
    DeterministicInputs,
    PortfolioRebalance,
    SimulationParameters,
    Shock,
)
from firestarter.config.correlation_matrix import CorrelationMatrix

# from firestarter.version import __version__
from firestarter.reporting.markdown_report import generate_markdown_report
from firestarter.reporting.console_report import print_console_summary
from firestarter.reporting.graph_report import generate_all_plots


from firestarter.core.simulation import SimulationBuilder


from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_simulation(
    det_inputs: DeterministicInputs,
    assets: dict[str, Any],
    correlation_matrix: CorrelationMatrix,
    portfolio_rebalances: list[PortfolioRebalance],
    shock_events: list[Shock],
    sim_params: SimulationParameters,
) -> dict[str, Any]:
    builder = SimulationBuilder.new()
    simulation = (
        builder.set_det_inputs(det_inputs)
        .set_assets(assets)
        .set_correlation_matrix(correlation_matrix)
        .set_portfolio_rebalances(portfolio_rebalances)
        .set_shock_events(shock_events)
        .set_sim_params(sim_params)
        .build()
    )
    simulation.init()
    return simulation.run()


def main() -> None:
    """
    Main workflow for the FIRE simulation tool.

    - Loads and validates configuration.
    - Runs Monte Carlo simulations using the current rebalance schedule.
    - Performs analysis and generates reports and plots.
    """
    import multiprocessing

    # Config loading, parameter assignment, derived calculations, and assertions
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

        config = Config(**config_data)

    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading or parsing config file '{config_file_path}': {e}")
        sys.exit(1)
    except Exception as e:  # Catches Pydantic's ValidationError
        print(f"Error validating configuration: {e}")
        sys.exit(1)

    # Create output directories from the Paths model
    output_root = config.paths.output_root if config.paths else "output"
    os.makedirs(os.path.join(output_root, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "reports"), exist_ok=True)

    print("Configuration file loaded and validated successfully.")

    # Extract validated data from the config model for simulation
    det_inputs = config.deterministic_inputs
    assets = config.assets
    correlation_matrix = config.correlation_matrix or CorrelationMatrix(
        assets_order=[], matrix=[]
    )
    portfolio_rebalances = config.portfolio_rebalances
    sim_params = config.simulation_parameters
    shocks = config.shocks or []
    num_simulations = sim_params.num_simulations

    print("All parameters successfully extracted and assigned to Python variables, ")

    # Run Monte Carlo simulations in parallel
    simulation_results = []
    start_time = time.time()
    print(f"\nRunning {num_simulations} Monte Carlo simulations ")

    max_workers = multiprocessing.cpu_count()
    term_width = shutil.get_terminal_size().columns
    bar_width = max(40, term_width // 2)  # Use half terminal width, min 40

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                det_inputs,
                assets,
                correlation_matrix,
                portfolio_rebalances,
                shocks,
                sim_params,
            )
            for _ in range(num_simulations)
        ]
        for future in tqdm(
            as_completed(futures),
            total=num_simulations,
            desc="Simulations",
            ncols=bar_width,
        ):
            result = future.result()
            simulation_results.append(result)

    sys.stdout.write("\n")
    sys.stdout.flush()

    end_simulation_time = time.time()
    total_simulation_elapsed_time = end_simulation_time - start_time

    print(
        "\nMonte Carlo Simulation Complete. "
        + f"Total time elapsed: {total_simulation_elapsed_time:.2f} seconds."
    )

    # Print config parameters and simulation result summary
    print_console_summary(simulation_results, config.model_dump(exclude_none=True))

    # Prepare plot paths dictionary
    plots = {
        "Failed Duration Distribution": os.path.join(
            "..", "plots", "failed_duration_distribution.png"
        ),
        "Final Wealth Distribution (Nominal)": os.path.join(
            "..", "plots", "final_wealth_distribution_nominal.png"
        ),
        "Final Wealth Distribution (Real)": os.path.join(
            "..", "plots", "final_wealth_distribution_real.png"
        ),
        "Wealth Evolution Samples (Real)": os.path.join(
            "..", "plots", "wealth_evolution_samples_real.png"
        ),
        "Wealth Evolution Samples (Nominal)": os.path.join(
            "..", "plots", "wealth_evolution_samples_nominal.png"
        ),
        "Failed Wealth Evolution Samples (Real)": os.path.join(
            "..", "plots", "failed_wealth_evolution_samples_real.png"
        ),
        "Failed Wealth Evolution Samples (Nominal)": os.path.join(
            "..", "plots", "failed_wealth_evolution_samples_nominal.png"
        ),
        "Bank Account Trajectories (Real)": os.path.join(
            "..", "plots", "bank_account_trajectories_real.png"
        ),
        "Bank Account Trajectories (Nominal)": os.path.join(
            "..", "plots", "bank_account_trajectories_nominal.png"
        ),
    }

    # Generate markdown report
    print("\n--- Generating markdown report ---")
    generate_markdown_report(
        simulation_results=simulation_results,
        config=config.model_dump(exclude_none=True),
        config_path=config_file_path,
        output_dir=os.path.join(output_root, "reports"),
        plot_paths=plots,
    )
    print("\n--- Markdown report generated ---")

    print("\n--- Generating Plots ---")
    generate_all_plots(
        simulation_results=simulation_results,
        output_root=output_root,
        det_inputs=det_inputs,
    )
    print("\nAll plots generated and saved.")

    print(f"\nReports path: {os.path.join(output_root, 'reports')}")
    print(f"Plots path: {os.path.join(output_root, 'plots')}")

    print("\nDisplaying interactive plot windows. Close them to exit.")


if __name__ == "__main__":
    main()
