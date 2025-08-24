#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Main entry point for running FIRE Monte Carlo simulations.

- Loads configuration from TOML files.
- Validates parameters using Pydantic models.
- Runs simulations and generates reports and plots.
"""

import sys
import os
from typing import Any
import time
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# Import the DeterministicInputs Pydantic model
from firecast.config.config import (
    Config,
    DeterministicInputs,
    PortfolioRebalance,
    SimulationParameters,
    Shock,
)
from firecast.config.correlation_matrix import CorrelationMatrix

from firecast.reporting.markdown_report import generate_markdown_report
from firecast.reporting.console_report import print_console_summary
from firecast.reporting.graph_report import generate_all_plots

from firecast.core.simulation import SimulationBuilder


# Setup CLI argument parsing
parser = argparse.ArgumentParser(
    description="Analyze historical stock market index data for n-years rolling windows."
)
parser.add_argument(
    "-f",
    "--config",
    type=str,
    default="config.toml",
    help="Path to the configuration file.",
)
args = parser.parse_args()
CONFIG_FILE_PATH = args.config

# Limit thread usage for numpy/scipy libraries to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


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
    - Runs Monte Carlo simulations.
    - Performs analysis and generates reports and plots.
    """
    import multiprocessing

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Configuration file not found at '{CONFIG_FILE_PATH}'")
        sys.exit(1)

    config_data: dict[str, Any]
    try:
        with open(CONFIG_FILE_PATH, "rb") as f:
            config_data = tomllib.load(f)

        config = Config(**config_data)

    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading or parsing config file '{CONFIG_FILE_PATH}': {e}")
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
    correlation_matrix = config.correlation_matrix or CorrelationMatrix(assets_order=[], matrix=[])
    portfolio_rebalances = config.portfolio_rebalances
    sim_params = config.simulation_parameters
    shocks = config.shocks or []
    num_simulations = sim_params.num_simulations

    print("All parameters successfully extracted.")

    # Run Monte Carlo simulations in parallel
    simulation_results = []
    start_time = time.time()
    print(f"\nRunning {num_simulations} Monte Carlo simulations")

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

    # Print simulation result summary
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
        config_path=CONFIG_FILE_PATH,
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
