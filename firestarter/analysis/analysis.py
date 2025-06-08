# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
analysis.py

This module provides functions for performing post-simulation analysis
of the Monte Carlo financial independence (FIRE) plan results.

It encompasses two primary responsibilities:

1.  **Data Preparation for Plotting:** The `perform_analysis_and_prepare_plots_data`
    function processes raw simulation outcomes, calculates real final wealth,
    identifies key simulation paths (e.g., worst, best, and percentile ranges),
    and structures this data into a format suitable for visualization by the
    `plots` module. It focuses solely on preparing data and does not output
    detailed textual summaries.

2.  **Summary Generation:** The `generate_fire_plan_summary` function computes
    and formats high-level statistics of the overall simulation campaign,
    including the FIRE plan's success rate, average duration of failed simulations,
    and detailed metrics (like final wealth and CAGR) for the worst, average,
    and best successful scenarios, along with their final asset allocations.
    This summary is returned as a readable string.

The module utilizes pandas for efficient data manipulation and numpy for
numerical operations on the simulation results.
"""

import pandas as pd
import numpy as np
from typing import TypedDict  # Import TypedDict for structured dictionaries
from numpy.typing import NDArray

from firestarter.core.helpers import calculate_cagr
from firestarter.core.simulation import (
    SimulationRunResult,
)  # Import the TypedDict from simulation.py


# Define TypedDicts for structured data within analysis and for plotting
class PlotLineData(TypedDict):
    """Represents data for a single line to be plotted in wealth evolution charts."""

    sim_idx: int
    label: str
    color: str
    linewidth: float


class PlotDataDict(TypedDict):
    """
    Represents the full dictionary of data prepared for plotting by the analysis module.
    `results_df` is `Any` because its type (pd.DataFrame) is specific to pandas,
    and we are not creating a stub for it here.
    """

    results_df: pd.DataFrame  # Explicitly use pd.DataFrame
    successful_sims: pd.DataFrame
    failed_sims: pd.DataFrame
    plot_lines_data: list[PlotLineData]
    bank_account_plot_indices: NDArray[np.intp]  # Use np.intp for index arrays


def perform_analysis_and_prepare_plots_data(
    simulation_results: list[SimulationRunResult],  # Use the imported TypedDict
    num_simulations: int,
) -> tuple[pd.DataFrame, PlotDataDict]:  # Explicitly use pd.DataFrame and the new TypedDict
    """
    Performs post-simulation analysis, identifies key scenarios, and prepares data
    structures needed for comprehensive plotting. This function now focuses ONLY
    on preparing data for plotting and does NOT print detailed scenario-specific
    analyses (like allocations), which are moved to generate_fire_plan_summary.

    Args:
        simulation_results (list[SimulationRunResult]): Raw results from run_single_fire_simulation.
        num_simulations (int): Total number of simulations run.

    Returns:
        tuple: (results_df, plot_data_dict)
               results_df (pd.DataFrame): DataFrame of all simulation results
               with calculated metrics.
               plot_data_dict (PlotDataDict): Dictionary containing data for all plots.
    """

    # Convert list of TypedDicts to DataFrame. Pandas will handle the unpacking.
    results_df: pd.DataFrame = pd.DataFrame(simulation_results)

    real_final_wealths_all_sims: list[float] = []
    for row_tuple in results_df.itertuples(index=True, name="SimulationRow"):
        row_success: bool = row_tuple.success
        row_final_investment: float = row_tuple.final_investment
        row_final_bank_balance: float = row_tuple.final_bank_balance
        row_annual_inflations_seq: NDArray[np.float64] = row_tuple.annual_inflations_seq

        if row_success:
            cumulative_inflation_factor: np.float64 = (
                np.prod(1.0 + row_annual_inflations_seq)
                if len(row_annual_inflations_seq) > 0
                else 1.0
            )
            real_wealth: float = (
                row_final_investment + row_final_bank_balance
            ) / cumulative_inflation_factor
        else:
            real_wealth = 0.0
        real_final_wealths_all_sims.append(real_wealth)

    results_df["real_final_wealth"] = real_final_wealths_all_sims

    # --- Prepare data for Time Evolution Samples (Wealth) ---
    print("\n--- Preparing Data for Time Evolution Samples (Wealth & Bank Account) ---")
    plot_lines_data: list[PlotLineData] = []

    successful_sims_for_plotting: pd.DataFrame = results_df[results_df["success"]]
    all_sims_sorted_by_real_wealth: pd.DataFrame = results_df.sort_values(
        by="real_final_wealth", ascending=True
    )

    if not all_sims_sorted_by_real_wealth.empty:
        worst_sim_row_for_plot: pd.Series = all_sims_sorted_by_real_wealth.iloc[0]
        worst_sim_idx_for_plot: int = worst_sim_row_for_plot.name
        if not bool(worst_sim_row_for_plot["success"]):
            plot_lines_data.append(
                PlotLineData(
                    sim_idx=worst_sim_idx_for_plot,
                    label=(
                        f"Worst Case (Failed Year "
                        f"{worst_sim_row_for_plot['months_lasted'] / 12.0:.1f})"
                    ),
                    color="red",
                    linewidth=2.5,
                )
            )
        else:
            plot_lines_data.append(
                PlotLineData(
                    sim_idx=worst_sim_idx_for_plot,
                    label=(
                        f"Worst Successful (Final Real: "
                        f"{worst_sim_row_for_plot['real_final_wealth']:,.0f}€)"
                    ),
                    color="darkred",
                    linewidth=2.0,
                )
            )

    if len(successful_sims_for_plotting) > 0:
        successful_sims_sorted_for_plotting: pd.DataFrame = (
            successful_sims_for_plotting.sort_values(by="real_final_wealth", ascending=True)
        )

        percentile_bins: list[int] = [0, 20, 40, 60, 80, 100]
        num_samples_per_bin: int = 5

        for i in range(len(percentile_bins) - 1):
            lower_percentile: int = percentile_bins[i]
            upper_percentile: int = percentile_bins[i + 1]

            start_idx_in_sorted: int = int(
                np.percentile(
                    np.arange(len(successful_sims_sorted_for_plotting), dtype=np.intp),
                    lower_percentile,
                )
            )
            if upper_percentile == 100:
                end_idx_in_sorted: int = len(successful_sims_sorted_for_plotting)
            else:
                end_idx_in_sorted: int = int(
                    np.percentile(
                        np.arange(len(successful_sims_sorted_for_plotting), dtype=np.intp),
                        upper_percentile,
                    )
                )

            range_indices: list[int] = successful_sims_sorted_for_plotting.iloc[
                start_idx_in_sorted:end_idx_in_sorted
            ].index.tolist()

            existing_indices: list[int] = [data["sim_idx"] for data in plot_lines_data]
            range_indices = [idx for idx in range_indices if idx not in existing_indices]

            if len(range_indices) > 0:
                sampled_indices: list[int] = np.random.choice(
                    range_indices,
                    min(len(range_indices), num_samples_per_bin),
                    replace=False,
                ).tolist()

                current_color: str
                if upper_percentile <= 20:
                    current_color = "darkorange"
                elif upper_percentile <= 40:
                    current_color = "gold"
                elif upper_percentile <= 60:
                    current_color = "forestgreen"
                elif upper_percentile <= 80:
                    current_color = "dodgerblue"
                else:
                    current_color = "mediumblue"

                for j, sim_idx in enumerate(sampled_indices):
                    label_to_use: str = (
                        f"{lower_percentile}-{upper_percentile}th Percentile Range"
                        if j == 0
                        else "_nolegend_"
                    )
                    plot_lines_data.append(
                        PlotLineData(
                            sim_idx=sim_idx,
                            label=label_to_use,
                            color=current_color,
                            linewidth=1.0,
                        )
                    )

    if not successful_sims_for_plotting.empty:
        best_sim_row_for_plot: pd.Series = successful_sims_sorted_for_plotting.iloc[-1]
        best_sim_idx_for_plot: int = best_sim_row_for_plot.name
        plot_lines_data.append(
            PlotLineData(
                sim_idx=best_sim_idx_for_plot,
                label=(
                    f"Best Successful (Final Real: "
                    f"{best_sim_row_for_plot['real_final_wealth']:,.0f}€)"
                ),
                color="green",
                linewidth=2.5,
            )
        )

    # --- Prepare data for Bank Account Trajectories ---
    # Use the same indices as plot_lines_data for bank account trajectories
    bank_account_plot_indices: NDArray[np.intp] = np.array(
        [d["sim_idx"] for d in plot_lines_data], dtype=np.intp
    )

    plot_data_dict: PlotDataDict = PlotDataDict(
        results_df=results_df,
        successful_sims=successful_sims_for_plotting,
        failed_sims=results_df[~results_df["success"]],
        plot_lines_data=plot_lines_data,
        bank_account_plot_indices=bank_account_plot_indices,
    )

    return results_df, plot_data_dict


def generate_fire_plan_summary(
    simulation_results: list[SimulationRunResult],
    initial_total_wealth_nominal: float,
    total_retirement_years: int,
) -> str:
    """
    Generate a summary of the FIRE simulation results.

    - Computes statistics such as success rate, average months failed, and final wealth.
    - Reports final allocations for worst, average, and best successful scenarios.
    - All allocations refer to liquid assets only; real estate is tracked separately.
    """
    successful_simulations_count: int = 0
    failed_simulations_count: int = 0
    months_lasted_in_failed_simulations: list[int] = []

    # Store the actual result TypedDicts for successful simulations to retrieve
    # allocation data later
    successful_results_data: list[SimulationRunResult] = []

    # Iterate through all simulation results to collect data for the summary
    for result in simulation_results:
        # Unpack directly from TypedDict. Pyright can now track types.
        success: bool = result["success"]
        months_lasted: int = result["months_lasted"]

        if success:
            successful_simulations_count += 1
            successful_results_data.append(result)
        else:
            failed_simulations_count += 1
            months_lasted_in_failed_simulations.append(months_lasted)

    # --- Summary Calculations ---
    total_simulations: int = len(simulation_results)
    fire_success_rate: float = (
        (float(successful_simulations_count) / total_simulations) * 100.0
        if total_simulations > 0
        else 0.0
    )

    avg_months_failed: float = (
        float(np.mean(months_lasted_in_failed_simulations)) if failed_simulations_count > 0 else 0.0
    )

    # Initialize variables to hold the *full result TypedDicts* for key scenarios
    worst_successful_result: SimulationRunResult | None = None
    average_successful_result: SimulationRunResult | None = None
    best_successful_result: SimulationRunResult | None = None

    if successful_simulations_count > 0:
        temp_successful_sorted: list[tuple[float, SimulationRunResult]] = []
        for res in successful_results_data:
            final_nom_wealth: float = res["final_investment"] + res["final_bank_balance"]
            annual_infl: NDArray[np.float64] = res["annual_inflations_seq"]
            cum_infl_factor: float = (
                np.prod(1.0 + annual_infl) if total_retirement_years > 0 else 1.0
            )
            real_wealth: float = final_nom_wealth / float(cum_infl_factor)
            temp_successful_sorted.append((real_wealth, res))

        temp_successful_sorted.sort(key=lambda x: x[0])

        worst_successful_result = temp_successful_sorted[0][1]
        best_successful_result = temp_successful_sorted[-1][1]

        median_real_wealth_among_successful: float = float(
            np.median([x[0] for x in temp_successful_sorted])
        )

        closest_to_median: tuple[float, SimulationRunResult] = min(
            temp_successful_sorted,
            key=lambda x: abs(x[0] - median_real_wealth_among_successful),
        )
        average_successful_result = closest_to_median[1]

    def _format_allocations_as_percentages(
        allocations_nominal_dict: dict[str, float],
    ) -> str:
        if not allocations_nominal_dict:
            return "N/A"

        total_nominal: float = sum(allocations_nominal_dict.values())
        if total_nominal == 0.0:
            return "All zero"

        percentage_strings: list[str] = []
        for asset, nom_val in allocations_nominal_dict.items():
            percentage: float = (nom_val / total_nominal) * 100.0
            if percentage >= 0.1 or (percentage == 0.0 and nom_val == 0.0):
                percentage_strings.append(f"{asset}: {percentage:.1f}%")
            elif percentage > 0.0:
                percentage_strings.append(f"{asset}: <0.1%")
        return ", ".join(percentage_strings)

    summary_lines: list[str] = [
        "\n--- FIRE Plan Simulation Summary ---",
        f"FIRE Plan Success Rate: {fire_success_rate:.2f}%",
        f"Number of failed simulations: {failed_simulations_count}",
    ]
    if failed_simulations_count > 0:
        summary_lines.append(
            "Average months lasted in failed simulations: " + f"{avg_months_failed:.1f}"
        )

    if successful_simulations_count > 0:
        summary_lines.append("\n--- Successful Cases Details ---")

        # Worst Successful Case
        if worst_successful_result:
            final_total_wealth_nominal: float = (
                worst_successful_result["final_investment"]
                + worst_successful_result["final_bank_balance"]
            )
            cum_infl_factor_np: float = (
                np.prod(1.0 + worst_successful_result["annual_inflations_seq"])
                if total_retirement_years > 0
                else 1.0
            )
            final_total_wealth_real: float = final_total_wealth_nominal / float(cum_infl_factor_np)
            cagr: float = calculate_cagr(
                initial_total_wealth_nominal,
                final_total_wealth_nominal,
                total_retirement_years,
            )

            summary_lines.append("Worst Successful Case:")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(
                "  Final Allocations: "
                + f"{_format_allocations_as_percentages(
                        worst_successful_result['final_allocations_nominal']
                    )}"
            )

        # Average Successful Case
        if average_successful_result:
            final_total_wealth_nominal: float = (
                average_successful_result["final_investment"]
                + average_successful_result["final_bank_balance"]
            )
            cum_infl_factor_np: float = (
                np.prod(1.0 + average_successful_result["annual_inflations_seq"])
                if total_retirement_years > 0
                else 1.0
            )
            final_total_wealth_real: float = final_total_wealth_nominal / float(cum_infl_factor_np)
            cagr: float = calculate_cagr(
                initial_total_wealth_nominal,
                final_total_wealth_nominal,
                total_retirement_years,
            )

            summary_lines.append("\nAverage Successful Case:")
            summary_lines.append("  (Simulation closest to median real final wealth)")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(
                "  Final Allocations: "
                + f"{_format_allocations_as_percentages(
                        average_successful_result['final_allocations_nominal']
                    )}"
            )

        # Best Successful Case
        if best_successful_result:
            final_total_wealth_nominal: float = (
                best_successful_result["final_investment"]
                + best_successful_result["final_bank_balance"]
            )
            cum_infl_factor_np: float = (
                np.prod(1.0 + best_successful_result["annual_inflations_seq"])
                if total_retirement_years > 0
                else 1.0
            )
            final_total_wealth_real: float = final_total_wealth_nominal / float(cum_infl_factor_np)
            cagr: float = calculate_cagr(
                initial_total_wealth_nominal,
                final_total_wealth_nominal,
                total_retirement_years,
            )

            summary_lines.append("\nBest Successful Case:")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(
                "  Final Allocations: "
                + f"{_format_allocations_as_percentages(
                        best_successful_result['final_allocations_nominal']
                    )}"
            )

    else:
        summary_lines.append("\nNo successful simulations to report details.")

    stats_dict = {
        "success_rate": fire_success_rate,
        "failed_simulations_count": failed_simulations_count,
        "avg_months_failed": avg_months_failed,
    }

    if worst_successful_result:
        ws_nom = (
            worst_successful_result["final_investment"]
            + worst_successful_result["final_bank_balance"]
        )
        ws_cum_infl = (
            np.prod(1.0 + worst_successful_result["annual_inflations_seq"])
            if total_retirement_years > 0
            else 1.0
        )
        ws_real = ws_nom / float(ws_cum_infl)
        ws_cagr = calculate_cagr(
            initial_total_wealth_nominal,
            ws_nom,
            total_retirement_years,
        )
        stats_dict["worst_successful"] = {
            "final_wealth_nominal": ws_nom,
            "final_wealth_real": ws_real,
            "cagr": ws_cagr,
            "allocations": _format_allocations_as_percentages(
                worst_successful_result["final_allocations_nominal"]
            ),
        }
    if average_successful_result:
        av_nom = (
            average_successful_result["final_investment"]
            + average_successful_result["final_bank_balance"]
        )
        av_cum_infl = (
            np.prod(1.0 + average_successful_result["annual_inflations_seq"])
            if total_retirement_years > 0
            else 1.0
        )
        av_real = av_nom / float(av_cum_infl)
        av_cagr = calculate_cagr(
            initial_total_wealth_nominal,
            av_nom,
            total_retirement_years,
        )
        stats_dict["average_successful"] = {
            "final_wealth_nominal": av_nom,
            "final_wealth_real": av_real,
            "cagr": av_cagr,
            "allocations": _format_allocations_as_percentages(
                average_successful_result["final_allocations_nominal"]
            ),
        }
    if best_successful_result:
        bs_nom = (
            best_successful_result["final_investment"]
            + best_successful_result["final_bank_balance"]
        )
        bs_cum_infl = (
            np.prod(1.0 + best_successful_result["annual_inflations_seq"])
            if total_retirement_years > 0
            else 1.0
        )
        bs_real = bs_nom / float(bs_cum_infl)
        bs_cagr = calculate_cagr(
            initial_total_wealth_nominal,
            bs_nom,
            total_retirement_years,
        )
        stats_dict["best_successful"] = {
            "final_wealth_nominal": bs_nom,
            "final_wealth_real": bs_real,
            "cagr": bs_cagr,
            "allocations": _format_allocations_as_percentages(
                best_successful_result["final_allocations_nominal"]
            ),
        }

    return "\n".join(summary_lines), stats_dict
