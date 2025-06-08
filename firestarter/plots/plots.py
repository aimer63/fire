# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
plots.py

This module is responsible for generating various visualizations and plots
from the financial simulation results. It uses Matplotlib to create insightful
charts that help analyze the outcomes of the Monte Carlo simulations.

Key functionalities include:
- Plotting the distribution of retirement durations for failed simulations.
- Visualizing the distribution of final wealth (nominal and real) for successful
  simulations, on a logarithmic scale for better clarity.
- Illustrating sampled wealth evolution paths (nominal and real) over the
  retirement period.
- Showing sample bank account balance trajectories (nominal and real) to track
  liquidity management.

All generated plots are saved as PNG files to a specified output directory
and are also displayed interactively for immediate review.
"""

import os
import matplotlib.pyplot as plt

# from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from typing import Any
from numpy.typing import NDArray

from firestarter.core.helpers import annual_to_monthly_compounded_rate
from firestarter.analysis.analysis import PlotLineData


plt.ion()  # Turn on interactive mode

# Set OUTPUT_DIR relative to the project root (or wherever you want)
OUTPUT_DIR = None  # Will be set from main.py


def set_output_dir(output_dir: str):
    """Set the output directory for all plots."""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir


def ensure_output_directory_exists() -> None:
    """Ensures the output directory exists."""
    if OUTPUT_DIR is None:
        raise RuntimeError("OUTPUT_DIR is not set. Call set_output_dir() before plotting.")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {OUTPUT_DIR}: {e}")


def _save_and_store_figure(fig: Any, filename: str) -> None:
    """Internal helper to save a figure to disk."""
    ensure_output_directory_exists()
    full_path = os.path.join(OUTPUT_DIR, filename)
    try:
        fig.savefig(full_path)
        print(f"Saved {os.path.abspath(full_path)}")
    except Exception as e:
        print(f"Error saving figure to {full_path}: {e}")


def plot_retirement_duration_distribution(
    failed_sims: pd.DataFrame,
    total_retirement_years: int,
    filename: str = "failed_sims_duration_distribution.png",
) -> None:
    """
    Plots a histogram of retirement duration for failed simulations.
    Saves to PNG and opens the figure interactively.
    """
    if failed_sims.empty:
        print("No failed simulations to plot retirement duration distribution.")
        return

    title: str = "Distribution of Retirement Duration for Failed Simulations"
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(10, 6))
    plt.hist(
        failed_sims["months_lasted"] / 12.0,
        bins=np.arange(0, total_retirement_years + 1, 1),
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel("Years Lasted")
    plt.ylabel("Number of Simulations")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_final_wealth_distribution_nominal(
    successful_sims: pd.DataFrame, filename: str = "nominal_final_wealth_distribution.png"
) -> None:
    """
    Plot the distribution of final nominal wealth for successful simulations.

    Only liquid assets are included in the plotted wealth.
    """
    if successful_sims.empty:
        print("No successful simulations to plot nominal final wealth distribution.")
        return

    title: str = (
        "Distribution of Total Wealth at End of Successful Simulations (Nominal - Log Scale)"
    )
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(12, 6))
    nominal_total_wealth: pd.Series = (
        successful_sims["final_investment"] + successful_sims["final_bank_balance"]
    )

    nominal_total_wealth_positive: pd.Series = nominal_total_wealth[nominal_total_wealth > 0.0]
    if nominal_total_wealth_positive.empty:
        print("\nWarning: No positive nominal final wealth to plot histogram on log scale.")
    else:
        min_nominal: float = max(1.0, float(nominal_total_wealth_positive.min()))
        max_nominal: float = float(nominal_total_wealth_positive.max())
        log_bins_nominal: NDArray[np.float64] = np.logspace(
            np.log10(min_nominal), np.log10(max_nominal), 50
        )

        plt.hist(nominal_total_wealth_positive, bins=log_bins_nominal, edgecolor="black")
        plt.xscale("log")
        plt.title(title)
        plt.xlabel("Total Wealth (EUR) - Log Scale")
        plt.ylabel("Number of Simulations")
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_final_wealth_distribution_real(
    successful_sims: pd.DataFrame, filename: str = "real_final_wealth_distribution.png"
) -> None:
    """
    Plots a histogram of real final wealth for successful simulations on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    if successful_sims.empty:
        print("No successful simulations to plot real final wealth distribution.")
        return

    title: str = (
        "Distribution of Total Wealth at End of Successful Simulations "
        + "(Real - Today's Money - Log Scale)"
    )
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(12, 6))
    real_final_wealth_positive: pd.Series = successful_sims["real_final_wealth"][
        successful_sims["real_final_wealth"] > 0.0
    ]
    if real_final_wealth_positive.empty:
        print("\nWarning: No positive real final wealth to plot histogram on log scale.")
    else:
        min_real: float = max(1.0, float(real_final_wealth_positive.min()))
        real_final_wealth_positive_max: float = float(real_final_wealth_positive.max())
        max_real: float = max(1.0, real_final_wealth_positive_max)

        log_bins_real: NDArray[np.float64] = np.logspace(np.log10(min_real), np.log10(max_real), 50)

        plt.hist(real_final_wealth_positive, bins=log_bins_real, edgecolor="black")
        plt.xscale("log")
        plt.title(title)
        plt.xlabel("Total Wealth (EUR in today's money) - Log Scale")
        plt.ylabel("Number of Simulations")
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_wealth_evolution_samples_real(
    results_df: pd.DataFrame,
    plot_lines_data: list[PlotLineData],
    pi_mu: float,
    filename: str = "wealth_evolution_real.png",
) -> None:
    """
    Plots sampled wealth evolution over retirement in real terms on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    title: str = "Sampled Wealth Evolution Over Retirement (Real Terms)"
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel("Years in Retirement")
    plt.ylabel("Total Wealth (EUR in today's money)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")

    for line_data in plot_lines_data:
        sim_idx: int = line_data["sim_idx"]
        label: str = line_data["label"]
        color: str = line_data["color"]
        linewidth: float = line_data["linewidth"]

        row: pd.Series = results_df.loc[sim_idx]
        nominal_history: NDArray[np.float64] = row["nominal_wealth_history"]

        real_history: NDArray[np.float64] = np.zeros_like(nominal_history, dtype=np.float64)

        current_cumulative_inflation_factor: float = 1.0

        for month_idx, nominal_balance in enumerate(nominal_history):
            year_idx: int = month_idx // 12
            monthly_inflation_rate: float
            if year_idx < len(row["annual_inflations_seq"]):
                monthly_inflation_rate = annual_to_monthly_compounded_rate(
                    row["annual_inflations_seq"][year_idx]
                )
            else:
                monthly_inflation_rate = annual_to_monthly_compounded_rate(pi_mu)

            current_cumulative_inflation_factor *= 1.0 + monthly_inflation_rate

            real_history[month_idx] = nominal_balance / current_cumulative_inflation_factor

        real_history_positive: NDArray[np.float64] = np.where(
            real_history <= 0.0, 1.0, real_history
        )

        adjusted_label: str = label
        if label != "_nolegend_":
            current_final_real_wealth: float = (
                float(real_history[-1]) if real_history.size > 0 else 0.0
            )

            if "Failed" in label:
                adjusted_label = label
            elif "Best Successful" in label:
                adjusted_label = f"Best Successful (Final Real: {current_final_real_wealth:,.0f}€)"
            elif "Worst Successful" in label:
                adjusted_label = f"Worst Successful (Final Real: {current_final_real_wealth:,.0f}€)"
            elif "Percentile Range" in label:
                adjusted_label = label.replace(
                    "Percentile Range",
                    f"Percentile Range (Final Real: {current_final_real_wealth:,.0f}€)",
                )
            else:
                adjusted_label = f"Sample (Final Real: {current_final_real_wealth:,.0f}€)"

        plt.plot(
            np.arange(0, len(nominal_history)) / 12.0,
            real_history_positive,
            label=adjusted_label,
            color=color,
            linewidth=linewidth,
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)


def plot_wealth_evolution_samples_nominal(
    results_df: pd.DataFrame,
    plot_lines_data: list[PlotLineData],
    filename: str = "wealth_evolution_nominal.png",
) -> None:
    """
    Plots sampled wealth evolution over retirement in nominal terms on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    title: str = "Sampled Wealth Evolution Over Retirement (Nominal Terms)"
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel("Years in Retirement")
    plt.ylabel("Total Wealth (EUR at time of value)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")

    for line_data in plot_lines_data:
        sim_idx: int = line_data["sim_idx"]
        label: str = line_data["label"]
        color: str = line_data["color"]
        linewidth: float = line_data["linewidth"]

        row: pd.Series = results_df.loc[sim_idx]
        nominal_history: NDArray[np.float64] = row["nominal_wealth_history"]

        # Adjust label to reflect nominal final wealth for the nominal plot, and handle _nolegend_
        adjusted_label: str = label
        if label != "_nolegend_":
            current_final_nominal_wealth: float = (
                float(nominal_history[-1]) if nominal_history.size > 0 else 0.0
            )

            if "Failed" in label:
                adjusted_label = label
            elif "Best Successful" in label:
                adjusted_label = (
                    f"Best Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
                )
            elif "Worst Successful" in label:
                adjusted_label = (
                    f"Worst Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
                )
            elif "Percentile Range" in label:
                adjusted_label = label.replace(
                    "Percentile Range",
                    f"Percentile Range (Final Nominal: {current_final_nominal_wealth:,.0f}€)",
                )
            else:
                adjusted_label = f"Sample (Final Nominal: {current_final_nominal_wealth:,.0f}€)"

        nominal_history_positive: NDArray[np.float64] = np.where(
            np.array(nominal_history) <= 0.0, 1.0, np.array(nominal_history)
        )

        plt.plot(
            np.arange(0, len(nominal_history)) / 12.0,
            nominal_history_positive,
            label=adjusted_label,
            color=color,
            linewidth=linewidth,
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)


def plot_bank_account_trajectories_real(
    results_df: pd.DataFrame,
    bank_account_plot_indices: NDArray[np.intp],
    real_bank_lower_bound: float,
    plot_lines_data: list[PlotLineData],
    filename: str = "bank_account_trajectories_real.png",
) -> None:
    """
    Plots sample bank account balance trajectories in real terms,
    using the same indices and colors as the wealth evolution samples.
    Saves to PNG and opens the figure interactively.
    """
    title: str = "Sampled Bank Account Balance Evolution (Real Terms)"
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel("Years in Retirement")
    plt.ylabel("Bank Account Balance (EUR in today's money)")
    plt.grid(True, linestyle="--", alpha=0.7)

    style_map = {d["sim_idx"]: d for d in plot_lines_data}

    for sim_idx in bank_account_plot_indices:
        row: pd.Series = results_df.loc[sim_idx]
        nominal_bank_history: NDArray[np.float64] = row["bank_balance_history"]
        cumulative_inflation_factors_monthly: NDArray[np.float64] = row[
            "cumulative_inflation_factors_monthly"
        ]

        # Ensure both arrays have the same length
        n = len(nominal_bank_history)
        real_bank_history: NDArray[np.float64] = (
            nominal_bank_history / cumulative_inflation_factors_monthly[:n]
        )
        plot_real_bank_history: NDArray[np.float64] = np.where(
            real_bank_history < 0.0, 0.0, real_bank_history
        )

        style = style_map.get(sim_idx, {})
        color = style.get("color", None)
        label = style.get("label", f"Sim {sim_idx}")
        linewidth = style.get("linewidth", 1.5)
        show_label = label != "_nolegend_"

        if label != "_nolegend_":
            final_real = (
                float(plot_real_bank_history[-1]) if len(plot_real_bank_history) > 0 else 0.0
            )
            if "Failed" in label:
                pass
            elif "Best Successful" in label:
                label = f"Best Successful (Final Real: {final_real:,.0f}€)"
            elif "Worst Successful" in label:
                label = f"Worst Successful (Final Real: {final_real:,.0f}€)"
            elif "Percentile Range" in label:
                label = label.replace(
                    "Percentile Range",
                    f"Percentile Range (Final Real: {final_real:,.0f}€)",
                )
            else:
                label = f"Sample (Final Real: {final_real:,.0f}€)"

        plt.plot(
            np.arange(0, len(nominal_bank_history)) / 12.0,
            plot_real_bank_history,
            alpha=0.7,
            linewidth=linewidth,
            color=color,
            label=label if show_label else None,
        )

    plt.axhline(
        y=real_bank_lower_bound,
        color="r",
        linestyle="--",
        label="Real Bank Lower Bound",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [
        (h, l)
        for i, (h, l) in enumerate(zip(handles, labels))
        if l and l != "_nolegend_" and l not in labels[:i]
    ]
    if unique:
        handles, labels = zip(*unique)
        plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)


def plot_bank_account_trajectories_nominal(
    results_df: pd.DataFrame,
    bank_account_plot_indices: NDArray[np.intp],
    plot_lines_data: list[PlotLineData],
    real_bank_lower_bound: float,
    filename: str = "bank_account_trajectories_nominal.png",
) -> None:
    """
    Plots sample bank account balance trajectories in nominal terms,
    using the same indices and colors as the wealth evolution samples.
    Saves to PNG and opens the figure interactively.
    """
    title: str = "Sampled Bank Account Balance Evolution (Nominal Terms)"
    print(f"Generating plot: {title}")
    fig: plt.Figure = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel("Years in Retirement")
    plt.ylabel("Bank Account Balance (EUR)")
    plt.grid(True, linestyle="--", alpha=0.7)

    style_map = {d["sim_idx"]: d for d in plot_lines_data}

    for sim_idx in bank_account_plot_indices:
        row: pd.Series = results_df.loc[sim_idx]
        nominal_bank_history: NDArray[np.float64] = row["bank_balance_history"]

        nominal_bank_history_np: NDArray[np.float64] = np.array(
            nominal_bank_history, dtype=np.float64
        )
        plot_nominal_bank_history: NDArray[np.float64] = np.where(
            nominal_bank_history_np < 0.0, 0.0, nominal_bank_history_np
        )

        style = style_map.get(sim_idx, {})
        label = style.get("label", f"Sim {sim_idx}")
        color = style.get("color", None)
        linewidth = style.get("linewidth", 1.5)
        show_label = label != "_nolegend_"

        if label != "_nolegend_":
            final_nominal = (
                float(plot_nominal_bank_history[-1]) if len(plot_nominal_bank_history) > 0 else 0.0
            )
            if "Failed" in label:
                pass
            elif "Best Successful" in label:
                label = f"Best Successful (Final Nominal: {final_nominal:,.0f}€)"
            elif "Worst Successful" in label:
                label = f"Worst Successful (Final Nominal: {final_nominal:,.0f}€)"
            elif "Percentile Range" in label:
                label = label.replace(
                    "Percentile Range",
                    f"Percentile Range (Final Nominal: {final_nominal:,.0f}€)",
                )
            else:
                label = f"Sample (Final Nominal: {final_nominal:,.0f}€)"

        plt.plot(
            np.arange(0, len(nominal_bank_history)) / 12.0,
            plot_nominal_bank_history,
            alpha=0.7,
            linewidth=linewidth,
            color=color,
            label=label if show_label else None,
        )

    plt.axhline(
        y=real_bank_lower_bound,
        color="r",
        linestyle="--",
        label="Real Bank Lower Bound (Nominal Reference)",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [
        (h, l)
        for i, (h, l) in enumerate(zip(handles, labels))
        if l and l != "_nolegend_" and l not in labels[:i]
    ]
    if unique:
        handles, labels = zip(*unique)
        plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)
