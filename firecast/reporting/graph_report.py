#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Provides functions for generating and saving plots that visualize the results of
FIRE Monte Carlo simulations.

Key functionalities:
- Plots distributions of durations for failed simulations.
- Plots distributions of final wealth (nominal and real) using histograms with log-scaled axes.
- Plots sample trajectories of wealth evolution for both successful and failed simulations,
  in both nominal and real terms.
- Highlights representative cases (worst, best, and percentile ranges) for deeper
  insight into simulation outcomes.
- Draws reference lines for bank account lower and upper bounds.
- Includes a utility function to generate all standard plots and save them to a specified
  output directory.

These visualizations are intended to support analysis, diagnostics, and communication
of FIRE simulation results by providing clear graphical summaries of key metrics and scenarios.
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from firecast.utils.colors import get_color

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")


def plot_retirement_duration_distribution(
    failed_sims: pd.DataFrame, total_retirement_years: int, filename: str
):
    if failed_sims.empty:
        print("No failed simulations to plot duration distribution.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(
        failed_sims["months_lasted"] / 12.0,
        bins=np.arange(0, total_retirement_years + 1, 1.0).tolist(),
        edgecolor="black",
        alpha=0.8,
        color=get_color("latte", "blue"),
    )
    plt.title("Distribution of duration for Failed Simulations")
    plt.xlabel("Years Lasted")
    plt.ylabel("Number of Simulations")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(filename)
    # Do not close the figure, keep it open for interactive display


def plot_final_wealth_distribution(
    successful_sims: pd.DataFrame, column: str, title: str, xlabel: str, filename: str
):
    data = successful_sims[column].clip(lower=1.0)
    if data.empty:
        print(f"No successful simulations to plot for {title}.")
        return

    plt.figure(figsize=(10, 6))  # Keep consistent figure size

    if (data == data.iloc[0]).all():
        center_val = data.iloc[0]

        # Define the x-axis viewing window for the single bar plot (e.g., +/- 2% around the value)
        # This determines how "zoomed in" we are.
        view_min_factor = 0.98
        view_max_factor = 1.02
        view_min = center_val * view_min_factor
        view_max = center_val * view_max_factor

        # Ensure view_min and view_max are valid for logspace and distinct
        if view_min <= 0:
            view_min = 1e-9  # Must be positive
        if (
            view_max <= view_min
        ):  # Handles center_val being very small or zero after clipping
            if center_val > 1e-9:  # if center_val is not effectively zero
                view_max = view_min * 1.01  # Ensure max > min by a small factor
            else:  # center_val is effectively zero, create a tiny range around it
                view_min = 0.9  # Arbitrary small range if value was ~1
                view_max = 1.1

        # Create 50 hypothetical log-spaced bins within this viewing window
        num_bins_for_width_calc = 50
        # N bins means N+1 edges
        hypothetical_bin_edges = np.logspace(
            np.log10(view_min), np.log10(view_max), num_bins_for_width_calc + 1
        )

        # Use the width of a middle hypothetical bin for our single bar
        # This makes its relative width consistent with the multi-bar plot's logic
        mid_idx = num_bins_for_width_calc // 2
        bar_width = (
            hypothetical_bin_edges[mid_idx + 1] - hypothetical_bin_edges[mid_idx]
        )

        plt.bar(
            [center_val],
            [len(data)],
            width=bar_width,
            color=get_color("latte", "blue"),
            alpha=0.8,
            edgecolor="black",
        )
        plt.xscale("log")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Simulations")
        plt.grid(axis="y", alpha=0.75)

        # Set x-limits to the defined viewing window
        plt.xlim(view_min, view_max)
    else:
        # Use 50 bins for the histogram (51 edges)
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 51)
        plt.hist(
            data,
            bins=bins.tolist(),
            alpha=0.8,
            color=get_color("latte", "blue"),
            edgecolor="black",
        )
        plt.xscale("log")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Simulations")
        plt.grid(axis="y", alpha=0.75)

    plt.tight_layout()
    plt.savefig(filename)
    # Do not close the figure, keep it open for interactive display


def plot_wealth_evolution_samples(results_df: pd.DataFrame, real: bool, filename: str):
    plt.figure(figsize=(14, 8))
    successful = results_df[results_df["success"]]
    if successful.empty:
        print(
            f"No successful simulations to plot {'real' if real else 'nominal'} wealth evolution."
        )
        return

    # Sort by final real/nominal wealth
    key = "final_real_wealth" if real else "final_nominal_wealth"
    sorted_successful = successful.sort_values(by=key).reset_index(drop=True)
    n = len(sorted_successful)

    # Percentile boundaries
    percentiles = [0, 20, 40, 60, 80, 100]
    colors = [
        get_color("latte", "peach"),
        get_color("latte", "yellow"),
        get_color("latte", "teal"),
        get_color("latte", "sky"),
        get_color("latte", "blue"),
    ]

    lw = 1.2

    # Plot 5 trajectories for each percentile range, but only one legend entry
    # per range (use upper value)
    for i in range(5):
        start = int(percentiles[i] / 100 * n)
        end = int(percentiles[i + 1] / 100 * n) if i < 4 else n
        count = end - start
        if count <= 0:
            continue
        # Evenly spaced 5 indices in this range
        if count < 5:
            indices = list(range(start, end))
        else:
            indices = np.linspace(start, end - 1, 5, dtype=int)
        for j, idx in enumerate(indices):
            row = sorted_successful.iloc[idx]
            wealth = np.array(row["wealth_history"], dtype=np.float64)
            if real:
                inflation = np.array(
                    row["monthly_cumulative_inflation_factors"], dtype=np.float64
                )
                wealth = wealth / inflation[: len(wealth)]
            if j == 0:
                if i == 4:  # 80-100th percentile, show the true max
                    max_final_val = 0.0
                    for k in range(start, end):
                        r = sorted_successful.iloc[k]
                        w = np.array(r["wealth_history"], dtype=np.float64)
                        if real:
                            infl = np.array(
                                r["monthly_cumulative_inflation_factors"],
                                dtype=np.float64,
                            )
                            w = w / infl[: len(w)]
                        if len(w) > 0 and w[-1] > max_final_val:
                            max_final_val = w[-1]
                    upper_final_val = max_final_val
                else:
                    upper_idx = end - 1 if end > start else start
                    upper_row = sorted_successful.iloc[upper_idx]
                    upper_wealth = np.array(
                        upper_row["wealth_history"], dtype=np.float64
                    )
                    if real:
                        inflation = np.array(
                            upper_row["monthly_cumulative_inflation_factors"],
                            dtype=np.float64,
                        )
                        upper_wealth = upper_wealth / inflation[: len(upper_wealth)]
                    upper_final_val = upper_wealth[-1] if len(upper_wealth) > 0 else 0.0
                label = (
                    f"{percentiles[i]}-{percentiles[i + 1]}th Percentile "
                    f"(Final: {upper_final_val:,.0f})"
                )
            else:
                label = None
            plt.plot(
                np.arange(0, len(wealth)) / 12.0,
                wealth,
                label=label,
                color=colors[i],
                linewidth=lw,
                alpha=0.8,
            )

    # Plot worst and best
    worst_row = sorted_successful.iloc[0]
    best_row = sorted_successful.iloc[-1]

    for row, label, color, width in [
        (
            worst_row,
            f"Worst Successful (Final {'Real' if real else 'Nominal'}: {worst_row[key]:,.0f})",
            get_color("latte", "red"),
            2.0,
        ),
        (
            best_row,
            f"Best Successful (Final {'Real' if real else 'Nominal'}: {best_row[key]:,.0f})",
            get_color("latte", "green"),
            2.0,
        ),
    ]:
        wealth = np.array(row["wealth_history"], dtype=np.float64)
        if real:
            inflation = np.array(
                row["monthly_cumulative_inflation_factors"], dtype=np.float64
            )
            wealth = wealth / inflation[: len(wealth)]
        plt.plot(
            np.arange(0, len(wealth)) / 12.0,
            wealth,
            label=label,
            color=color,
            linewidth=width,
            alpha=1.0,
        )

    plt.title(f"Sampled Wealth Evolution({'Real' if real else 'Nominal'} Terms)")
    plt.xlabel("Years")
    plt.ylabel(f"Total Wealth ({'real value' if real else 'nominal value'})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    # plt.tight_layout(rect=(0, 0, 0.85, 1))
    plt.tight_layout()
    plt.savefig(filename)
    # Do not close the figure, keep it open for interactive display


def plot_failed_wealth_evolution_samples(
    results_df: pd.DataFrame, real: bool, filename: str
):
    failed = results_df[~results_df["success"]]
    if failed.empty:
        print(
            f"No failed simulations to plot {'real' if real else 'nominal'} wealth evolution."
        )
        return
    plt.figure(figsize=(14, 8))

    # Take a sample of up to 25 simulations for clarity
    num_samples = 25
    sample_df = (
        failed.sample(n=num_samples, random_state=42)
        if len(failed) > num_samples
        else failed
    )

    # Sort sample by duration before fail
    sample_df = sample_df.sort_values("months_lasted")

    # Create a gradient colormap from two Catppuccin colors (shortest to longest duration)
    from matplotlib.colors import LinearSegmentedColormap

    gradient_colors = [
        get_color("latte", "red"),  # shortest duration
        get_color("latte", "mauve"),  # longest duration
    ]
    cmap = LinearSegmentedColormap.from_list("fail_gradient", gradient_colors)

    # Normalize duration for color mapping
    durations = sample_df["months_lasted"].to_numpy()
    min_dur, max_dur = durations.min(), durations.max()

    def norm(d):
        return (d - min_dur) / (max_dur - min_dur) if max_dur > min_dur else 0.5

    for _, row in sample_df.iterrows():
        wealth = np.array(row["wealth_history"], dtype=np.float64)
        if real:
            inflation = np.array(
                row["monthly_cumulative_inflation_factors"], dtype=np.float64
            )
            wealth = wealth / inflation[: len(wealth)]
        # Map duration to color in gradient
        color = cmap(norm(row["months_lasted"]))
        plt.plot(
            np.arange(0, len(wealth)) / 12.0,
            wealth,
            color=color,
            linewidth=1.0,
            alpha=0.7,
        )

    plt.title(
        f"Sampled Wealth Evolution for Failed Simulations ({'Real' if real else 'Nominal'} Terms)"
    )
    plt.xlabel("Years")
    plt.ylabel(f"Total Wealth ({'real value' if real else 'nominal value'})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filename)


def plot_bank_account_trajectories(
    results_df: pd.DataFrame,
    real: bool,
    bank_lower_bound: float,
    bank_upper_bound: float,
    filename: str,
):
    plt.figure(figsize=(14, 8))
    successful = results_df[results_df["success"]]
    if successful.empty:
        print(
            (
                f"No successful simulations to plot "
                f"{'real' if real else 'nominal'} bank account trajectories."
            )
        )
        return

    # Sort by final real/nominal wealth (same as wealth plot)
    key = "final_real_wealth" if real else "final_nominal_wealth"
    sorted_successful = successful.sort_values(by=key).reset_index(drop=True)
    n = len(sorted_successful)

    # Percentile boundaries and colors (same as wealth plot)
    percentiles = [0, 20, 40, 60, 80, 100]
    colors = [
        get_color("latte", "peach"),
        get_color("latte", "yellow"),
        get_color("latte", "teal"),
        get_color("latte", "sky"),
        get_color("latte", "blue"),
    ]
    lw = 1.2

    # Plot 5 trajectories for each percentile range, only first in legend
    for i in range(5):
        start = int(percentiles[i] / 100 * n)
        end = int(percentiles[i + 1] / 100 * n) if i < 4 else n
        count = end - start
        if count <= 0:
            continue
        if count < 5:
            indices = list(range(start, end))
        else:
            indices = np.linspace(start, end - 1, 5, dtype=int)
        for j, idx in enumerate(indices):
            row = sorted_successful.iloc[idx]
            bank = np.array(row["bank_balance_history"], dtype=np.float64)
            if real:
                inflation = np.array(
                    row["monthly_cumulative_inflation_factors"], dtype=np.float64
                )
                bank = bank / inflation[: len(bank)]
            final_val = bank[-1] if len(bank) > 0 else 0.0
            label = (
                f"{percentiles[i]}-{percentiles[i + 1]}th Percentile (Final: {final_val:,.0f})"
                if j == 0
                else None
            )
            plt.plot(
                np.arange(0, len(bank)) / 12.0,
                bank,
                label=label,
                color=colors[i],
                linewidth=lw,
                alpha=0.8,
            )

    # Plot worst and best
    worst_row = sorted_successful.iloc[0]
    best_row = sorted_successful.iloc[-1]
    for row, color, width, case in [
        (worst_row, get_color("latte", "red"), 2.0, "Worst Successful"),
        (best_row, get_color("latte", "green"), 2.0, "Best Successful"),
    ]:
        bank = np.array(row["bank_balance_history"], dtype=np.float64)
        if real:
            inflation = np.array(
                row["monthly_cumulative_inflation_factors"], dtype=np.float64
            )
            bank = bank / inflation[: len(bank)]
        final_val = bank[-1] if len(bank) > 0 else 0.0
        label = f"{case} (Final {'Real' if real else 'Nominal'}: {final_val:,.0f})"
        plt.plot(
            np.arange(0, len(bank)) / 12.0,
            bank,
            label=label,
            color=color,
            linewidth=width,
            alpha=1.0,
        )

    # Plot lower and upper bounds (real value in both plots)
    plt.axhline(
        y=bank_lower_bound,
        color=get_color("mocha", "yellow"),
        linestyle="--",
        linewidth=2.0,
        label=f"Bank Lower Bound ({bank_lower_bound:,.0f}, real value)",
    )
    plt.axhline(
        y=bank_upper_bound,
        color=get_color("latte", "mauve"),
        linestyle="--",
        linewidth=2.0,
        label=f"Bank Upper Bound ({bank_upper_bound:,.0f}, real value)",
    )

    plt.title(f"Bank Account Trajectories ({'Real' if real else 'Nominal'})")
    plt.xlabel("Years")
    plt.ylabel(f"Bank Account Balance ({'real value' if real else 'nominal value'})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    # plt.tight_layout(rect=(0, 0, 0.85, 1))
    plt.tight_layout()
    plt.savefig(filename)
    # Do not close the figure, keep it open for interactive display


def generate_all_plots(
    simulation_results: List[Dict[str, Any]],
    output_root: str,
    det_inputs: Any,
):
    """
    Generate all required plots for the FIRE simulator, using only simulation results.
    Plots are saved to the output directory.
    """
    plots_dir = os.path.join(output_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results_df = pd.DataFrame(simulation_results)
    failed_sims = results_df.loc[~results_df["success"]]
    successful_sims = results_df.loc[results_df["success"]]

    # 1. Retirement Duration Distribution (failed sims)
    plot_retirement_duration_distribution(
        failed_sims,
        det_inputs.years_to_simulate,
        os.path.join(plots_dir, "failed_duration_distribution.png"),
    )

    # 2. Final Wealth Distribution (Nominal)
    plot_final_wealth_distribution(
        successful_sims,
        "final_nominal_wealth",
        "Distribution of Final Wealth (Nominal)",
        "Total Wealth - Log Scale",
        os.path.join(plots_dir, "final_wealth_distribution_nominal.png"),
    )

    # 3. Final Wealth Distribution (Real)
    plot_final_wealth_distribution(
        successful_sims,
        "final_real_wealth",
        "Distribution of Final Wealth (Real)",
        "Total Wealth - Log Scale",
        os.path.join(plots_dir, "final_wealth_distribution_real.png"),
    )

    # 4. Wealth Evolution Samples (Real)
    plot_wealth_evolution_samples(
        results_df,
        real=True,
        filename=os.path.join(plots_dir, "wealth_evolution_samples_real.png"),
    )

    # 5. Wealth Evolution Samples (Nominal)
    plot_wealth_evolution_samples(
        results_df,
        real=False,
        filename=os.path.join(plots_dir, "wealth_evolution_samples_nominal.png"),
    )

    # 6. Failed Wealth Evolution Samples (Real)
    plot_failed_wealth_evolution_samples(
        results_df,
        real=True,
        filename=os.path.join(plots_dir, "failed_wealth_evolution_samples_real.png"),
    )

    # 7. Failed Wealth Evolution Samples (Nominal)
    plot_failed_wealth_evolution_samples(
        results_df,
        real=False,
        filename=os.path.join(plots_dir, "failed_wealth_evolution_samples_nominal.png"),
    )

    # 8. Bank Account Trajectories (Real)
    plot_bank_account_trajectories(
        results_df,
        real=True,
        bank_lower_bound=det_inputs.bank_lower_bound,
        bank_upper_bound=det_inputs.bank_upper_bound,
        filename=os.path.join(plots_dir, "bank_account_trajectories_real.png"),
    )

    # 9. Bank Account Trajectories (Nominal)
    plot_bank_account_trajectories(
        results_df,
        real=False,
        bank_lower_bound=det_inputs.bank_lower_bound,
        bank_upper_bound=det_inputs.bank_upper_bound,
        filename=os.path.join(plots_dir, "bank_account_trajectories_nominal.png"),
    )

    print(f"All plots generated and saved to {plots_dir}")
    # Show all open figures interactively, block until user closes them
    plt.show()
