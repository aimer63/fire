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
from matplotlib.lines import Line2D
import numpy as np

from helpers import annual_to_monthly_compounded_rate

# --- Add this line at the very beginning of the script ---
plt.ion() # Turn on interactive mode

# Define the output directory
OUTPUT_DIR = 'output'

def ensure_output_directory_exists():
    """Ensures the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def _save_and_store_figure(fig, filename):
    """Internal helper to save a figure to disk."""
    ensure_output_directory_exists()
    full_path = os.path.join(OUTPUT_DIR, filename)

    fig.savefig(full_path)
    print(f"Saved {full_path}")

def plot_retirement_duration_distribution(
        failed_sims,
        t_ret_years,
        filename="failed_sims_duration_distribution.png"
    ):
    """
    Plots a histogram of retirement duration for failed simulations.
    Saves to PNG and opens the figure interactively.
    """
    if failed_sims.empty:
        print("No failed simulations to plot retirement duration distribution.")
        return

    title = 'Distribution of Retirement Duration for Failed Simulations'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(10, 6)) # REMOVED: visible=False
    plt.hist(
        failed_sims['months_lasted'] / 12,
        bins=np.arange(0, t_ret_years + 1, 1),
        edgecolor='black'
    )
    plt.title(title)
    plt.xlabel('Years Lasted')
    plt.ylabel('Number of Simulations')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_final_wealth_distribution_nominal(
        successful_sims,
        filename="nominal_final_wealth_distribution.png"
    ):
    """
    Plots a histogram of nominal final wealth for successful simulations on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    if successful_sims.empty:
        print("No successful simulations to plot nominal final wealth distribution.")
        return

    title = 'Distribution of Total Wealth at End of Successful Simulations (Nominal - Log Scale)'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(12, 6)) # REMOVED: visible=False
    nominal_total_wealth = \
        successful_sims['final_investment'] + successful_sims['final_bank_balance']

    nominal_total_wealth_positive = nominal_total_wealth[nominal_total_wealth > 0]
    if nominal_total_wealth_positive.empty:
        print("\nWarning: No positive nominal final wealth to plot histogram on log scale.")
    else:
        min_nominal = max(1.0, nominal_total_wealth_positive.min())
        max_nominal = nominal_total_wealth_positive.max()
        log_bins_nominal = np.logspace(np.log10(min_nominal), np.log10(max_nominal), 50)

        plt.hist(nominal_total_wealth_positive, bins=log_bins_nominal, edgecolor='black')
        plt.xscale('log')
        plt.title(title)
        plt.xlabel('Total Wealth (EUR) - Log Scale')
        plt.ylabel('Number of Simulations')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_final_wealth_distribution_real(
        successful_sims,
        filename="real_final_wealth_distribution.png"
    ):
    """
    Plots a histogram of real final wealth for successful simulations on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    if successful_sims.empty:
        print("No successful simulations to plot real final wealth distribution.")
        return

    title = (
        "Distribution of Total Wealth at End of Successful Simulations "
        "(Real - Today's Money - Log Scale)"
    )
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(12, 6)) # REMOVED: visible=False
    real_final_wealth_positive = \
        successful_sims['real_final_wealth'][successful_sims['real_final_wealth'] > 0]
    if real_final_wealth_positive.empty:
        print("\nWarning: No positive real final wealth to plot histogram on log scale.")
    else:
        min_real = max(1.0, real_final_wealth_positive.min())
        real_final_wealth_positive_max = real_final_wealth_positive.max()
        max_real = max(1.0, real_final_wealth_positive_max)

        log_bins_real = np.logspace(np.log10(min_real), np.log10(max_real), 50)

        plt.hist(real_final_wealth_positive, bins=log_bins_real, edgecolor='black')
        plt.xscale('log')
        plt.title(title)
        plt.xlabel("Total Wealth (EUR in today's money) - Log Scale")
        plt.ylabel('Number of Simulations')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

    _save_and_store_figure(fig, filename)


def plot_wealth_evolution_samples_real(
        results_df,
        plot_lines_data,
        mu_pi,
        filename="wealth_evolution_real.png"
    ):
    """
    Plots sampled wealth evolution over retirement in real terms on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    title = 'Sampled Wealth Evolution Over Retirement (Real Terms)'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel('Years in Retirement')
    plt.ylabel("Total Wealth (EUR in today's money)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')

    for line_data in plot_lines_data:
        sim_idx = line_data['sim_idx']
        label = line_data['label']
        color = line_data['color']
        linewidth = line_data['linewidth']

        row = results_df.loc[sim_idx]
        nominal_history = row['nominal_wealth_history']

        real_history = np.zeros_like(nominal_history, dtype=float)

        plot_cumulative_inflation_factor_monthly = 1.0

        for month_idx, nominal_balance in enumerate(nominal_history):
            year_idx = month_idx // 12
            if year_idx < len(row['annual_inflations_seq']):
                monthly_inflation_rate = annual_to_monthly_compounded_rate(
                    row['annual_inflations_seq'][year_idx]
                )
            else:
                # Fallback to average inflation
                monthly_inflation_rate = annual_to_monthly_compounded_rate(mu_pi)

            plot_cumulative_inflation_factor_monthly *= (1 + monthly_inflation_rate)

            real_history[month_idx] = nominal_balance / plot_cumulative_inflation_factor_monthly

         # Replace non-positive with 1 for log plot
        real_history_positive = np.where(real_history <= 0, 1, real_history)

        adjusted_label = label
        if label != '_nolegend_':
            current_final_real_wealth = real_history[-1] if real_history.size > 0 else 0

            if "Failed" in label:
                adjusted_label = label
            elif "Best Successful" in label:
                adjusted_label = f"Best Successful (Final Real: {current_final_real_wealth:,.0f}€)"
            elif "Worst Successful" in label:
                adjusted_label = f"Worst Successful (Final Real: {current_final_real_wealth:,.0f}€)"
            elif "Percentile Range" in label:
                adjusted_label = \
                    label.replace(
                        "Percentile Range",
                        f"Percentile Range (Final Real: {current_final_real_wealth:,.0f}€)"
                    )
            else:
                adjusted_label = f"Sample (Final Real: {current_final_real_wealth:,.0f}€)"

        plt.plot(
            np.arange(0, len(nominal_history)) / 12,
            real_history_positive,
            label=adjusted_label, # Now uses the adjusted label
            color=color,
            linewidth=linewidth
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)

def plot_wealth_evolution_samples_nominal(
        results_df,
        plot_lines_data,
        filename="wealth_evolution_nominal.png"
    ):
    """
    Plots sampled wealth evolution over retirement in nominal terms on a log scale.
    Saves to PNG and opens the figure interactively.
    """
    title = 'Sampled Wealth Evolution Over Retirement (Nominal Terms)'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(14, 8)) # REMOVED: visible=False
    plt.title(title)
    plt.xlabel('Years in Retirement')
    plt.ylabel("Total Wealth (EUR at time of value)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')

    for line_data in plot_lines_data:
        sim_idx = line_data['sim_idx']
        label = line_data['label']
        color = line_data['color']
        linewidth = line_data['linewidth']

        row = results_df.loc[sim_idx]
        nominal_history = row['nominal_wealth_history']

        # Adjust label to reflect nominal final wealth for the nominal plot, and handle _nolegend_
        adjusted_label = label
        if label != '_nolegend_':
            # current_final_nominal_wealth = nominal_history[-1] if nominal_history else 0
            current_final_nominal_wealth = nominal_history[-1] if nominal_history.size > 0 else 0

            if "Failed" in label:
                adjusted_label = label
            elif "Best Successful" in label:
                adjusted_label = \
                    f"Best Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
            elif "Worst Successful" in label:
                adjusted_label = \
                    f"Worst Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
            elif "Percentile Range" in label:
                adjusted_label = \
                    label.replace(
                        "Percentile Range",
                        f"Percentile Range (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
                    )
            else:
                adjusted_label = f"Sample (Final Nominal: {current_final_nominal_wealth:,.0f}€)"

        nominal_history_positive = \
            np.where(np.array(nominal_history) <= 0, 1, np.array(nominal_history))

        plt.plot(
            np.arange(0, len(nominal_history)) / 12,
            nominal_history_positive,
            label=adjusted_label,
            color=color,
            linewidth=linewidth
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)


def plot_bank_account_trajectories_real(
        results_df,
        bank_account_plot_indices,
        real_bank_lower_bound,
        filename="bank_account_trajectories_real.png"
    ):
    """
    Plots sample bank account balance trajectories in real terms.
    Saves to PNG and opens the figure interactively.
    """
    title = 'Sampled Bank Account Balance Evolution (Real Terms)'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(14, 8)) # REMOVED: visible=False
    plt.title(title)
    plt.xlabel('Years in Retirement')
    plt.ylabel("Bank Account Balance (EUR in today's money)")
    plt.grid(True, linestyle='--', alpha=0.7)

    for sim_idx in bank_account_plot_indices:
        row = results_df.loc[sim_idx]
        nominal_bank_history = row['bank_balance_history']
        annual_inflations_seq = row['annual_inflations_seq']

        real_bank_history = []

        for month_idx, nominal_balance in enumerate(nominal_bank_history):
            year_idx = month_idx // 12
            month_in_year_idx = month_idx % 12

            monthly_inflation_rate_this_year = \
                annual_to_monthly_compounded_rate(annual_inflations_seq[year_idx])

            # Calculate cumulative inflation factor up to the current month in the simulation year
            cumulative_inflation_factor_up_to_current_month = \
                (1 + monthly_inflation_rate_this_year)**(month_in_year_idx + 1)
            # If there are prior years, multiply by the cumulative inflation of those years
            for prev_year_rate in annual_inflations_seq[:year_idx]:
                cumulative_inflation_factor_up_to_current_month *= (1 + prev_year_rate)

            if cumulative_inflation_factor_up_to_current_month <= 0:
                cumulative_inflation_factor_up_to_current_month = 1.0

            real_balance = nominal_balance / cumulative_inflation_factor_up_to_current_month
            real_bank_history.append(real_balance)

        real_bank_history_np = np.array(real_bank_history)
        plot_real_bank_history = np.where(real_bank_history_np < 0, 0, real_bank_history_np)


        plt.plot(np.arange(0, len(nominal_bank_history)) / 12, plot_real_bank_history,
                 alpha=0.7, linewidth=1.5,
                 label=f"Sim {sim_idx} ({'Success' if row['success'] else 'Fail'})")

    plt.axhline(y=real_bank_lower_bound, color='r', linestyle='--', label='Real Bank Lower Bound')

    if len(bank_account_plot_indices) > 5:
        legend_elements = \
            [Line2D([0], [0], color='r', linestyle='--', label='Real Bank Lower Bound')]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)


def plot_bank_account_trajectories_nominal(
        results_df,
        bank_account_plot_indices,
        real_bank_lower_bound,
        filename="bank_account_trajectories_nominal.png"
    ):
    """
    Plots sample bank account balance trajectories in nominal terms.
    Saves to PNG and opens the figure interactively.
    """
    title = 'Sampled Bank Account Balance Evolution (Nominal Terms)'
    print(f"Generating plot: {title}")
    fig = plt.figure(figsize=(14, 8)) # REMOVED: visible=False
    plt.title(title)
    plt.xlabel('Years in Retirement')
    plt.ylabel("Bank Account Balance (EUR)")
    plt.grid(True, linestyle='--', alpha=0.7)

    for sim_idx in bank_account_plot_indices:
        row = results_df.loc[sim_idx]
        nominal_bank_history = row['bank_balance_history']

        nominal_bank_history_np = np.array(nominal_bank_history)
        plot_nominal_bank_history = \
            np.where(nominal_bank_history_np < 0, 0, nominal_bank_history_np)

        plt.plot(np.arange(0, len(nominal_bank_history)) / 12, plot_nominal_bank_history,
                 alpha=0.7, linewidth=1.5,
                 label=f"Sim {sim_idx} ({'Success' if row['success'] else 'Fail'})")

    # Add the horizontal line for the REAL_BANK_LOWER_BOUND_EUROS as a constant nominal reference
    plt.axhline(
        y=real_bank_lower_bound, color='r',
        linestyle='--',
        label='Real Bank Lower Bound (Nominal Reference)'
    )

    if len(bank_account_plot_indices) > 5:
        legend_elements = \
            [Line2D([0], [0], color='r', linestyle='--',
            label='Real Bank Lower Bound (Nominal Reference)')]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    _save_and_store_figure(fig, filename)
