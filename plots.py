# plots.py

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd # Needed for DataFrame operations if any, or just for type hinting
from matplotlib.lines import Line2D # For custom legend elements
from helpers import annual_to_monthly_compounded_rate # Import the helper function


def plot_retirement_duration_distribution(failed_sims, T_ret_years, filename="failed_sims_duration_distribution.png"):
    """
    Plots a histogram of retirement duration for failed simulations.
    """
    if failed_sims.empty:
        print("No failed simulations to plot retirement duration distribution.")
        return

    print(f"Generating plot: Distribution of Retirement Duration for Failed Simulations")
    plt.figure(figsize=(10, 6))
    plt.hist(failed_sims['months_lasted'] / 12, bins=np.arange(0, T_ret_years + 1, 1), edgecolor='black')
    plt.title('Distribution of Retirement Duration for Failed Simulations')
    plt.xlabel('Years Lasted')
    plt.ylabel('Number of Simulations')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_final_wealth_distribution_nominal(successful_sims, filename="nominal_final_wealth_distribution.png"):
    """
    Plots a histogram of nominal final wealth for successful simulations on a log scale.
    """
    if successful_sims.empty:
        print("No successful simulations to plot nominal final wealth distribution.")
        return

    print(f"Generating plot: Distribution of Total Wealth (Nominal)")
    plt.figure(figsize=(12, 6))
    nominal_total_wealth = successful_sims['final_investment'] + successful_sims['final_bank_balance']
    
    nominal_total_wealth_positive = nominal_total_wealth[nominal_total_wealth > 0]
    if nominal_total_wealth_positive.empty:
        print("\nWarning: No positive nominal final wealth to plot histogram on log scale.")
    else:
        min_nominal = max(1.0, nominal_total_wealth_positive.min())
        max_nominal = nominal_total_wealth_positive.max()
        log_bins_nominal = np.logspace(np.log10(min_nominal), np.log10(max_nominal), 50)
        
        plt.hist(nominal_total_wealth_positive, bins=log_bins_nominal, edgecolor='black')
        plt.xscale('log')
        plt.title('Distribution of Total Wealth at End of Successful Simulations (Nominal - Log Scale)')
        plt.xlabel('Total Wealth (EUR) - Log Scale')
        plt.ylabel('Number of Simulations')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

def plot_final_wealth_distribution_real(successful_sims, filename="real_final_wealth_distribution.png"):
    """
    Plots a histogram of real final wealth for successful simulations on a log scale.
    """
    if successful_sims.empty:
        print("No successful simulations to plot real final wealth distribution.")
        return

    print(f"Generating plot: Distribution of Total Wealth (Real)")
    plt.figure(figsize=(12, 6))
    real_final_wealth_positive = successful_sims['real_final_wealth'][successful_sims['real_final_wealth'] > 0]
    if real_final_wealth_positive.empty:
        print("\nWarning: No positive real final wealth to plot histogram on log scale.")
    else:
        min_real = max(1.0, real_final_wealth_positive.min())
        max_real = real_final_wealth_positive.max()
        log_bins_real = np.logspace(np.log10(min_real), np.log10(max_real), 50)
        
        plt.hist(real_final_wealth_positive, bins=log_bins_real, edgecolor='black')
        plt.xscale('log')
        plt.title("Distribution of Total Wealth at End of Successful Simulations (Real - Today's Money - Log Scale)")
        plt.xlabel("Total Wealth (EUR in today's money) - Log Scale")
        plt.ylabel('Number of Simulations')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        #plt.savefig(filename)
        plt.show()
        plt.close()
        print(f"Saved {filename}")

def plot_wealth_evolution_samples_real(results_df, plot_lines_data, mu_pi, filename="wealth_evolution_real.png"):
    """
    Plots sampled wealth evolution over retirement in real terms on a log scale.
    """
    print(f"Generating plot: Sampled Wealth Evolution (Real Terms)")
    plt.figure(figsize=(14, 8))
    plt.title('Sampled Wealth Evolution Over Retirement (Real Terms)')
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
        for month in range(len(nominal_history)):
            year_idx = month // 12
            if year_idx < len(row['annual_inflations_seq']):
                monthly_inflation_rate = annual_to_monthly_compounded_rate(row['annual_inflations_seq'][year_idx])
            else:
                monthly_inflation_rate = annual_to_monthly_compounded_rate(mu_pi) # Fallback to average inflation
                
            plot_cumulative_inflation_factor_monthly *= (1 + monthly_inflation_rate)
            
            real_history[month] = nominal_history[month] / plot_cumulative_inflation_factor_monthly
        
        real_history_positive = np.where(real_history <= 0, 1, real_history) # Replace non-positive with 1 for log plot
        
        plt.plot(
            np.arange(0, len(nominal_history)) / 12,
            real_history_positive,
            label=label,
            color=color,
            linewidth=linewidth
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_wealth_evolution_samples_nominal(results_df, plot_lines_data, filename="wealth_evolution_nominal.png"):
    """
    Plots sampled wealth evolution over retirement in nominal terms on a log scale.
    """
    print(f"Generating plot: Sampled Wealth Evolution (Nominal Terms)")
    plt.figure(figsize=(14, 8))
    plt.title('Sampled Wealth Evolution Over Retirement (Nominal Terms)')
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
            current_final_nominal_wealth = nominal_history[-1] if nominal_history else 0 
            
            if "Failed" in label:
                adjusted_label = label
            elif "Best Successful" in label:
                adjusted_label = f"Best Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
            elif "Worst Successful" in label:
                adjusted_label = f"Worst Successful (Final Nominal: {current_final_nominal_wealth:,.0f}€)"
            elif "Percentile Range" in label:
                adjusted_label = label.replace("Percentile Range", f"Percentile Range (Final Nominal: {current_final_nominal_wealth:,.0f}€)")
            else:
                adjusted_label = f"Sample (Final Nominal: {current_final_nominal_wealth:,.0f}€)"

        nominal_history_positive = np.where(np.array(nominal_history) <= 0, 1, np.array(nominal_history))

        plt.plot(
            np.arange(0, len(nominal_history)) / 12,
            nominal_history_positive,
            label=adjusted_label,
            color=color,
            linewidth=linewidth
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_bank_account_trajectories_real(results_df, bank_account_plot_indices, REAL_BANK_LOWER_BOUND_EUROS, filename="bank_account_trajectories_real.png"):
    """
    Plots sample bank account balance trajectories in real terms.
    """
    print(f"Generating plot: Sampled Bank Account Trajectories (Real Terms)")
    plt.figure(figsize=(14, 8))
    plt.title('Sampled Bank Account Balance Evolution (Real Terms)')
    plt.xlabel('Years in Retirement')
    plt.ylabel(f"Bank Account Balance (EUR in today's money)")
    plt.grid(True, linestyle='--', alpha=0.7)

    for sim_idx in bank_account_plot_indices:
        row = results_df.loc[sim_idx]
        nominal_bank_history = row['bank_balance_history']
        annual_inflations_seq = row['annual_inflations_seq']
        
        real_bank_history = []
        
        for month_idx, nominal_balance in enumerate(nominal_bank_history):
            year_idx = month_idx // 12
            month_in_year_idx = month_idx % 12
            
            monthly_inflation_rate_this_year = annual_to_monthly_compounded_rate(annual_inflations_seq[year_idx])
            
            if year_idx == 0:
                cumulative_inflation_factor_up_to_current_month = (1 + monthly_inflation_rate_this_year)**(month_in_year_idx + 1)
            else:
                cumulative_inflation_factor_up_to_current_month = np.prod(1 + annual_inflations_seq[:year_idx]) * \
                                                                   ((1 + monthly_inflation_rate_this_year)**(month_in_year_idx + 1))
            
            if cumulative_inflation_factor_up_to_current_month <= 0:
                 cumulative_inflation_factor_up_to_current_month = 1.0 

            real_balance = nominal_balance / cumulative_inflation_factor_up_to_current_month
            real_bank_history.append(real_balance)

        real_bank_history_np = np.array(real_bank_history)
        plot_real_bank_history = np.where(real_bank_history_np < 0, 0, real_bank_history_np) # Bank balance can't be negative


        plt.plot(np.arange(0, len(nominal_bank_history)) / 12, plot_real_bank_history, 
                 alpha=0.7, linewidth=1.5,
                 label=f"Sim {sim_idx} ({'Success' if row['success'] else 'Fail'})")

    plt.axhline(y=REAL_BANK_LOWER_BOUND_EUROS, color='r', linestyle='--', label='Real Bank Lower Bound')

    if len(bank_account_plot_indices) > 5:
        legend_elements = [Line2D([0], [0], color='r', linestyle='--', label='Real Bank Lower Bound')]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_bank_account_trajectories_nominal(results_df, bank_account_plot_indices, REAL_BANK_LOWER_BOUND_EUROS, filename="bank_account_trajectories_nominal.png"):
    """
    Plots sample bank account balance trajectories in nominal terms.
    """
    print(f"Generating plot: Sampled Bank Account Trajectories (Nominal Terms)")
    plt.figure(figsize=(14, 8))
    plt.title('Sampled Bank Account Balance Evolution (Nominal Terms)')
    plt.xlabel('Years in Retirement')
    plt.ylabel(f"Bank Account Balance (EUR)")
    plt.grid(True, linestyle='--', alpha=0.7)

    for sim_idx in bank_account_plot_indices:
        row = results_df.loc[sim_idx]
        nominal_bank_history = row['bank_balance_history']
        
        nominal_bank_history_np = np.array(nominal_bank_history)
        plot_nominal_bank_history = np.where(nominal_bank_history_np < 0, 0, nominal_bank_history_np)

        plt.plot(np.arange(0, len(nominal_bank_history)) / 12, plot_nominal_bank_history, 
                 alpha=0.7, linewidth=1.5,
                 label=f"Sim {sim_idx} ({'Success' if row['success'] else 'Fail'})")

    # The real bank lower bound is a constant nominal value only at the start.
    # Its nominal equivalent would change over time due to inflation.
    # Plotting the *initial* real lower bound as a nominal line might be misleading.
    # If a nominal lower bound is desired, it would need to be calculated based on inflation.
    # For now, keeping the real lower bound as a horizontal line here is technically wrong
    # if it's meant to be a *nominal* threshold, but if it's a visual reference to the *initial*
    # real bound, then it's fine.
    # Let's keep it commented out for now as it makes more sense for the real plot.
    # plt.axhline(y=REAL_BANK_LOWER_BOUND_EUROS, color='r', linestyle='--', label='Real Bank Lower Bound (Initial)')

    if len(bank_account_plot_indices) > 5:
        plt.legend(loc='upper right', fontsize='small')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_portfolio_allocations(
    initial_allocations_nominal, pre_rebalancing_allocations_nominal, rebalancing_allocations_nominal, final_allocations_nominal,
    initial_allocations_real, pre_rebalancing_allocations_real, rebalancing_allocations_real, final_allocations_real,
    scenario_type="Representative Simulation", filename_prefix="portfolio_allocations"
):
    """
    Plots a series of pie charts for nominal and real portfolio allocations at different stages
    for a specific scenario (e.g., worst, average, best).

    Args:
        initial_allocations_nominal (dict): Initial nominal allocations.
        pre_rebalancing_allocations_nominal (dict): Pre-rebalancing nominal allocations.
        rebalancing_allocations_nominal (dict): Post-rebalancing nominal allocations.
        final_allocations_nominal (dict): Final nominal allocations.
        initial_allocations_real (dict): Initial real allocations.
        pre_rebalancing_allocations_real (dict): Pre-rebalancing real allocations.
        rebalancing_allocations_real (dict): Post-rebalancing real allocations.
        final_allocations_real (dict): Final real allocations.
        scenario_type (str): Description of the scenario (e.g., "Worst Case").
        filename_prefix (str): Prefix for saved plot filenames.
    """
    # Helper to clean and plot a single pie chart
    def plot_single_pie(ax, data_dict, title):
        if not data_dict:
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return
        
        # Filter out assets with zero value to avoid empty slices
        labels = [k for k, v in data_dict.items() if v > 0]
        sizes = [v for v in data_dict.values() if v > 0]
        
        if not sizes: # Handle case where all filtered values are zero
            ax.text(0.5, 0.5, "No Assets > 0", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.1)
        ax.axis('equal')
        ax.set_title(title)

    # Plot Nominal Allocations
    print(f"Generating nominal portfolio allocation plots for {scenario_type}: {filename_prefix}_{scenario_type.lower().replace(' ', '_')}_nominal.png")
    fig_nom, axs_nom = plt.subplots(1, 4, figsize=(20, 6))
    fig_nom.suptitle(f"Nominal Portfolio Allocations - {scenario_type}", fontsize=16)

    plot_single_pie(axs_nom[0], initial_allocations_nominal, "Initial")
    plot_single_pie(axs_nom[1], pre_rebalancing_allocations_nominal, "Pre-Rebalancing")
    plot_single_pie(axs_nom[2], rebalancing_allocations_nominal, "Post-Rebalancing")
    plot_single_pie(axs_nom[3], final_allocations_nominal, "Final")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{filename_prefix}_{scenario_type.lower().replace(' ', '_')}_nominal.png")
    plt.close(fig_nom)
    print(f"Saved {filename_prefix}_{scenario_type.lower().replace(' ', '_')}_nominal.png")


    # Plot Real Allocations
    print(f"Generating real portfolio allocation plots for {scenario_type}: {filename_prefix}_{scenario_type.lower().replace(' ', '_')}_real.png")
    fig_real, axs_real = plt.subplots(1, 4, figsize=(20, 6))
    fig_real.suptitle(f"Real Portfolio Allocations - {scenario_type}", fontsize=16)

    plot_single_pie(axs_real[0], initial_allocations_real, "Initial")
    plot_single_pie(axs_real[1], pre_rebalancing_allocations_real, "Pre-Rebalancing")
    plot_single_pie(axs_real[2], rebalancing_allocations_real, "Post-Rebalancing")
    plot_single_pie(axs_real[3], final_allocations_real, "Final")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{filename_prefix}_{scenario_type.lower().replace(' ', '_')}_real.png")
    plt.close(fig_real)
    print(f"Saved {filename_prefix}_{scenario_type.lower().replace(' ', '_')}_real.png")


def plot_inflation_distribution(
    annual_inflations_sequences,
    plot_title="Annual Inflation Distribution (All Simulations)",
    filename="inflation_distribution.png"
):
    """
    Plots a histogram of annual inflation rates across all simulations.
    """
    print(f"Generating plot: {plot_title}")
    
    all_inflations = np.concatenate(annual_inflations_sequences) # Flatten all annual inflations into one array

    plt.figure(figsize=(10, 6))
    plt.hist(all_inflations, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Annual Inflation Rate")
    plt.ylabel("Density")
    plt.title(plot_title)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")