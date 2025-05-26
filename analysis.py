import pandas as pd
import numpy as np
import random
from helpers import calculate_cagr, annual_to_monthly_compounded_rate, calculate_initial_asset_values

# Removed: print_allocations function (no longer needed)

def perform_analysis_and_prepare_plots_data(
    simulation_results, T_ret_years, I0,
    W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE,
    REBALANCING_YEAR_IDX, num_simulations, mu_pi
):
    """
    Performs post-simulation analysis, identifies key scenarios, and prepares data
    structures needed for comprehensive plotting. This function now focuses ONLY
    on preparing data for plotting and does NOT print detailed scenario-specific
    analyses (like allocations), which are moved to generate_fire_plan_summary.

    Args:
        simulation_results (list of tuples): Raw results from run_single_fire_simulation.
        T_ret_years (int): Total simulation duration in years.
        I0 (float): Initial investment.
        W_P1_STOCKS, W_P1_BONDS, ...: Phase 1 portfolio weights for initial allocation.
        REBALANCING_YEAR_IDX (int): Year index for rebalancing.
        num_simulations (int): Total number of simulations run.
        mu_pi (float): Average annual inflation rate (for plotting fallback).

    Returns:
        tuple: (results_df, plot_data_dict)
               results_df (pd.DataFrame): DataFrame of all simulation results with calculated metrics.
               plot_data_dict (dict): Dictionary containing data for all plots.
    """
    # Removed: print("\n--- Starting Post-Simulation Analysis (Detailed Breakdowns) ---")

    results_df = pd.DataFrame(simulation_results, columns=[
        'success', 'months_lasted', 'final_investment', 'final_bank_balance',
        'annual_inflations_seq', 'nominal_wealth_history', 'bank_balance_history',
        'pre_rebalancing_allocations_nominal', 'pre_rebalancing_allocations_real',
        'rebalancing_allocations_nominal', 'rebalancing_allocations_real',
        'final_allocations_nominal', 'final_allocations_real'
    ])

    real_final_wealths_all_sims = []
    for idx, row in results_df.iterrows():
        if row['success']:
            cumulative_inflation_factor = np.prod(1 + np.array(row['annual_inflations_seq'])) if len(row['annual_inflations_seq']) > 0 else 1.0
            real_wealth = (row['final_investment'] + row['final_bank_balance']) / cumulative_inflation_factor
        else:
            real_wealth = 0
        real_final_wealths_all_sims.append(real_wealth)

    results_df['real_final_wealth'] = real_final_wealths_all_sims

    # Removed: All previous logic for identifying and printing Worst, Average, Best Case allocation snapshots
    # This entire block (lines 40-159 in original analysis.py) is now removed from here.

    # --- Prepare data for Time Evolution Samples (Wealth) ---
    print("\n--- Preparing Data for Time Evolution Samples (Wealth & Bank Account) ---")
    plot_lines_data = []

    successful_sims_for_plotting = results_df[results_df['success']]
    all_sims_sorted_by_real_wealth = results_df.sort_values(by='real_final_wealth', ascending=True)

    if not all_sims_sorted_by_real_wealth.empty:
        worst_sim_row_for_plot = all_sims_sorted_by_real_wealth.iloc[0]
        worst_sim_idx_for_plot = worst_sim_row_for_plot.name
        if worst_sim_row_for_plot['success'] == False:
            plot_lines_data.append({
                'sim_idx': worst_sim_idx_for_plot,
                'label': f"Worst Case (Failed Year {worst_sim_row_for_plot['months_lasted']/12:.1f})",
                'color': 'red',
                'linewidth': 2.5
            })
        else:
            plot_lines_data.append({
                'sim_idx': worst_sim_idx_for_plot,
                'label': f"Worst Successful (Final Real: {worst_sim_row_for_plot['real_final_wealth']:,.0f}€)",
                'color': 'darkred',
                'linewidth': 2.0
            })

    if len(successful_sims_for_plotting) > 0:
        successful_sims_sorted_for_plotting = successful_sims_for_plotting.sort_values(by='real_final_wealth', ascending=True)
        
        percentile_bins = [0, 20, 40, 60, 80, 100]
        num_samples_per_bin = 5
        
        for i in range(len(percentile_bins) - 1):
            lower_percentile = percentile_bins[i]
            upper_percentile = percentile_bins[i+1]
            
            start_idx_in_sorted = int(np.percentile(np.arange(len(successful_sims_sorted_for_plotting)), lower_percentile))
            if upper_percentile == 100:
                end_idx_in_sorted = len(successful_sims_sorted_for_plotting)
            else:
                end_idx_in_sorted = int(np.percentile(np.arange(len(successful_sims_sorted_for_plotting)), upper_percentile))
            
            range_indices = successful_sims_sorted_for_plotting.iloc[start_idx_in_sorted:end_idx_in_sorted].index.tolist()
            
            existing_indices = [data['sim_idx'] for data in plot_lines_data]
            range_indices = [idx for idx in range_indices if idx not in existing_indices]

            if len(range_indices) > 0:
                sampled_indices = np.random.choice(range_indices, 
                                                   min(len(range_indices), num_samples_per_bin), 
                                                   replace=False).tolist()
                
                if upper_percentile <= 20: current_color = 'darkorange'
                elif upper_percentile <= 40: current_color = 'gold'
                elif upper_percentile <= 60: current_color = 'forestgreen'
                elif upper_percentile <= 80: current_color = 'dodgerblue'
                else: current_color = 'mediumblue'

                for j, sim_idx in enumerate(sampled_indices):
                    label_to_use = f"{lower_percentile}-{upper_percentile}th Percentile Range" if j == 0 else '_nolegend_'
                    plot_lines_data.append({
                        'sim_idx': sim_idx,
                        'label': label_to_use,
                        'color': current_color,
                        'linewidth': 1.0
                    })

    if not successful_sims_for_plotting.empty:
        best_sim_row_for_plot = successful_sims_sorted_for_plotting.iloc[-1]
        best_sim_idx_for_plot = best_sim_row_for_plot.name
        existing_indices = [data['sim_idx'] for data in plot_lines_data]
        if best_sim_idx_for_plot not in existing_indices:
            plot_lines_data.append({
                'sim_idx': best_sim_idx_for_plot,
                'label': f"Best Successful (Final Real: {best_sim_row_for_plot['real_final_wealth']:,.0f}€)",
                'color': 'green',
                'linewidth': 2.5
            })

    # --- Prepare data for Bank Account Trajectories ---
    num_trajectories_to_plot = 20
    if num_simulations < num_trajectories_to_plot:
        print(f"Warning: Only {num_simulations} simulations available, plotting all bank account trajectories.")
        bank_account_plot_indices = results_df.index.tolist()
    else:
        if len(successful_sims_for_plotting) >= num_trajectories_to_plot:
            bank_account_plot_indices = np.random.choice(successful_sims_for_plotting.index, num_trajectories_to_plot, replace=False)
        else:
            failed_indices = results_df[~results_df['success']].index.tolist()
            random.shuffle(failed_indices)
            bank_account_plot_indices = successful_sims_for_plotting.index.tolist() + failed_indices[:(num_trajectories_to_plot - len(successful_sims_for_plotting))]
            random.shuffle(bank_account_plot_indices)

    plot_data_dict = {
        'results_df': results_df,
        'successful_sims': successful_sims_for_plotting,
        'failed_sims': results_df[~results_df['success']],
        'plot_lines_data': plot_lines_data,
        'bank_account_plot_indices': bank_account_plot_indices
    }

    return results_df, plot_data_dict


def generate_fire_plan_summary(simulation_results, initial_total_wealth_nominal, T_ret_years):
    """
    Analyzes simulation results and generates a single, formatted summary string
    containing success rate, average wealth, CAGR, and best/worst/average cases,
    including final asset allocations as percentages.

    Args:
        simulation_results (list): List of tuples, each containing results from a single simulation.
        initial_total_wealth_nominal (float): The starting total nominal wealth for CAGR calculation.
        T_ret_years (int): The total number of retirement years simulated.

    Returns:
        str: A multi-line string containing the formatted simulation summary.
    """
    successful_simulations_count = 0
    failed_simulations_count = 0
    months_lasted_in_failed_simulations = []
    
    # Store the actual result tuples for successful simulations to retrieve allocation data later
    successful_results_data = []

    # Iterate through all simulation results to collect data for the summary
    for result in simulation_results:
        # Unpack the relevant parts of the result tuple.
        # Ensure this matches the order of elements returned by run_single_fire_simulation
        # (success, months_lasted, final_investment, final_bank_balance, annual_inflations_seq, ...)
        success, months_lasted, final_investment, final_bank_balance, annual_inflations_seq, \
        nominal_wealth_history, bank_balance_history, \
        pre_rebalancing_allocations_nominal, pre_rebalancing_allocations_real, \
        rebalancing_allocations_nominal, rebalancing_allocations_real, \
        final_allocations_nominal, final_allocations_real = result

        if success:
            successful_simulations_count += 1
            final_total_wealth_nominal = final_investment + final_bank_balance
            # Note: We don't append to successful_final_wealth_nominal/real/cagrs directly here anymore
            # as we'll derive them from the specific result tuples identified later.
            
            # Store the full result tuple for successful simulations
            successful_results_data.append(result)

        else:
            failed_simulations_count += 1
            months_lasted_in_failed_simulations.append(months_lasted)

    # --- Summary Calculations ---
    total_simulations = len(simulation_results)
    fire_success_rate = (successful_simulations_count / total_simulations) * 100 if total_simulations > 0 else 0

    avg_months_failed = np.mean(months_lasted_in_failed_simulations) if failed_simulations_count > 0 else 0

    # Initialize variables to hold the *full result tuples* for key scenarios
    worst_successful_result = None
    average_successful_result = None
    best_successful_result = None

    if successful_simulations_count > 0:
        # Create a temporary list of (real_final_wealth, result_tuple) from successful_results_data
        temp_successful_sorted = []
        for res in successful_results_data:
            # Re-unpack only what's needed for sorting
            _, _, final_inv, final_bank, annual_infl, *_ = res
            final_nom_wealth = final_inv + final_bank
            cum_infl_factor = np.prod(1 + np.array(annual_infl)) if T_ret_years > 0 else 1.0
            real_wealth = final_nom_wealth / cum_infl_factor
            temp_successful_sorted.append((real_wealth, res))
        
        temp_successful_sorted.sort(key=lambda x: x[0]) # Sort by real wealth

        worst_successful_result = temp_successful_sorted[0][1] # Smallest real wealth
        best_successful_result = temp_successful_sorted[-1][1] # Largest real wealth

        # Find Average Case (closest to median real_final_wealth among successful sims)
        median_real_wealth_among_successful = np.median([x[0] for x in temp_successful_sorted])
        
        closest_to_median = min(temp_successful_sorted, key=lambda x: np.abs(x[0] - median_real_wealth_among_successful))
        average_successful_result = closest_to_median[1]


    # Helper function to format allocations as percentages
    def _format_allocations_as_percentages(allocations_nominal_dict):
        if not allocations_nominal_dict:
            return "N/A"
        
        total_nominal = sum(allocations_nominal_dict.values())
        if total_nominal == 0:
            return "All zero"
            
        percentage_strings = []
        for asset, nom_val in allocations_nominal_dict.items():
            percentage = (nom_val / total_nominal) * 100
            # Only include if percentage is significant or if it's explicitly zero (not just tiny float)
            if percentage >= 0.1 or (percentage == 0 and nom_val == 0):
                 percentage_strings.append(f"{asset}: {percentage:.1f}%")
            elif percentage > 0: # Handle very small non-zero percentages
                 percentage_strings.append(f"{asset}: <0.1%")
        return ", ".join(percentage_strings)

    # --- Construct the summary string ---
    summary_lines = [
        "\n--- FIRE Plan Simulation Summary ---",
        f"FIRE Plan Success Rate: {fire_success_rate:.2f}%",
        f"Number of failed simulations: {failed_simulations_count}",
    ]
    if failed_simulations_count > 0:
        summary_lines.append(f"Average months lasted in failed simulations: {avg_months_failed:.1f}")

    if successful_simulations_count > 0:
        summary_lines.append("\n--- Successful Cases Details ---")

        # Worst Successful Case
        if worst_successful_result:
            # Unpack specific result tuple for its final values and allocations
            _, _, final_inv, final_bank, annual_infl, *_, _, _, _, _, _, final_allocs_nom = worst_successful_result
            final_total_wealth_nominal = final_inv + final_bank
            cum_infl_factor = np.prod(1 + np.array(annual_infl)) if T_ret_years > 0 else 1.0
            final_total_wealth_real = final_total_wealth_nominal / cum_infl_factor
            cagr = calculate_cagr(initial_total_wealth_nominal, final_total_wealth_nominal, T_ret_years)

            summary_lines.append(f"Worst Successful Case:")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(f"  Final Allocations: {_format_allocations_as_percentages(final_allocs_nom)}")

        # Average Successful Case
        if average_successful_result:
            _, _, final_inv, final_bank, annual_infl, *_, _, _, _, _, _, final_allocs_nom = average_successful_result
            final_total_wealth_nominal = final_inv + final_bank
            cum_infl_factor = np.prod(1 + np.array(annual_infl)) if T_ret_years > 0 else 1.0
            final_total_wealth_real = final_total_wealth_nominal / cum_infl_factor
            cagr = calculate_cagr(initial_total_wealth_nominal, final_total_wealth_nominal, T_ret_years)

            summary_lines.append(f"\nAverage Successful Case:")
            summary_lines.append(f"  (Simulation closest to median real final wealth)")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(f"  Final Allocations: {_format_allocations_as_percentages(final_allocs_nom)}")

        # Best Successful Case
        if best_successful_result:
            _, _, final_inv, final_bank, annual_infl, *_, _, _, _, _, _, final_allocs_nom = best_successful_result
            final_total_wealth_nominal = final_inv + final_bank
            cum_infl_factor = np.prod(1 + np.array(annual_infl)) if T_ret_years > 0 else 1.0
            final_total_wealth_real = final_total_wealth_nominal / cum_infl_factor
            cagr = calculate_cagr(initial_total_wealth_nominal, final_total_wealth_nominal, T_ret_years)

            summary_lines.append(f"\nBest Successful Case:")
            summary_lines.append(f"  Final Wealth (Nominal): {final_total_wealth_nominal:,.2f} EUR")
            summary_lines.append(f"  Final Wealth (Real): {final_total_wealth_real:,.2f} EUR")
            summary_lines.append(f"  Your life CAGR: {cagr:.2%}")
            summary_lines.append(f"  Final Allocations: {_format_allocations_as_percentages(final_allocs_nom)}")

    else:
        summary_lines.append("\nNo successful simulations to report details.")

    return "\n".join(summary_lines)