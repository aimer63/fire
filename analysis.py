# analysis.py

import pandas as pd
import numpy as np
import random # For random sampling for plotting
from helpers import annual_to_monthly_compounded_rate, calculate_initial_asset_values # Assuming these are in helpers

def print_allocations(title, allocations_nominal, allocations_real):
    """Helper function to print allocation details."""
    print(f"\n--- {title} ---")
    if not allocations_nominal: # Check nominal, as real depends on it
        print("  Allocation data not available.")
        return

    # Assuming allocations_nominal and allocations_real are dictionaries with asset names as keys
    # And asset names are consistent between nominal and real
    for asset in allocations_nominal.keys():
        nom_val = allocations_nominal.get(asset, 0)
        real_val = allocations_real.get(asset, 0)
        print(f"  {asset}: Nominal={nom_val:,.2f}€, Real={real_val:,.2f}€")


def perform_analysis_and_prepare_plots_data(
    simulation_results, T_ret_years, I0,
    W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE, # For initial allocation calc
    REBALANCING_YEAR_IDX, num_simulations, mu_pi # mu_pi for fallback inflation in plotting
):
    """
    Performs post-simulation analysis, identifies key scenarios, and prepares data
    structures needed for comprehensive plotting.

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
    print("\n--- Starting Post-Simulation Analysis ---")

    # Updated columns to include new allocation data AND bank_balance_history
    # Ensure this matches the exact return order of run_single_fire_simulation
    # (success, months_lasted, final_investment, final_bank_balance, annual_inflations_seq,
    #  nominal_wealth_history, bank_balance_history,
    #  pre_rebalancing_allocations_nominal, pre_rebalancing_allocations_real,
    #  rebalancing_allocations_nominal, rebalancing_allocations_real,
    #  final_allocations_nominal, final_allocations_real)
    results_df = pd.DataFrame(simulation_results, columns=[
        'success', 'months_lasted', 'final_investment', 'final_bank_balance',
        'annual_inflations_seq', 'nominal_wealth_history', 'bank_balance_history',
        'pre_rebalancing_allocations_nominal', 'pre_rebalancing_allocations_real',
        'rebalancing_allocations_nominal', 'rebalancing_allocations_real',
        'final_allocations_nominal', 'final_allocations_real'
    ])

    # --- Calculate real_final_wealth for ALL simulations and add to results_df ---
    real_final_wealths_all_sims = []
    for idx, row in results_df.iterrows():
        if row['success']:
            # Only calculate for successful paths
            # Ensure annual_inflations_seq is not empty for np.prod
            cumulative_inflation_factor = np.prod(1 + row['annual_inflations_seq']) if len(row['annual_inflations_seq']) > 0 else 1.0
            real_wealth = (row['final_investment'] + row['final_bank_balance']) / cumulative_inflation_factor
        else:
            # For failed simulations, set to 0 or a very low value to place them at the bottom in sorting
            real_wealth = 0
        real_final_wealths_all_sims.append(real_wealth)

    results_df['real_final_wealth'] = real_final_wealths_all_sims


    success_rate = results_df['success'].mean() * 100
    print(f"\nFIRE Plan Success Rate: {success_rate:.2f}%")

    failed_sims = results_df[~results_df['success']]
    if not failed_sims.empty:
        print(f"\nNumber of failed simulations: {len(failed_sims)}")
        print(f"Average months lasted in failed simulations: {failed_sims['months_lasted'].mean():.1f}")
        
    # Re-filter successful_sims AFTER 'real_final_wealth' has been added to results_df
    successful_sims = results_df[results_df['success']]

    if not successful_sims.empty:
        print(f"\nNumber of successful simulations: {len(successful_sims)}")
        
        # Calculate nominal total wealth
        nominal_total_wealth = successful_sims['final_investment'] + successful_sims['final_bank_balance']
        print(f"Average total wealth at end of successful simulations (Nominal): {nominal_total_wealth.mean():,.2f} EUR")

        average_real_final_wealth = successful_sims['real_final_wealth'].mean()
        print(f"Average total wealth at end of successful simulations (Real - Today's Money): {average_real_final_wealth:,.2f} EUR")


    # --- Print Portfolio Allocation Snapshots for Worst, Average, Best Cases ---

    print("\n--- Portfolio Allocation Snapshots ---")

    # Identify Worst, Average, Best Cases
    # Sort ALL simulations by their real final wealth
    all_sims_sorted = results_df.sort_values(by='real_final_wealth', ascending=True)

    # Worst case (first in all_sims_sorted)
    worst_sim_idx = all_sims_sorted.iloc[0].name

    # Best case (last in successful_sims_sorted, if any successful sims exist)
    best_sim_idx = None
    if not successful_sims.empty:
        successful_sims_sorted = successful_sims.sort_values(by='real_final_wealth', ascending=True)
        best_sim_idx = successful_sims_sorted.iloc[-1].name

    # Find Average Case (closest to median real_final_wealth among successful sims)
    average_sim_idx = None
    if not successful_sims.empty:
        median_real_wealth = successful_sims['real_final_wealth'].median()
        diffs = np.abs(successful_sims['real_final_wealth'] - median_real_wealth)
        average_sim_idx = diffs.idxmin() # Get index of row with minimum difference

    # Helper function to print a scenario's allocations
    def print_scenario_allocations(sim_id, scenario_type, row, REBALANCING_YEAR_IDX, T_ret_years, I0, W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE):
        # Calculate nominal total wealth for this specific row
        nominal_total_wealth = row['final_investment'] + row['final_bank_balance']
        
        scenario_str = f"Scenario: {scenario_type} (Sim ID: {sim_id}, Nominal Final Wealth: {nominal_total_wealth:,.0f}€, Real Final Wealth: {row['real_final_wealth']:,.0f}€)"
        print(f"\n{scenario_str}")
        print(f"  Outcome: {'SUCCESS' if row['success'] else f'FAILURE (lasted {row["months_lasted"]/12:.1f} years)'}")

        # Initial Allocations (calculated dynamically from initial parameters)
        initial_allocs_tuple = calculate_initial_asset_values(
            I0, W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE
        )
        initial_allocations_nominal_dict = {
            'Stocks': initial_allocs_tuple[0], 'Bonds': initial_allocs_tuple[1], 'STR': initial_allocs_tuple[2],
            'Fun': initial_allocs_tuple[3], 'Real Estate': initial_allocs_tuple[4]
        }
        # Assuming I0 is real initially, so nominal and real initial are the same at month 0.
        initial_allocations_real_dict = initial_allocations_nominal_dict.copy()
        print_allocations("Initial Allocations", initial_allocations_nominal_dict, initial_allocations_real_dict)


        # Pre-Rebalancing Allocations
        if REBALANCING_YEAR_IDX > 0 and REBALANCING_YEAR_IDX < T_ret_years and row['pre_rebalancing_allocations_nominal']:
            print_allocations(
                f"Allocations Just BEFORE Rebalancing (Year {REBALANCING_YEAR_IDX+1})",
                row['pre_rebalancing_allocations_nominal'],
                row['pre_rebalancing_allocations_real']
            )
        else:
            print(f"\n--- Allocations Just BEFORE Rebalancing (Year {REBALANCING_YEAR_IDX+1}) ---")
            print("  Pre-rebalancing data not available or rebalancing year out of scope for this simulation path.")

        # Rebalancing Allocations (After Rebalancing)
        if REBALANCING_YEAR_IDX > 0 and REBALANCING_YEAR_IDX < T_ret_years and row['rebalancing_allocations_nominal']:
            print_allocations(
                f"Allocations Just AFTER Rebalancing (Year {REBALANCING_YEAR_IDX+1})",
                row['rebalancing_allocations_nominal'],
                row['rebalancing_allocations_real']
            )
        else:
            print(f"\n--- Allocations Just AFTER Rebalancing (Year {REBALANCING_YEAR_IDX+1}) ---")
            print("  Post-rebalancing data not available or rebalancing year out of scope for this simulation path.")

        # Final Allocations
        print_allocations(
            "Allocations at End of Simulation",
            row['final_allocations_nominal'],
            row['final_allocations_real']
        )


    if worst_sim_idx is not None:
        print_scenario_allocations(worst_sim_idx, "Worst Case", results_df.loc[worst_sim_idx], REBALANCING_YEAR_IDX, T_ret_years, I0, W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE)
    
    if average_sim_idx is not None:
        print_scenario_allocations(average_sim_idx, "Average Case", results_df.loc[average_sim_idx], REBALANCING_YEAR_IDX, T_ret_years, I0, W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE)

    if best_sim_idx is not None:
        print_scenario_allocations(best_sim_idx, "Best Case", results_df.loc[best_sim_idx], REBALANCING_YEAR_IDX, T_ret_years, I0, W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE)

    # --- Prepare data for Time Evolution Samples (Wealth) ---
    print("\n--- Preparing Data for Time Evolution Samples (Wealth & Bank Account) ---")
    plot_lines_data = []

    # 1. Add the very worst path (earliest failure if any, or lowest final wealth if all succeed)
    if not all_sims_sorted.empty:
        worst_sim_row_for_plot = all_sims_sorted.iloc[0]
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

    # 2. Add samples for percentile ranges for successful simulations
    if len(successful_sims) > 0:
        successful_sims_sorted = successful_sims.sort_values(by='real_final_wealth', ascending=True)
        
        percentile_bins = [0, 20, 40, 60, 80, 100]
        num_samples_per_bin = 5
        
        for i in range(len(percentile_bins) - 1):
            lower_percentile = percentile_bins[i]
            upper_percentile = percentile_bins[i+1]
            
            start_idx_in_sorted = int(np.percentile(np.arange(len(successful_sims_sorted)), lower_percentile))
            if upper_percentile == 100:
                end_idx_in_sorted = len(successful_sims_sorted)
            else:
                end_idx_in_sorted = int(np.percentile(np.arange(len(successful_sims_sorted)), upper_percentile))
            
            range_indices = successful_sims_sorted.iloc[start_idx_in_sorted:end_idx_in_sorted].index.tolist()
            
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

    # Ensure the best successful path is always included and clearly labeled
    if not successful_sims.empty:
        best_sim_row_for_plot = successful_sims_sorted.iloc[-1]
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
        if len(successful_sims) >= num_trajectories_to_plot:
            bank_account_plot_indices = np.random.choice(successful_sims.index, num_trajectories_to_plot, replace=False)
        else:
            failed_indices = failed_sims.index.tolist()
            random.shuffle(failed_indices)
            bank_account_plot_indices = successful_sims.index.tolist() + failed_indices[:(num_trajectories_to_plot - len(successful_sims))]
            random.shuffle(bank_account_plot_indices) # Shuffle the combined list to mix randomly

    plot_data_dict = {
        'results_df': results_df,
        'successful_sims': successful_sims,
        'failed_sims': failed_sims,
        'plot_lines_data': plot_lines_data, # For wealth evolution plots
        'bank_account_plot_indices': bank_account_plot_indices # For bank account plots
    }

    return results_df, plot_data_dict


def calculate_and_display_cagr(simulation_results, I0, b0, T_ret_years):
    """
    Calculates and displays the Compound Annual Growth Rate (CAGR) for
    the worst, average (median), and best performing successful simulations,
    for both Nominal and Real terms.

    Args:
        simulation_results (list): List of results from each simulation run.
        I0 (float): Initial total investment (real terms, which is nominal at Year 0).
        b0 (float): Initial bank account balance (real terms, which is nominal at Year 0).
        T_ret_years (int): Total simulation duration in years.
    """
    # Beginning Wealth (Nominal and Real are the same at Year 0)
    beginning_wealth = I0 + b0 

    if beginning_wealth <= 0:
        print("\nCannot calculate CAGR: Initial total wealth (I0 + b0) must be positive.")
        return

    if T_ret_years <= 0:
        print("\nCannot calculate CAGR: Simulation duration (T_ret_years) must be positive.")
        return

    nominal_cagr_list = []
    real_cagr_list = []

    for res in simulation_results:
        success, _, final_investment, final_bank_balance, annual_inflations_seq, *_ = res
        
        if success: # Only consider successful simulations for CAGR
            # --- Calculate Nominal CAGR ---
            total_final_wealth_nominal = final_investment + final_bank_balance
            
            if total_final_wealth_nominal >= 0:
                if total_final_wealth_nominal == 0:
                    cagr_nominal = -1.0 # -100% return
                else:
                    cagr_nominal = (total_final_wealth_nominal / beginning_wealth)**(1 / T_ret_years) - 1
                nominal_cagr_list.append(cagr_nominal)
            else:
                # If nominal final wealth is negative, CAGR is conceptually complex or undefined for growth.
                # For simplicity, we can treat it as -100% for reporting purposes if initial wealth was positive.
                # Or, if you want to strictly exclude, you could 'continue' here.
                # For now, we'll include as -1.0 if beginning wealth was positive and final negative
                if beginning_wealth > 0:
                     nominal_cagr_list.append(-1.0) # Represents complete loss relative to positive start


            # --- Calculate Real CAGR ---
            cumulative_inflation_factor = np.prod(1 + annual_inflations_seq) if len(annual_inflations_seq) > 0 else 1.0
            total_final_wealth_real = total_final_wealth_nominal / cumulative_inflation_factor

            if total_final_wealth_real >= 0:
                if total_final_wealth_real == 0:
                    cagr_real = -1.0 # -100% return
                else:
                    cagr_real = (total_final_wealth_real / beginning_wealth)**(1 / T_ret_years) - 1
                real_cagr_list.append(cagr_real)
            else:
                # Same handling for real final wealth being negative
                if beginning_wealth > 0:
                    real_cagr_list.append(-1.0)


    if not nominal_cagr_list: # If no successful simulations, both lists will be empty
        print("\nNo successful simulations to calculate CAGR for.")
        return

    nominal_cagrs = np.array(nominal_cagr_list)
    real_cagrs = np.array(real_cagr_list)

    # --- Print Nominal CAGR Results ---
    print("\n--- Compound Annual Growth Rate (CAGR) of Total Wealth (Nominal Terms) ---")
    print(f"Worst Case CAGR: {np.min(nominal_cagrs):.2%}")
    print(f"Average Case (Median) CAGR: {np.median(nominal_cagrs):.2%}")
    print(f"Best Case CAGR: {np.max(nominal_cagrs):.2%}")
    print("-------------------------------------------------------------------------")

    # --- Print Real CAGR Results ---
    print("\n--- Compound Annual Growth Rate (CAGR) of Total Wealth (Real Terms) ---")
    print(f"Worst Case CAGR: {np.min(real_cagrs):.2%}")
    print(f"Average Case (Median) CAGR: {np.median(real_cagrs):.2%}")
    print(f"Best Case CAGR: {np.max(real_cagrs):.2%}")
    print("---------------------------------------------------------------------")
