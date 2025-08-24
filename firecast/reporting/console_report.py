#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

"""
This module provides functions for generating and printing human-readable summaries
of FIRE Monte Carlo simulation results directly to the console.

The main function, print_console_summary, displays a comprehensive summary of simulation
outcomes, including success rates, failure statistics, and detailed statistics for both
nominal and real wealth distributions. It also presents representative simulation cases
(worst, median, best) for both nominal and real wealth, including final allocations and
asset breakdowns.

Designed for aiding users in quickly understanding the key results and scenario
characteristics of their FIRE plan simulations.
"""

from typing import Any, List, Dict
import numpy as np
import json
from firecast.utils.helpers import calculate_cagr


def dump_config_parameters(config: Dict[str, Any]) -> None:
    """
    Print all loaded configuration parameters to the console for transparency and reproducibility.
    """
    print("\n--- Loaded Configuration Parameters ---")
    print(json.dumps(config, indent=2, ensure_ascii=False))


def print_console_summary(simulation_results: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Print the main simulation summary and key scenario details to the console.
    Accepts the raw simulation results (list of dicts from Simulation.build_result()) and config.
    Only formats and presents data, does not compute except for CAGR.
    """

    print("\n--- FIRE Plan Simulation Summary ---")
    num_simulations = len(simulation_results)
    num_failed = sum(1 for r in simulation_results if not r["success"])
    num_successful = num_simulations - num_failed
    success_rate = 100.0 * num_successful / num_simulations if num_simulations else 0.0

    print(f"FIRE Plan Success Rate: {success_rate:.2f}%")
    print(f"Number of failed simulations: {num_failed}")

    if num_failed > 0:
        avg_months_failed = (
            sum(r["months_lasted"] for r in simulation_results if not r["success"]) / num_failed
        )
        print(f"Average months lasted in failed simulations: {avg_months_failed:.1f}")

    successful_sims = [r for r in simulation_results if r["success"]]
    if not successful_sims:
        print("\nNo successful simulations to report.\n")
        return

    nominal_final_wealths = np.array([s["final_nominal_wealth"] for s in successful_sims])
    real_final_wealths = np.array([s["final_real_wealth"] for s in successful_sims])

    # Calculate percentiles for Nominal Wealth
    p25_nominal_wealth = np.percentile(nominal_final_wealths, 25)
    median_nominal_wealth = np.percentile(nominal_final_wealths, 50)  # or np.median()
    p75_nominal_wealth = np.percentile(nominal_final_wealths, 75)
    iqr_nominal_wealth = p75_nominal_wealth - p25_nominal_wealth

    # Calculate percentiles for Real Wealth
    p25_real_wealth = np.percentile(real_final_wealths, 25)
    median_real_wealth = np.percentile(real_final_wealths, 50)  # or np.median()
    p75_real_wealth = np.percentile(real_final_wealths, 75)
    iqr_real_wealth = p75_real_wealth - p25_real_wealth

    print("\n--- Final Wealth Distribution Statistics (Successful Simulations) ---")
    print("Nominal Final Wealth:")
    print(f"  Median (P50): {median_nominal_wealth:,.2f} ")
    print(f"  25th Percentile (P25): {p25_nominal_wealth:,.2f} ")
    print(f"  75th Percentile (P75): {p75_nominal_wealth:,.2f} ")
    print(f"  Interquartile Range (P75-P25): {iqr_nominal_wealth:,.2f} ")
    print("Real Final Wealth (Today's Money):")
    print(f"  Median (P50): {median_real_wealth:,.2f} ")
    print(f"  25th Percentile (P25): {p25_real_wealth:,.2f} ")
    print(f"  75th Percentile (P75): {p75_real_wealth:,.2f} ")
    print(f"  Interquartile Range (P75-P25): {iqr_real_wealth:,.2f} ")

    # --- Nominal Results ---
    print("\n=== Nominal Results (cases selected by nominal final wealth) ===\n")
    sorted_by_nominal = sorted(successful_sims, key=lambda r: r["final_nominal_wealth"])
    worst_nom = sorted_by_nominal[0]
    best_nom = sorted_by_nominal[-1]
    median_nom = sorted_by_nominal[len(sorted_by_nominal) // 2]

    def print_case_nominal(label: str, case: Dict[str, Any]) -> None:
        print(f"{label} Successful Case:")
        print(f"  Final Wealth (Nominal): {case['final_nominal_wealth']:,.2f} ")
        print(f"  Final Wealth (Real): {case['final_real_wealth']:,.2f} ")
        print(f"  Cumulative Inflation Factor: {case['final_cumulative_inflation_factor']:.4f}")
        initial_wealth = case["initial_total_wealth"]
        final_wealth = case["final_nominal_wealth"]
        months_lasted = case["months_lasted"]
        years = months_lasted / 12 if months_lasted else 0
        if initial_wealth is not None and final_wealth is not None and years > 0:
            cagr = calculate_cagr(initial_wealth, final_wealth, years)
            print(f"  Your life CAGR (Nominal): {cagr:.2%}")
        else:
            print("  Your life CAGR (Nominal): N/A")
        allocations = case["final_allocations_nominal"]
        total_nominal = case["final_nominal_wealth"]
        bank = case["final_bank_balance"]
        if allocations:
            total_assets = sum(v for k, v in allocations.items() if k != "inflation")
            alloc_percent = ", ".join(
                f"{k}: {v / total_assets * 100:.1f}%" if total_assets else f"{k}: 0.0%"
                for k, v in allocations.items()
                if k != "inflation"
            )
            print(f"  Final Allocations (percent): {alloc_percent}")
        if allocations:
            asset_str = ", ".join(
                f"{k}: {v:,.2f} " for k, v in allocations.items() if k != "inflation"
            )
            print(f"  Nominal Asset Values: {asset_str}, Bank: {bank:,.2f} ")
            summed = sum(v for k, v in allocations.items() if k != "inflation") + bank
            if abs(summed - total_nominal) > 1e-2:
                print("  WARNING: Sum does not match final total wealth!")

    print_case_nominal("Worst", worst_nom)
    print()
    print_case_nominal("Median", median_nom)
    print()
    print_case_nominal("Best", best_nom)
    print()

    # --- Real Results ---
    print("=== Real Results (cases selected by real final wealth) ===\n")
    sorted_by_real = sorted(successful_sims, key=lambda r: r["final_real_wealth"])
    worst_real = sorted_by_real[0]
    best_real = sorted_by_real[-1]
    median_real = sorted_by_real[len(sorted_by_real) // 2]

    def print_case_real(label: str, case: Dict[str, Any]) -> None:
        print(f"{label} Successful Case:")
        print(f"  Final Wealth (Real): {case['final_real_wealth']:,.2f} ")
        print(f"  Final Wealth (Nominal): {case['final_nominal_wealth']:,.2f} ")
        print(f"  Cumulative Inflation Factor: {case['final_cumulative_inflation_factor']:.4f}")
        initial_wealth = case["initial_total_wealth"]
        final_wealth = case["final_real_wealth"]
        months_lasted = case["months_lasted"]
        years = months_lasted / 12 if months_lasted else 0
        if initial_wealth is not None and final_wealth is not None and years > 0:
            cagr = calculate_cagr(initial_wealth, final_wealth, years)
            print(f"  Your life CAGR (Real): {cagr:.2%}")
        else:
            print("  Your life CAGR (Real): N/A")
        allocations = case["final_allocations_nominal"]
        total_nominal = case["final_nominal_wealth"]
        bank = case["final_bank_balance"]
        if allocations:
            total_assets = sum(v for k, v in allocations.items() if k != "inflation")
            alloc_percent = ", ".join(
                f"{k}: {v / total_assets * 100:.1f}%" if total_assets else f"{k}: 0.0%"
                for k, v in allocations.items()
                if k != "inflation"
            )
            print(f"  Final Allocations (percent): {alloc_percent}")
        if allocations:
            asset_str = ", ".join(
                f"{k}: {v:,.2f} " for k, v in allocations.items() if k != "inflation"
            )
            print(f"  Nominal Asset Values: {asset_str}, Bank: {bank:,.2f} ")
            summed = sum(v for k, v in allocations.items() if k != "inflation") + bank
            if abs(summed - total_nominal) > 1e-2:
                print("  WARNING: Sum does not match final total wealth!")

    print_case_real("Worst", worst_real)
    print()
    print_case_real("Median", median_real)
    print()
    print_case_real("Best", best_real)
