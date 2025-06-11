from typing import Any, List, Dict
import json
from firestarter.core.helpers import calculate_cagr


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
    dump_config_parameters(config)

    print("\n--- FIRE Plan Simulation Summary ---")
    num_simulations = len(simulation_results)
    num_failed = sum(1 for r in simulation_results if not r.get("success", False))
    num_successful = num_simulations - num_failed
    success_rate = 100.0 * num_successful / num_simulations if num_simulations else 0.0

    print(f"FIRE Plan Success Rate: {success_rate:.2f}%")
    print(f"Number of failed simulations: {num_failed}")

    if num_failed > 0:
        avg_months_failed = (
            sum(
                r.get("months_lasted", 0) for r in simulation_results if not r.get("success", False)
            )
            / num_failed
        )
        print(f"Average months lasted in failed simulations: {avg_months_failed:.1f}")

    # --- Key scenario details: worst, median, best successful cases ---
    successful_sims = [r for r in simulation_results if r.get("success", False)]
    if not successful_sims:
        print("\nNo successful simulations to report.")
        return

    # Sort by real final wealth for scenario selection
    sorted_by_real = sorted(successful_sims, key=lambda r: r.get("final_real_wealth", 0.0))
    worst = sorted_by_real[0]
    best = sorted_by_real[-1]
    median = sorted_by_real[len(sorted_by_real) // 2]

    def print_case(label: str, case: Dict[str, Any]) -> None:
        print(f"\n{label} Successful Case:")
        print(f"  Final Wealth (Nominal): {case.get('final_nominal_wealth', 0.0):,.2f} EUR")
        print(f"  Final Wealth (Real): {case.get('final_real_wealth', 0.0):,.2f} EUR")
        # Calculate CAGR using initial and final nominal wealth and years
        initial_wealth = case.get("initial_total_wealth")
        final_wealth = case.get("final_nominal_wealth")
        months_lasted = case.get("months_lasted", 0)
        years = months_lasted / 12 if months_lasted else 0
        if initial_wealth is not None and final_wealth is not None and years > 0:
            cagr = calculate_cagr(initial_wealth, final_wealth, years)
            print(f"  Your life CAGR: {cagr:.2%}")
        else:
            print("  Your life CAGR: N/A")
        allocations = case.get("final_allocations_nominal", {})
        total_nominal = case.get("final_nominal_wealth", 0.0)
        bank = case.get("final_bank_balance", 0.0)
        # Print allocations as percentages
        if allocations:
            total_assets = sum(allocations.values())
            print("  Final Allocations (percent): ", end="")
            print(
                ", ".join(
                    f"{k}: {v / total_assets * 100:.1f}%" if total_assets else f"{k}: 0.0%"
                    for k, v in allocations.items()
                )
            )
        # Print nominal asset values
        if allocations:
            print("  Nominal Asset Values: ", end="")
            print(", ".join(f"{k}: {v:,.2f} EUR" for k, v in allocations.items()), end="")
            print(f", Bank: {bank:,.2f} EUR")
            summed = sum(allocations.values()) + bank
            if abs(summed - total_nominal) > 1e-2:
                print(
                    f"    WARNING: Sum of assets ({summed:,.2f}) != Final Nominal Wealth ({total_nominal:,.2f})"
                )

    print_case("Worst", worst)
    print_case("Median", median)
    print_case("Best", best)
