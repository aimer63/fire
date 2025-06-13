from typing import Any, List, Dict
from datetime import datetime
from pathlib import Path
import os
import json
from firestarter.core.helpers import calculate_cagr


def generate_markdown_report(
    simulation_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str,
    plots: Dict[str, str],
) -> str:
    """
    Generate a Markdown report summarizing the FIRE simulation results.
    Saves the report to the specified output directory and returns the report path.
    """
    now = datetime.now()
    run_date = now.strftime("%Y-%m-%d %H:%M")
    md = []

    md.append("# FIRE Simulation Report\n")
    md.append(f"**Run date:** {run_date}\n")
    md.append(f"**Config:** `{config.get('config_file', 'N/A')}`\n")

    # --- Config parameters dump ---
    md.append("## Loaded Configuration Parameters\n")
    md.append("```json")
    md.append(json.dumps(config, indent=2, ensure_ascii=False))
    md.append("```\n")

    # --- Simulation summary ---
    md.append("## FIRE Plan Simulation Summary\n")
    num_simulations = len(simulation_results)
    num_failed = sum(1 for r in simulation_results if not r["success"])
    num_successful = num_simulations - num_failed
    success_rate = 100.0 * num_successful / num_simulations if num_simulations else 0.0

    md.append(f"- **Success Rate:** {success_rate:.2f}%")
    md.append(f"- **Number of failed simulations:** {num_failed}")

    if num_failed > 0:
        avg_months_failed = (
            sum(r["months_lasted"] for r in simulation_results if not r["success"]) / num_failed
        )
        md.append(f"- **Average months lasted in failed simulations:** {avg_months_failed:.1f}")

    # --- Key scenario details: Nominal and Real Results ---
    successful_sims = [r for r in simulation_results if r["success"]]
    if not successful_sims:
        md.append("\nNo successful simulations to report.\n")
    else:
        # --- Nominal Results ---
        md.append("\n## Nominal Results (cases selected by nominal final wealth)\n")
        sorted_by_nominal = sorted(successful_sims, key=lambda r: r["final_nominal_wealth"])
        worst_nom = sorted_by_nominal[0]
        best_nom = sorted_by_nominal[-1]
        median_nom = sorted_by_nominal[len(sorted_by_nominal) // 2]

        def case_md_nominal(label: str, case: Dict[str, Any]) -> str:
            lines = [f"\n### {label} Successful Case"]
            lines.append(f"- Final Wealth (Nominal): {case['final_nominal_wealth']:,.2f} EUR")
            lines.append(f"- Final Wealth (Real): {case['final_real_wealth']:,.2f} EUR")
            initial_wealth = case["initial_total_wealth"]
            final_wealth = case["final_nominal_wealth"]
            months_lasted = case["months_lasted"]
            years = months_lasted / 12 if months_lasted else 0
            if initial_wealth is not None and final_wealth is not None and years > 0:
                cagr = calculate_cagr(initial_wealth, final_wealth, years)
                lines.append(f"- Your life CAGR (Nominal): {cagr:.2%}")
            else:
                lines.append("- Your life CAGR (Nominal): N/A")
            allocations = case["final_allocations_nominal"]
            total_nominal = case["final_nominal_wealth"]
            bank = case["final_bank_balance"]
            if allocations:
                total_assets = sum(allocations.values())
                alloc_percent = ", ".join(
                    f"{k}: {v / total_assets * 100:.1f}%" if total_assets else f"{k}: 0.0%"
                    for k, v in allocations.items()
                )
                lines.append(f"- Final Allocations (percent): {alloc_percent}")
            if allocations:
                lines.append("\n| Asset        | Value (EUR)      |")
                lines.append("|--------------|------------------|")
                for k, v in allocations.items():
                    lines.append(f"| {k:<12} | {v:,.2f}           |")
                lines.append(f"| Bank         | {bank:,.2f}           |")
                summed = sum(allocations.values()) + bank
                lines.append(f"| **Sum**      | **{summed:,.2f}**     |")
                if abs(summed - total_nominal) > 1e-2:
                    lines.append("| **WARNING**  | **Sum does not match final total wealth!** |")
            return "\n".join(lines)

        md.append(case_md_nominal("Worst", worst_nom))
        md.append(case_md_nominal("Median", median_nom))
        md.append(case_md_nominal("Best", best_nom))

        # --- Real Results ---
        md.append("\n## Real Results (cases selected by real final wealth)\n")
        sorted_by_real = sorted(successful_sims, key=lambda r: r["final_real_wealth"])
        worst_real = sorted_by_real[0]
        best_real = sorted_by_real[-1]
        median_real = sorted_by_real[len(sorted_by_real) // 2]

        def case_md_real(label: str, case: Dict[str, Any]) -> str:
            lines = [f"\n### {label} Successful Case"]
            lines.append(f"- Final Wealth (Real): {case['final_real_wealth']:,.2f} EUR")
            lines.append(f"- Final Wealth (Nominal): {case['final_nominal_wealth']:,.2f} EUR")
            initial_wealth = case["initial_total_wealth"]
            final_wealth = case["final_real_wealth"]
            months_lasted = case["months_lasted"]
            years = months_lasted / 12 if months_lasted else 0
            if initial_wealth is not None and final_wealth is not None and years > 0:
                cagr = calculate_cagr(initial_wealth, final_wealth, years)
                lines.append(f"- Your life CAGR (Real): {cagr:.2%}")
            else:
                lines.append("- Your life CAGR (Real): N/A")
            allocations = case["final_allocations_nominal"]
            total_nominal = case["final_nominal_wealth"]
            bank = case["final_bank_balance"]
            if allocations:
                total_assets = sum(allocations.values())
                alloc_percent = ", ".join(
                    f"{k}: {v / total_assets * 100:.1f}%" if total_assets else f"{k}: 0.0%"
                    for k, v in allocations.items()
                )
                lines.append(f"- Final Allocations (percent): {alloc_percent}")
            if allocations:
                lines.append("\n| Asset        | Value (EUR)      |")
                lines.append("|--------------|------------------|")
                for k, v in allocations.items():
                    lines.append(f"| {k:<12} | {v:,.2f}           |")
                lines.append(f"| Bank         | {bank:,.2f}           |")
                summed = sum(allocations.values()) + bank
                lines.append(f"| **Sum**      | **{summed:,.2f}**     |")
                if abs(summed - total_nominal) > 1e-2:
                    lines.append("| **WARNING**  | **Sum does not match final total wealth!** |")
            return "\n".join(lines)

        md.append(case_md_real("Worst", worst_real))
        md.append(case_md_real("Median", median_real))
        md.append(case_md_real("Best", best_real))

    # --- Plots ---
    md.append("\n## Plots\n")
    report_dir = Path(output_dir)
    for plot_name, plot_path in plots.items():
        rel_path = os.path.relpath(plot_path, start=report_dir)
        md.append(f"- [{plot_name}]({rel_path})")

    md.append(f"\n---\n*Generated by FIRE Simulator on {run_date}*")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"summary_{now.strftime('%Y%m%d_%H%M')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return str(report_path)
