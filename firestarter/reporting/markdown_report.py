from typing import Any, List, Dict
import numpy as np
import json
import os
from datetime import datetime
from firestarter.core.helpers import calculate_cagr


# Placeholder for a function to format config into Markdown
def format_config_for_markdown(config: Dict[str, Any]) -> List[str]:
    """Formats configuration parameters into a Markdown block."""
    md_config_lines = ["### Loaded Configuration Parameters\n"]
    md_config_lines.append("```json\n")
    md_config_lines.append(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    md_config_lines.append("```\n\n")
    return md_config_lines


# Placeholder for a function to format a single case (worst/median/best) into Markdown
def format_case_for_markdown(
    label: str, case: Dict[str, Any], case_type: str = "Nominal"
) -> List[str]:
    """Formats a single simulation case (worst, median, best) into Markdown lines."""
    md_case_lines = []
    initial_wealth = case["initial_total_wealth"]  # Use .get for safety
    months_lasted = case["months_lasted"]
    years = months_lasted / 12 if months_lasted is not None else 0

    final_nominal_wealth = case["final_nominal_wealth"]
    final_real_wealth = case["final_real_wealth"]

    cagr_wealth_for_calc_val = None

    if case_type == "Nominal":
        final_wealth_display = final_nominal_wealth
        final_wealth_other = final_real_wealth
        cagr_wealth_for_calc_val = final_nominal_wealth
        md_case_lines.append(f"#### {label} Successful Case (Nominal):\n\n")
        md_case_lines.append(
            f"- **Final Wealth (Nominal):** {final_wealth_display:,.2f} EUR\n"
        )
        md_case_lines.append(
            f"- **Final Wealth (Real):** {final_wealth_other:,.2f} EUR\n"
        )
        cagr_label = "Nominal"
    else:  # Real
        final_wealth_display = final_real_wealth
        final_wealth_other = final_nominal_wealth
        cagr_wealth_for_calc_val = final_real_wealth
        md_case_lines.append(f"#### {label} Successful Case (Real):\n\n")
        md_case_lines.append(
            f"- **Final Wealth (Real):** {final_wealth_display:,.2f} EUR\n"
        )
        md_case_lines.append(
            f"- **Final Wealth (Nominal):** {final_wealth_other:,.2f} EUR\n"
        )
        cagr_label = "Real"

    cagr = calculate_cagr(initial_wealth, cagr_wealth_for_calc_val, years)
    md_case_lines.append(f"- **Your life CAGR ({cagr_label}):** {cagr:.2%}\n")

    allocations = case["final_allocations_nominal"]
    bank = case["final_bank_balance"]

    total_assets = sum(allocations.values())

    alloc_percent_parts = []
    for k, v_asset in allocations.items():
        percent = v_asset / total_assets * 100
        alloc_percent_parts.append(f"{k}: {percent:.1f}%")
    md_case_lines.append(
        f"- **Final Allocations (percent):** {', '.join(alloc_percent_parts)}\n"
    )

    asset_value_parts = []
    for k, v_asset in allocations.items():
        asset_value_parts.append(f"{k}: {v_asset:,.2f} EUR")
    md_case_lines.append(
        f"- **Nominal Asset Values:** {', '.join(asset_value_parts)}, Bank: {bank:,.2f} EUR\n"
    )
    md_case_lines.append("\n")
    return md_case_lines


def generate_markdown_report(
    simulation_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    config_path: str,
    output_dir: str,
    plot_paths: Dict[str, str],
) -> None:  # Added config_path
    report_generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_filename = os.path.basename(config_path)

    md_content = ["# FIRE Plan Simulation Report\n\n"]
    md_content.append(f"Report generated on: {report_generated_time}\n")
    md_content.append(f"Using configuration: `{config_filename}`\n\n")

    # Add FIRE Plan Simulation Summary
    md_content.append("## FIRE Plan Simulation Summary\n\n")
    num_simulations = len(simulation_results)
    num_failed = sum(1 for r in simulation_results if not r["success"])
    num_successful = num_simulations - num_failed

    success_rate = 100.0 * num_successful / num_simulations

    md_content.append(f"- **FIRE Plan Success Rate:** {success_rate:.2f}%\n")
    md_content.append(f"- **Number of failed simulations:** {num_failed}\n")

    failed_months = [r["months_lasted"] for r in simulation_results if not r["success"]]
    avg_months_failed = sum(failed_months) / num_failed

    md_content.append(
        f"- **Average months lasted in failed simulations:** {avg_months_failed:.1f}\n"
    )
    md_content.append("\n")

    # Final Wealth Distribution Statistics
    successful_sims = [r for r in simulation_results if r["success"]]
    if (
        not successful_sims
    ):  # This check is for flow control, not error prevention for calculation
        md_content.append(
            "## Final Wealth Distribution Statistics (Successful Simulations)\n\n"
        )
        md_content.append("No successful simulations to report.\n\n")
    else:
        nominal_final_wealths = np.array(
            [s["final_nominal_wealth"] for s in successful_sims], dtype=float
        )
        real_final_wealths = np.array(
            [s["final_real_wealth"] for s in successful_sims], dtype=float
        )

        p25_nominal_wealth = np.percentile(nominal_final_wealths, 25)
        median_nominal_wealth = np.percentile(nominal_final_wealths, 50)
        p75_nominal_wealth = np.percentile(nominal_final_wealths, 75)
        iqr_nominal_wealth = p75_nominal_wealth - p25_nominal_wealth

        p25_real_wealth = np.percentile(real_final_wealths, 25)
        median_real_wealth = np.percentile(real_final_wealths, 50)
        p75_real_wealth = np.percentile(real_final_wealths, 75)
        iqr_real_wealth = p75_real_wealth - p25_real_wealth

        md_content.append(
            "## Final Wealth Distribution Statistics (Successful Simulations)\n\n"
        )
        md_content.append(
            "| Statistic                     | Nominal Final Wealth          | Real Final Wealth (Today's Money) |\n"
        )
        md_content.append(
            "|-------------------------------|-------------------------------|-----------------------------------|\n"
        )
        md_content.append(
            f"| Median (P50)                  | {median_nominal_wealth:,.2f} EUR    | {median_real_wealth:,.2f} EUR           |\n"
        )
        md_content.append(
            f"| 25th Percentile (P25)         | {p25_nominal_wealth:,.2f} EUR     | {p25_real_wealth:,.2f} EUR           |\n"
        )
        md_content.append(
            f"| 75th Percentile (P75)         | {p75_nominal_wealth:,.2f} EUR     | {p75_real_wealth:,.2f} EUR           |\n"
        )
        md_content.append(
            f"| Interquartile Range (P75-P25) | {iqr_nominal_wealth:,.2f} EUR     | {iqr_real_wealth:,.2f} EUR           |\n"
        )
        md_content.append("\n")

    # Nominal and Real Results sections
    if successful_sims:
        md_content.append(
            "## Nominal Results (cases selected by nominal final wealth)\n\n"
        )
        sorted_by_nominal = sorted(
            successful_sims, key=lambda r: r["final_nominal_wealth"]
        )
        if sorted_by_nominal:
            md_content.extend(
                format_case_for_markdown("Worst", sorted_by_nominal[0], "Nominal")
            )
            md_content.extend(
                format_case_for_markdown(
                    "Median", sorted_by_nominal[len(sorted_by_nominal) // 2], "Nominal"
                )
            )
            md_content.extend(
                format_case_for_markdown("Best", sorted_by_nominal[-1], "Nominal")
            )

        md_content.append("## Real Results (cases selected by real final wealth)\n\n")
        sorted_by_real = sorted(successful_sims, key=lambda r: r["final_real_wealth"])
        if sorted_by_real:
            md_content.extend(
                format_case_for_markdown("Worst", sorted_by_real[0], "Real")
            )
            md_content.extend(
                format_case_for_markdown(
                    "Median", sorted_by_real[len(sorted_by_real) // 2], "Real"
                )
            )
            md_content.extend(
                format_case_for_markdown("Best", sorted_by_real[-1], "Real")
            )

    # Visualizations
    md_content.append("## Visualizations\n\n")
    if plot_paths:
        for title, path in plot_paths.items():
            md_content.append(f"### {title}\n\n")
            md_content.append(f"![{title}]({path})\n\n")
    else:
        md_content.append("No plots generated or provided.\n\n")

    # Add Configuration Parameters
    md_content.extend(format_config_for_markdown(config))

    # Add footer
    md_content.append("---\n")  # Horizontal rule before footer
    md_content.append("Generated by firestarter FIRE Plan Monte Carlo simulation\n")

    # Filename generation and writing to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_filename = f"summary_{timestamp}.md"
    final_output_path = os.path.join(output_dir, report_filename)

    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write("".join(md_content))

    print(f"Markdown report generated at {final_output_path}")
