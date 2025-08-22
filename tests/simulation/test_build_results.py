#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation
from firestarter.config.config import PlannedExtraExpense


def _get_expected_allocations_from_history(
    sim: Simulation, month_index: int
) -> dict[str, float]:
    """Helper to build expected allocations from simulation history for a given month."""
    allocations = {}
    for asset in sim.assets:
        history_key = f"{asset}_history"
        if history_key in sim.results:
            allocations[asset] = sim.results[history_key][month_index]
        else:
            allocations[asset] = 0.0
    return allocations


def test_build_result_successful_simulation(initialized_simulation: Simulation):
    """
    Tests that _build_result correctly formats the output for a successful simulation
    that runs to completion.
    """
    sim = initialized_simulation
    result = sim.run()

    total_months = sim.simulation_months

    assert result["success"] is True
    assert result["months_lasted"] == total_months
    assert len(result["wealth_history"]) == total_months

    # Check that final values match the last recorded month in history
    last_month_index = total_months - 1
    expected_final_nominal_wealth = sim.results["wealth_history"][last_month_index]
    assert result["final_nominal_wealth"] == pytest.approx(
        expected_final_nominal_wealth
    )

    expected_final_cumulative_inflation = (
        sim.state.monthly_cumulative_inflation_factors[last_month_index]
    )
    assert result["final_cumulative_inflation_factor"] == pytest.approx(
        expected_final_cumulative_inflation
    )

    expected_final_real_wealth = (
        expected_final_nominal_wealth / expected_final_cumulative_inflation
    )
    assert result["final_real_wealth"] == pytest.approx(expected_final_real_wealth)
    assert result["final_real_wealth"] == pytest.approx(expected_final_real_wealth)

    expected_allocations = _get_expected_allocations_from_history(sim, last_month_index)
    assert result["final_allocations_nominal"] == expected_allocations


def test_build_result_failed_simulation_midway(initialized_simulation: Simulation):
    """
    Tests that _build_result correctly formats output for a simulation that fails
    partway through, ensuring values are from the last successful month.
    """
    sim = initialized_simulation
    # Engineer a failure by setting a huge one-time expense mid-simulation
    failure_year = sim.det_inputs.years_to_simulate // 2
    current_expenses = sim.det_inputs.planned_extra_expenses.copy()
    current_expenses.append(
        PlannedExtraExpense(year=failure_year, amount=1_000_000_000)
    )
    sim.det_inputs = sim.det_inputs.model_copy(
        update={"planned_extra_expenses": current_expenses}
    )
    sim.init()
    result = sim.run()

    failure_month = failure_year * 12

    assert result["success"] is False
    assert result["months_lasted"] == failure_month

    # Check history lengths are truncated to the point of failure
    assert len(result["wealth_history"]) == failure_month
    assert len(result["bank_balance_history"]) == failure_month

    # Check final values are from the month *before* failure
    last_successful_month_idx = failure_month - 1
    expected_final_nominal_wealth = sim.results["wealth_history"][
        last_successful_month_idx
    ]
    assert result["final_nominal_wealth"] == pytest.approx(
        expected_final_nominal_wealth
    )

    expected_allocations = _get_expected_allocations_from_history(
        sim, last_successful_month_idx
    )
    assert result["final_allocations_nominal"] == expected_allocations


def test_build_result_failed_simulation_immediately(
    initialized_simulation: Simulation,
):
    """
    Tests that _build_result correctly handles a simulation that fails on the
    first month (months_lasted == 0), reporting initial state values.
    """
    sim = initialized_simulation
    # Engineer an immediate failure by having no assets and high expenses
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_expenses": 1_000_000,
            "initial_bank_balance": 0,
            "planned_contributions": [],
        }
    )
    sim.init()
    result = sim.run()

    assert result["success"] is False
    assert result["months_lasted"] == 0

    # Histories should be empty
    assert len(result["wealth_history"]) == 0
    assert len(result["bank_balance_history"]) == 0

    # Final values should reflect the initial state
    assert result["final_nominal_wealth"] == sim.state.initial_total_wealth
    assert result["final_bank_balance"] == sim.det_inputs.initial_bank_balance
    assert result["final_cumulative_inflation_factor"] == 1.0
    assert result["initial_total_wealth"] == 0.0
    expected_allocations = {
        k: sim.state.portfolio[k]
        for k in result["final_allocations_nominal"].keys()
    }
    assert result["final_allocations_nominal"] == expected_allocations
