#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation


def test_record_results_initialization_and_first_month(
    initialized_simulation: Simulation,
):
    """
    Tests that _record_results initializes the results structure on first call
    and correctly records the state for the first month.
    """
    sim = initialized_simulation
    total_months = sim.simulation_months

    # Set a mock state for month 0 with realistic values for all assets
    mock_portfolio = {k: 0.0 for k in sim.assets.keys()}
    # Assign some values to liquid assets for testing
    for asset in mock_portfolio:
        if sim.assets[asset].withdrawal_priority is not None:
            mock_portfolio[asset] = 100_000.0
    sim.state.portfolio = mock_portfolio

    sim.det_inputs = sim.det_inputs.model_copy(
        update={"initial_bank_balance": 25_000.0}
    )
    sim.init()  # Re-initialize state with the new portfolio

    # Initialize results structure for all assets
    sim.results = {
        "wealth_history": [None] * total_months,
        "bank_balance_history": [None] * total_months,
    }
    for asset in sim.assets:
        sim.results[f"{asset}_history"] = [None] * total_months

    # Record state for month 0
    sim._record_results(month=0)

    # Check that results dictionary is initialized for all assets
    for asset in sim.assets:
        assert f"{asset}_history" in sim.results
        assert len(sim.results[f"{asset}_history"]) == total_months

    # Check that wealth and bank balance are recorded
    assert sim.results["wealth_history"][0] == pytest.approx(
        sim.state.current_bank_balance + sum(sim.state.portfolio.values())
    )
    assert sim.results["bank_balance_history"][0] == pytest.approx(
        sim.state.current_bank_balance
    )

    # Check that asset histories are recorded correctly for month 0
    for asset in sim.assets:
        expected_value = sim.state.portfolio.get(asset, 0.0)
        assert sim.results[f"{asset}_history"][0] == pytest.approx(expected_value)

    # Check that other months are still None
    for key in sim.results:
        if isinstance(sim.results[key], list) and len(sim.results[key]) > 1:
            assert all(x is None for x in sim.results[key][1:])


def test_record_results_subsequent_month(initialized_simulation: Simulation):
    """
    Tests that _record_results correctly records data for a subsequent month
    without altering previous records.
    """
    sim = initialized_simulation
    total_months = sim.simulation_months

    # Set initial state for month 0 with all assets
    initial_portfolio = {k: 0.0 for k in sim.assets.keys()}
    for asset in initial_portfolio:
        if sim.assets[asset].withdrawal_priority is not None:
            initial_portfolio[asset] = 200_000.0
    sim.state.portfolio = initial_portfolio
    sim.state.current_bank_balance = 20_000.0
    sim.init()

    # Initialize results structure for all assets
    sim.results = {
        "wealth_history": [None] * total_months,
        "bank_balance_history": [None] * total_months,
    }
    for asset in sim.assets:
        sim.results[f"{asset}_history"] = [None] * total_months

    sim._record_results(month=0)
    month_0_bank = sim.results["bank_balance_history"][0]
    month_0_portfolio_snapshot = dict(sim.state.portfolio)

    # Set state for month 1
    next_portfolio = {k: v for k, v in initial_portfolio.items()}
    for asset in next_portfolio:
        if sim.assets[asset].withdrawal_priority is not None:
            next_portfolio[asset] += 10_000.0
    sim.state.portfolio = next_portfolio
    sim.state.current_bank_balance = 22_000.0
    sim.init()
    sim._record_results(month=1)

    # Check values for month 1
    expected_wealth_1 = sim.state.current_bank_balance + sum(
        sim.state.portfolio.values()
    )
    assert sim.results["wealth_history"][1] == pytest.approx(expected_wealth_1)
    assert sim.results["bank_balance_history"][1] == pytest.approx(
        sim.state.current_bank_balance
    )

    for asset in sim.assets:
        expected_value = sim.state.portfolio.get(asset, 0.0)
        assert sim.results[f"{asset}_history"][1] == pytest.approx(expected_value)

    # Check month 0 is unchanged
    assert sim.results["bank_balance_history"][0] == pytest.approx(month_0_bank)
    for asset in sim.assets:
        expected_value = month_0_portfolio_snapshot.get(asset, 0.0)
        assert sim.results[f"{asset}_history"][0] == pytest.approx(expected_value)
