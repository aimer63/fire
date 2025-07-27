#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation
from firestarter.config.config import PortfolioRebalance


def test_rebalance_event_successful(initialized_simulation: Simulation) -> None:
    """
    Tests that a rebalance event correctly updates portfolio weights and
    redistributes liquid assets when triggered on the first month of a
    scheduled year.
    """
    sim = initialized_simulation
    rebalance_year = 5

    # Define a new set of weights for the rebalance event
    new_weights = {"stocks": 0.6, "bonds": 0.4, "str": 0.0, "fun": 0.0}
    rebalance_event = PortfolioRebalance(year=rebalance_year, weights=new_weights)
    initial_weights = {"stocks": 1.0, "bonds": 0.0, "str": 0.0, "fun": 0.0}
    initial_rebalance = PortfolioRebalance(year=0, weights=initial_weights)

    # Replace the rebalance schedule on the simulation instance from the fixture
    sim.portfolio_rebalances = [initial_rebalance, rebalance_event]

    # Set a specific portfolio for this test
    sim.state.portfolio = {
        "stocks": 100_000.0,
        "bonds": 0.0,
        "str": 0.0,
        "fun": 0.0,
        "ag": 10_000.0,  # Non-liquid, should be ignored
    }
    sim.init()  # Re-initialize state with new portfolio and rebalances

    total_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if k != "inflation"
    )

    # Execute the rebalance logic for the correct month
    month_to_test = rebalance_year * 12
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # 1. Target weights should be updated in the state
    for k, v in new_weights.items():
        assert sim.state.current_target_portfolio_weights[k] == v
    # Check that all other assets have weight 0.0
    for k in sim.state.current_target_portfolio_weights:
        if k not in new_weights:
            assert sim.state.current_target_portfolio_weights[k] == 0.0

    # 2. Assets should be redistributed according to the new weights
    for asset, weight in new_weights.items():
        expected_value = total_liquid_assets * weight
        assert sim.state.portfolio[asset] == pytest.approx(expected_value)


def test_rebalance_no_event_scheduled_for_year(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that no rebalance occurs if none is scheduled for the given year.
    """
    sim = initialized_simulation
    rebalance_year = 5
    year_to_test = 3  # A year with no rebalance scheduled

    initial_weights = {"stocks": 1.0, "bonds": 0.0, "str": 0.0, "fun": 0.0}
    initial_rebalance = PortfolioRebalance(year=0, weights=initial_weights)
    rebalance_event = PortfolioRebalance(
        year=rebalance_year,
        weights={"stocks": 1.0, "bonds": 0.0, "str": 0.0, "fun": 0.0},
    )
    sim.portfolio_rebalances = [initial_rebalance, rebalance_event]
    sim.init()  # Re-initialize state with new rebalances

    # Store initial state for comparison
    initial_weights = sim.state.current_target_portfolio_weights.copy()
    initial_portfolio = sim.state.portfolio.copy()

    # Execute logic for a month where no rebalance is scheduled
    month_to_test = year_to_test * 12
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # Weights and asset values should be unchanged
    assert sim.state.current_target_portfolio_weights == initial_weights
    assert sim.state.portfolio == initial_portfolio


def test_rebalance_not_first_month_of_year(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that no rebalance occurs even if one is scheduled for the year,
    if it is not the first month of that year.
    """
    sim = initialized_simulation
    rebalance_year = 5

    initial_weights = {"stocks": 1.0, "bonds": 0.0, "str": 0.0, "fun": 0.0}
    initial_rebalance = PortfolioRebalance(year=0, weights=initial_weights)
    new_weights = {"stocks": 0.6, "bonds": 0.4, "str": 0.0, "fun": 0.0}
    rebalance_event = PortfolioRebalance(year=rebalance_year, weights=new_weights)
    sim.portfolio_rebalances = [initial_rebalance, rebalance_event]
    sim.init()  # Re-initialize state with new rebalances

    # Store initial state for comparison
    initial_weights = sim.state.current_target_portfolio_weights.copy()
    initial_portfolio = sim.state.portfolio.copy()

    # Execute logic for the *second* month of the scheduled year
    month_to_test = rebalance_year * 12 + 1
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # Weights and asset values should be unchanged
    assert sim.state.current_target_portfolio_weights == initial_weights
    assert sim.state.portfolio == initial_portfolio
