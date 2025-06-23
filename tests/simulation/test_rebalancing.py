import pytest
from firestarter.core.simulation import Simulation
from firestarter.config.config import PortfolioRebalances, PortfolioRebalance


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
    rebalance_event = PortfolioRebalance(year=rebalance_year, **new_weights)
    sim.portfolio_rebalances = PortfolioRebalances(rebalances=[rebalance_event])

    # Set initial assets to be clearly imbalanced
    sim.initial_assets = {
        "stocks": 100_000.0,
        "bonds": 0.0,
        "str": 0.0,
        "fun": 0.0,
        "real_estate": 0.0,
    }
    sim.init()

    total_liquid_assets = sum(sim.state.liquid_assets.values())

    # Execute the rebalance logic for the correct month
    month_to_test = rebalance_year * 12
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # 1. Target weights should be updated in the state
    assert sim.state.current_target_portfolio_weights == new_weights

    # 2. Liquid assets should be redistributed according to the new weights
    for asset, weight in new_weights.items():
        expected_value = total_liquid_assets * weight
        assert sim.state.liquid_assets[asset] == pytest.approx(expected_value)


def test_rebalance_no_event_scheduled_for_year(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that no rebalance occurs if none is scheduled for the given year.
    """
    sim = initialized_simulation
    rebalance_year = 5
    year_to_test = 3  # A year with no rebalance scheduled

    rebalance_event = PortfolioRebalance(
        year=rebalance_year, stocks=1.0, bonds=0.0, str=0.0, fun=0.0
    )
    sim.portfolio_rebalances = PortfolioRebalances(rebalances=[rebalance_event])
    sim.init()

    # Store initial state for comparison
    initial_weights = sim.state.current_target_portfolio_weights.copy()
    initial_assets = sim.state.liquid_assets.copy()

    # Execute logic for a month where no rebalance is scheduled
    month_to_test = year_to_test * 12
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # Weights and asset values should be unchanged
    assert sim.state.current_target_portfolio_weights == initial_weights
    assert sim.state.liquid_assets == initial_assets


def test_rebalance_not_first_month_of_year(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that no rebalance occurs even if one is scheduled for the year,
    if it is not the first month of that year.
    """
    sim = initialized_simulation
    rebalance_year = 5

    new_weights = {"stocks": 0.6, "bonds": 0.4, "str": 0.0, "fun": 0.0}
    rebalance_event = PortfolioRebalance(year=rebalance_year, **new_weights)
    sim.portfolio_rebalances = PortfolioRebalances(rebalances=[rebalance_event])
    sim.init()

    # Store initial state for comparison
    initial_weights = sim.state.current_target_portfolio_weights.copy()
    initial_assets = sim.state.liquid_assets.copy()

    # Execute logic for the *second* month of the scheduled year
    month_to_test = rebalance_year * 12 + 1
    sim._rebalance_if_needed(month_to_test)

    # --- Assertions ---
    # Weights and asset values should be unchanged
    assert sim.state.current_target_portfolio_weights == initial_weights
    assert sim.state.liquid_assets == initial_assets
