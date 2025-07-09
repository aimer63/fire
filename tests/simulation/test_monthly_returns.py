#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation


def test_apply_monthly_returns(initialized_simulation: Simulation) -> None:
    """
    Tests that monthly returns (positive, negative, and zero) are correctly
    applied to all assets, including real estate.
    """
    sim = initialized_simulation

    # Define initial portfolio
    initial_portfolio = {
        "stocks": 100_000.0,
        "bonds": 50_000.0,
        "str": 20_000.0,
        "fun": 10_000.0,
        "real_estate": 500_000.0,
    }

    # Override det_inputs to set the specific portfolio for this test
    sim.det_inputs = sim.det_inputs.model_copy(
        update={"initial_portfolio": initial_portfolio}
    )
    sim.init()  # Re-initialize state with the new portfolio

    # Store initial values to compare against
    initial_portfolio_values = sim.state.portfolio.copy()
    initial_bank_balance = sim.state.current_bank_balance

    # Define mock monthly returns for a specific month (e.g., month 12)
    month_to_test = 12
    mock_returns = {
        "stocks": 0.02,  # +2%
        "bonds": -0.005,  # -0.5%
        "str": 0.001,  # +0.1%
        "fun": 0.0,  # 0%
        "real_estate": 0.005,  # +0.5%
    }

    # Manually inject these returns into the precomputed sequences in the state
    for asset, rate in mock_returns.items():
        sim.state.monthly_returns_sequences[asset][month_to_test] = rate

    # Execute the method under test
    sim._apply_monthly_returns(month_to_test)

    # --- Assertions ---
    # Check that each asset in the portfolio was updated correctly
    assert sim.state.portfolio["stocks"] == pytest.approx(
        initial_portfolio_values["stocks"] * (1 + mock_returns["stocks"])
    )
    assert sim.state.portfolio["bonds"] == pytest.approx(
        initial_portfolio_values["bonds"] * (1 + mock_returns["bonds"])
    )
    assert sim.state.portfolio["str"] == pytest.approx(
        initial_portfolio_values["str"] * (1 + mock_returns["str"])
    )
    assert sim.state.portfolio["fun"] == pytest.approx(
        initial_portfolio_values["fun"] * (1 + mock_returns["fun"])
    )

    # Check that real estate was also updated correctly
    assert sim.state.portfolio["real_estate"] == pytest.approx(
        initial_portfolio_values["real_estate"] * (1 + mock_returns["real_estate"])
    )

    # Check that bank balance is untouched
    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)
