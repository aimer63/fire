# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from firestarter.core.simulation import Simulation


def test_apply_monthly_returns(initialized_simulation: Simulation) -> None:
    """
    Tests that monthly returns (positive, negative, and zero) are correctly
    applied to all assets, including real estate.
    """
    sim = initialized_simulation

    # Set initial asset values
    sim.initial_assets = {
        "stocks": 100_000.0,
        "bonds": 50_000.0,
        "str": 20_000.0,
        "fun": 10_000.0,
        "real_estate": 500_000.0,
    }
    sim.init()

    # Store initial values to compare against
    initial_liquid_assets = sim.state.liquid_assets.copy()
    initial_real_estate_value = sim.state.current_real_estate_value
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
        sim.state.monthly_returns_lookup[asset][month_to_test] = rate

    # Execute the method under test
    sim._apply_monthly_returns(month_to_test)

    # --- Assertions ---
    # Check that each liquid asset was updated correctly
    assert sim.state.liquid_assets["stocks"] == pytest.approx(
        initial_liquid_assets["stocks"] * (1 + mock_returns["stocks"])
    )
    assert sim.state.liquid_assets["bonds"] == pytest.approx(
        initial_liquid_assets["bonds"] * (1 + mock_returns["bonds"])
    )
    assert sim.state.liquid_assets["str"] == pytest.approx(
        initial_liquid_assets["str"] * (1 + mock_returns["str"])
    )
    assert sim.state.liquid_assets["fun"] == pytest.approx(
        initial_liquid_assets["fun"] * (1 + mock_returns["fun"])
    )

    # Check that real estate was also updated correctly
    assert sim.state.current_real_estate_value == pytest.approx(
        initial_real_estate_value * (1 + mock_returns["real_estate"])
    )

    # Check that bank balance is untouched
    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)
