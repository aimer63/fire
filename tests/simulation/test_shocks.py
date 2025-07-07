# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
from firestarter.core.simulation import Simulation
from firestarter.config.config import Shock, Shocks


def test_shock_overwrites_return_sequence(initialized_simulation: Simulation) -> None:
    """
    Tests that a shock event correctly overwrites the pre-computed monthly
    return sequence for the specified asset and year.
    """
    sim = initialized_simulation
    shock_year = 4
    asset_to_shock = "stocks"
    annual_shock_rate = -0.50  # A 50% crash

    # Define the shock event and assign it correctly to the simulation
    shock = Shock(
        year=shock_year,
        asset=asset_to_shock,
        magnitude=annual_shock_rate,
        description="Market Crash",
    )
    # The `shock_events` attribute is a `Shocks` object, not a list
    sim.shock_events = Shocks(events=[shock])

    # Run the pre-computation logic which applies the shocks
    sim._precompute_sequences()

    # --- Assertions ---
    # Calculate the expected constant monthly rate from the annual shock rate
    expected_monthly_rate = (1.0 + annual_shock_rate) ** (1.0 / 12.0) - 1.0

    # Get the return sequence for the shocked asset
    shocked_sequence = sim.state.monthly_returns_sequences[asset_to_shock]

    # Check that all 12 months of the shock year have the new rate
    for month_offset in range(12):
        month_idx = shock_year * 12 + month_offset
        assert shocked_sequence[month_idx] == pytest.approx(expected_monthly_rate)


def test_non_shocked_year_is_unaffected(initialized_simulation: Simulation) -> None:
    """
    Tests that years without a shock retain their original stochastic returns.
    """
    sim = initialized_simulation
    shock_year = 4
    asset_to_shock = "stocks"
    annual_shock_rate = -0.20

    # 1. Precompute sequences WITHOUT the shock to get a baseline
    sim._precompute_sequences()
    baseline_returns = np.copy(sim.state.monthly_returns_sequences[asset_to_shock])

    # 2. Define the shock and re-run the pre-computation
    shock = Shock(
        year=shock_year,
        asset=asset_to_shock,
        magnitude=annual_shock_rate,
        description="Test Shock",
    )
    sim.shock_events = Shocks(events=[shock])
    sim._precompute_sequences()
    shocked_returns = sim.state.monthly_returns_sequences[asset_to_shock]

    # 3. Assert that a non-shocked month retains its original value
    # We check the month immediately before the shock period starts
    month_to_check = shock_year * 12 - 1
    if month_to_check >= 0:
        assert shocked_returns[month_to_check] == pytest.approx(
            baseline_returns[month_to_check]
        )
