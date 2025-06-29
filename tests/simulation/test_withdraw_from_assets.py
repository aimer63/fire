# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from firestarter.core.simulation import Simulation

# The withdrawal priority is documented in `_withdraw_from_assets` as:
# STR, Bonds, Stocks, Fun
# These tests are based on that priority order.


def test_withdraw_from_assets_success_single_asset(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests a successful withdrawal that is fully covered by the first-priority asset.
    """
    sim = initialized_simulation
    sim.initial_assets = {
        "stocks": 50_000.0,
        "bonds": 50_000.0,
        "str": 50_000.0,
        "fun": 50_000.0,
        "real_estate": 0.0,
    }
    sim.init()
    sim.state.current_bank_balance = 0.0

    amount_to_withdraw = 10_000.0
    sim._withdraw_from_assets(amount_to_withdraw)

    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(amount_to_withdraw)
    assert sim.state.liquid_assets["str"] == pytest.approx(40_000.0)  # 50k - 10k
    assert sim.state.liquid_assets["bonds"] == pytest.approx(50_000.0)  # Unchanged


def test_withdraw_from_assets_success_multiple_assets(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests a successful withdrawal that depletes the first-priority asset and
    partially draws from the second.
    """
    sim = initialized_simulation
    sim.initial_assets = {
        "stocks": 50_000.0,
        "bonds": 50_000.0,
        "str": 10_000.0,  # Lower amount to force depletion
        "fun": 50_000.0,
        "real_estate": 0.0,
    }
    sim.init()
    sim.state.current_bank_balance = 0.0

    amount_to_withdraw = 25_000.0  # More than is in 'str'
    sim._withdraw_from_assets(amount_to_withdraw)

    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(amount_to_withdraw)
    assert sim.state.liquid_assets["str"] == pytest.approx(0.0)  # Depleted
    assert sim.state.liquid_assets["bonds"] == pytest.approx(35_000.0)  # 50k - 15k
    assert sim.state.liquid_assets["stocks"] == pytest.approx(50_000.0)  # Unchanged


def test_withdraw_from_assets_failure_insufficient_funds(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests a failed withdrawal where the requested amount exceeds the total value
    of all liquid assets.
    """
    sim = initialized_simulation
    sim.initial_assets = {
        "stocks": 10_000.0,
        "bonds": 10_000.0,
        "str": 10_000.0,
        "fun": 10_000.0,
        "real_estate": 0.0,
    }
    sim.init()
    sim.state.current_bank_balance = 0.0
    total_liquid_assets = sum(sim.state.liquid_assets.values())

    amount_to_withdraw = 50_000.0  # More than the 40k available
    sim._withdraw_from_assets(amount_to_withdraw)

    assert sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(total_liquid_assets)
    assert sum(sim.state.liquid_assets.values()) == pytest.approx(0.0)


def test_withdraw_from_assets_success_all_assets(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests a successful withdrawal that draws from all liquid assets according to priority.
    """
    sim = initialized_simulation
    sim.initial_assets = {
        "stocks": 10_000.0,
        "bonds": 10_000.0,
        "str": 10_000.0,
        "fun": 50_000.0,
        "real_estate": 0.0,
    }
    sim.init()
    sim.state.current_bank_balance = 0.0

    # This will deplete str, bonds, stocks, and take 5k from fun
    amount_to_withdraw = 35_000.0
    sim._withdraw_from_assets(amount_to_withdraw)

    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(amount_to_withdraw)
    assert sim.state.liquid_assets["str"] == pytest.approx(0.0)
    assert sim.state.liquid_assets["bonds"] == pytest.approx(0.0)
    assert sim.state.liquid_assets["stocks"] == pytest.approx(0.0)
    assert sim.state.liquid_assets["fun"] == pytest.approx(
        45_000.0
    )  # 50k - 5k needed after others depleted
