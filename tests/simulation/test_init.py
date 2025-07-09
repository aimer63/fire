#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

from firestarter.core.constants import ASSET_KEYS
from firestarter.core.simulation import Simulation
# Fixtures are now in conftest.py


def test_simulation_months(initialized_simulation: Simulation) -> None:
    """Tests the simulation_months property."""
    expected_months = initialized_simulation.det_inputs.years_to_simulate * 12
    assert initialized_simulation.simulation_months == expected_months


def test_simulation_init_initial_state(initialized_simulation: Simulation) -> None:
    """Tests the initial state variables after simulation.init() is called."""
    state = initialized_simulation.state
    det_inputs = initialized_simulation.det_inputs
    initial_portfolio = det_inputs.initial_portfolio
    first_rebalance = initialized_simulation.portfolio_rebalances[0]

    assert state, "State dictionary should be initialized."

    assert state.current_bank_balance == det_inputs.initial_bank_balance
    assert state.portfolio["stocks"] == initial_portfolio["stocks"]
    assert state.portfolio["bonds"] == initial_portfolio["bonds"]
    assert state.portfolio["str"] == initial_portfolio["str"]
    assert state.portfolio["fun"] == initial_portfolio["fun"]
    assert state.portfolio["real_estate"] == initial_portfolio["real_estate"]

    expected_target_weights = {
        "stocks": first_rebalance.weights["stocks"],
        "bonds": first_rebalance.weights["bonds"],
        "str": first_rebalance.weights["str"],
        "fun": first_rebalance.weights["fun"],
    }
    assert state.current_target_portfolio_weights == expected_target_weights

    expected_initial_total_wealth = (
        det_inputs.initial_bank_balance
        + initial_portfolio[
            "stocks"
        ]  # Only stocks has a non-zero value in basic_initial_assets
        + initial_portfolio["real_estate"]
    )
    assert state.initial_total_wealth == expected_initial_total_wealth
    assert not state.simulation_failed


def test_simulation_precompute_sequences(initialized_simulation: Simulation) -> None:
    """Tests that _precompute_sequences correctly populates the state."""
    state = initialized_simulation.state
    total_months = initialized_simulation.simulation_months

    # Check for existence and length of key sequences
    # Check for existence and length of key sequences
    sequences_to_check = [
        "monthly_nominal_pension_sequence",
        "monthly_nominal_salary_sequence",
    ]
    for seq_name in sequences_to_check:
        assert hasattr(state, seq_name), f"'{seq_name}' should be in state."
        assert len(getattr(state, seq_name)) == total_months, (
            f"'{seq_name}' should have length {total_months}."
        )

    assert hasattr(state, "monthly_returns_sequences")
    # ASSET_KEYS + inflation
    assert len(state.monthly_returns_sequences) == len(ASSET_KEYS) + 1
    for asset_key in state.monthly_returns_sequences:
        assert len(state.monthly_returns_sequences[asset_key]) == total_months

    assert hasattr(state, "monthly_cumulative_inflation_factors")
    assert len(state.monthly_cumulative_inflation_factors) == total_months + 1, (
        "Cumulative inflation factors should have length total_months + 1."
    )

    assert hasattr(state, "monthly_returns_sequences")
    for asset_key in ASSET_KEYS:
        assert asset_key in state.monthly_returns_sequences
        assert len(state.monthly_returns_sequences[asset_key]) == total_months
