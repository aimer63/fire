#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation


def test_handle_bank_account_successful_top_up(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that the bank balance is successfully topped up from liquid assets
    when it falls below the lower bound.
    """
    sim = initialized_simulation
    month_to_test = 12
    lower_bound = 10_000.0
    upper_bound = 20_000.0

    # Configure bounds and set initial state
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "bank_lower_bound": lower_bound,
            "bank_upper_bound": upper_bound,
            "initial_portfolio": {
                "stocks": 50_000.0,
                "bonds": 0,
                "str": 0,
                "fun": 0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()
    sim.state.current_bank_balance = 4_000.0  # Below lower bound

    initial_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    inflation_factor = sim.state.monthly_cumulative_inflation_factors[month_to_test]
    expected_nominal_lower_bound = lower_bound * inflation_factor
    shortfall = expected_nominal_lower_bound - sim.state.current_bank_balance

    # Execute the method
    sim._handle_bank_account(month_to_test)

    # --- Assertions ---
    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(expected_nominal_lower_bound)
    current_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    assert current_liquid_assets == pytest.approx(initial_liquid_assets - shortfall)


def test_handle_bank_account_failed_top_up(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that the simulation fails when the bank balance is below the lower
    bound and liquid assets are insufficient to cover the shortfall.
    """
    sim = initialized_simulation
    month_to_test = 12
    lower_bound = 10_000.0
    upper_bound = 20_000.0

    # Configure bounds and set initial state
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "bank_lower_bound": lower_bound,
            "bank_upper_bound": upper_bound,
            "initial_portfolio": {
                "stocks": 3_000.0,
                "bonds": 0,
                "str": 0,
                "fun": 0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()
    sim.state.current_bank_balance = 4_000.0  # Shortfall of 6_000

    # Execute the method
    sim._handle_bank_account(month_to_test)

    # --- Assertions ---
    assert sim.state.simulation_failed


def test_handle_bank_account_invest_excess(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that excess cash is invested into liquid assets when the bank
    balance is above the upper bound.
    """
    sim = initialized_simulation
    month_to_test = 12
    lower_bound = 10_000.0
    upper_bound = 20_000.0

    # Configure bounds and set initial state
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "bank_lower_bound": lower_bound,
            "bank_upper_bound": upper_bound,
            "initial_portfolio": {
                "stocks": 50_000.0,
                "bonds": 50_000.0,
                "str": 0,
                "fun": 0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()
    sim.state.current_bank_balance = 25_000.0  # Above upper bound

    initial_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    inflation_factor = sim.state.monthly_cumulative_inflation_factors[month_to_test]
    expected_nominal_upper_bound = upper_bound * inflation_factor
    excess = sim.state.current_bank_balance - expected_nominal_upper_bound

    # Execute the method
    sim._handle_bank_account(month_to_test)

    # --- Assertions ---
    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(expected_nominal_upper_bound)
    current_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    assert current_liquid_assets == pytest.approx(initial_liquid_assets + excess)


def test_handle_bank_account_no_action(initialized_simulation: Simulation) -> None:
    """
    Tests that no action is taken when the bank balance is within its bounds.
    """
    sim = initialized_simulation
    month_to_test = 12
    lower_bound = 10_000.0
    upper_bound = 20_000.0

    # Configure bounds and set initial state
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "bank_lower_bound": lower_bound,
            "bank_upper_bound": upper_bound,
            "initial_portfolio": {
                "stocks": 50_000.0,
                "bonds": 0,
                "str": 0,
                "fun": 0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()
    sim.state.current_bank_balance = 15_000.0  # Within bounds

    initial_bank_balance = sim.state.current_bank_balance
    initial_liquid_assets = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )

    # Execute the method
    sim._handle_bank_account(month_to_test)

    # --- Assertions ---
    assert not sim.state.simulation_failed
    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)
    assert sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    ) == pytest.approx(initial_liquid_assets)
