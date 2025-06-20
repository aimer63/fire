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
    initial_assets = initialized_simulation.initial_assets
    first_rebalance = initialized_simulation.portfolio_rebalances.rebalances[0]

    assert state, "State dictionary should be initialized."

    assert state["current_bank_balance"] == det_inputs.initial_bank_balance
    assert state["liquid_assets"]["stocks"] == initial_assets["stocks"]
    assert state["liquid_assets"]["bonds"] == initial_assets["bonds"]
    assert state["liquid_assets"]["str"] == initial_assets["str"]
    assert state["liquid_assets"]["fun"] == initial_assets["fun"]
    assert state["current_real_estate_value"] == initial_assets["real_estate"]

    expected_target_weights = {
        "stocks": first_rebalance.stocks,
        "bonds": first_rebalance.bonds,
        "str": first_rebalance.str,
        "fun": first_rebalance.fun,
    }
    assert state["current_target_portfolio_weights"] == expected_target_weights

    expected_initial_total_wealth = (
        det_inputs.initial_bank_balance
        + initial_assets[
            "stocks"
        ]  # Only stocks has a non-zero value in basic_initial_assets
        + initial_assets["real_estate"]
    )
    assert state["initial_total_wealth"] == expected_initial_total_wealth
    assert not state["simulation_failed"]


def test_simulation_precompute_sequences(initialized_simulation: Simulation) -> None:
    """Tests that _precompute_sequences correctly populates the state."""
    state = initialized_simulation.state
    total_months = initialized_simulation.simulation_months

    # Check for existence and length of key sequences
    sequences_to_check = [
        "monthly_inflations_sequence",
        "monthly_stocks_returns_sequence",
        "monthly_bonds_returns_sequence",
        "monthly_str_returns_sequence",
        "monthly_fun_returns_sequence",
        "monthly_real_estate_returns_sequence",
        "monthly_nominal_pension_sequence",
        "monthly_nominal_salary_sequence",
    ]
    for seq_name in sequences_to_check:
        assert seq_name in state, f"'{seq_name}' should be in state."
        assert len(state[seq_name]) == total_months, (
            f"'{seq_name}' should have length {total_months}."
        )

    assert "monthly_cumulative_inflation_factors" in state
    assert len(state["monthly_cumulative_inflation_factors"]) == total_months + 1, (
        "Cumulative inflation factors should have length total_months + 1."
    )

    assert "monthly_returns_lookup" in state
    for asset_key in ["Stocks", "Bonds", "STR", "Fun", "Real Estate"]:
        assert asset_key in state["monthly_returns_lookup"]
        assert len(state["monthly_returns_lookup"][asset_key]) == total_months
