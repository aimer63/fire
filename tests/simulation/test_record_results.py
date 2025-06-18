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

    # Set a mock state for month 0 with more realistic values
    sim.det_inputs = sim.det_inputs.model_copy(
        update={"initial_bank_balance": 25_000.0}
    )
    sim.initial_assets = {
        "stocks": 500_000.0,
        "bonds": 250_000.0,
        "str": 100_000.0,
        "fun": 50_000.0,
        "real_estate": 0.0,
    }
    sim.init()  # Re-initialize to reflect the new inputs

    # Before first recording, results should be an empty dict
    assert sim.results == {}

    # Record state for month 0
    sim._record_results(month=0)

    # Check that results dictionary is now initialized
    assert "wealth_history" in sim.results
    assert len(sim.results["wealth_history"]) == total_months
    # Check that other months are still None
    assert all(x is None for x in sim.results["wealth_history"][1:])
    expected_wealth = 25_000.0 + 500_000.0 + 250_000.0 + 100_000.0 + 50_000.0 + 0.0
    assert sim.results["wealth_history"][0] == pytest.approx(expected_wealth)
    assert sim.results["bank_balance_history"][0] == pytest.approx(25_000.0)
    assert sim.results["stocks_history"][0] == pytest.approx(500_000.0)
    assert sim.results["bonds_history"][0] == pytest.approx(250_000.0)
    assert sim.results["str_history"][0] == pytest.approx(100_000.0)
    assert sim.results["fun_history"][0] == pytest.approx(50_000.0)
    assert sim.results["real_estate_history"][0] == pytest.approx(0.0)


def test_record_results_subsequent_month(initialized_simulation: Simulation):
    """
    Tests that _record_results correctly records data for a subsequent month
    without altering previous records.
    """
    sim = initialized_simulation

    # Record for month 0 with realistic values, assuming a house is owned
    sim.state["current_bank_balance"] = 20_000.0
    sim.state["liquid_assets"] = {
        "stocks": 500_000.0,
        "bonds": 250_000.0,
        "str": 100_000.0,
        "fun": 50_000.0,
    }
    sim.state["current_real_estate_value"] = 300_000.0
    sim._record_results(month=0)
    month_0_bank = sim.results["bank_balance_history"][0]

    # Set state for month 1
    sim.state["current_bank_balance"] = 22_000.0
    sim.state["liquid_assets"] = {
        "stocks": 510_000.0,
        "bonds": 255_000.0,
        "str": 100_000.0,
        "fun": 51_000.0,
    }
    sim.state["current_real_estate_value"] = 301_000.0
    sim._record_results(month=1)

    # Check values for month 1
    expected_wealth_1 = (
        22_000.0 + 510_000.0 + 255_000.0 + 100_000.0 + 51_000.0 + 301_000.0
    )
    assert sim.results["wealth_history"][1] == pytest.approx(expected_wealth_1)
    assert sim.results["bank_balance_history"][1] == pytest.approx(22_000.0)

    # Check month 0 is unchanged
    assert sim.results["bank_balance_history"][0] == pytest.approx(month_0_bank)
