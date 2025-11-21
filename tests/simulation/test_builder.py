#
# Copyright (c) 2025-Present Imerio Dall'Olio
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from unittest.mock import Mock
from firecast.core.simulation import SimulationBuilder, Simulation


def test_simulation_builder_new_initial_state():
    """
    Tests that SimulationBuilder.new() creates an instance with all
    configurable attributes initialized to None.
    """
    builder = SimulationBuilder.new()

    assert isinstance(builder, SimulationBuilder), "Should be an instance of SimulationBuilder"
    assert builder.det_inputs is None, "det_inputs should be None initially"
    assert builder.assets is None, "market_assumptions should be None initially"
    assert builder.correlation_matrix is None, "correlation_matrix should be None initially"
    assert builder.portfolio_rebalances is None, "portfolio_rebalances should be None initially"
    assert builder.shock_events is None, "shock_events should be None initially"
    assert builder.sim_params is None, "sim_params should be None initially"


@pytest.mark.parametrize(
    "setter_method_name, attribute_name",
    [
        ("set_det_inputs", "det_inputs"),
        ("set_assets", "assets"),
        ("set_correlation_matrix", "correlation_matrix"),
        ("set_portfolio_rebalances", "portfolio_rebalances"),
        ("set_shock_events", "shock_events"),
        ("set_sim_params", "sim_params"),
    ],
)
def test_simulation_builder_setters(setter_method_name, attribute_name):
    """
    Tests all setter methods of SimulationBuilder.
    """
    builder = SimulationBuilder.new()
    mock_value = Mock()

    setter_method = getattr(builder, setter_method_name)
    returned_builder = setter_method(mock_value)

    assert (
        getattr(builder, attribute_name) is mock_value
    ), f"{attribute_name} attribute should be set"
    assert returned_builder is builder, "Setter method should return the builder instance"


def test_simulation_builder_build_success(complete_builder: SimulationBuilder) -> None:
    """
    Tests that build() returns a Simulation instance when all attributes are set.
    """
    simulation: Simulation = complete_builder.build()
    assert isinstance(simulation, Simulation), "Should return an instance of Simulation"

    # Check if the simulation instance received the correct attributes
    assert simulation.det_inputs is complete_builder.det_inputs
    assert simulation.assets is complete_builder.assets
    assert simulation.portfolio_rebalances is complete_builder.portfolio_rebalances
    assert simulation.shock_events is complete_builder.shock_events
    assert simulation.sim_params is complete_builder.sim_params


@pytest.mark.parametrize(
    "missing_attribute, error_message_part",
    [
        ("det_inputs", "det_inputs must be set"),
        ("assets", "assets must be set"),
        ("correlation_matrix", "correlation_matrix must be set"),
        ("portfolio_rebalances", "portfolio_rebalances must be set"),
        ("shock_events", "shock_events must be set"),
        ("sim_params", "sim_params (SimulationParameters) must be set"),
    ],
)
def test_simulation_builder_build_failure_missing_attributes(
    missing_attribute: str, error_message_part: str
) -> None:
    """
    Tests that build() raises ValueError if a required attribute is missing.
    """
    builder: SimulationBuilder = SimulationBuilder.new()
    # Set all attributes except the one we want to test for being missing
    all_attributes: dict[str, Mock] = {
        "det_inputs": Mock(name="det_inputs"),
        "assets": Mock(name="assets"),
        "correlation_matrix": Mock(name="correlation_matrix"),
        "portfolio_rebalances": Mock(name="portfolio_rebalances"),
        "shock_events": Mock(name="shock_events"),
        "sim_params": Mock(name="sim_params"),
    }

    for attr_name, mock_value in all_attributes.items():
        if attr_name != missing_attribute:
            setter_method = getattr(builder, f"set_{attr_name}")
            setter_method(mock_value)

    with pytest.raises(ValueError) as excinfo:
        builder.build()

    assert error_message_part in str(excinfo.value)
