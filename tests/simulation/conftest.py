#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from typing import Dict

from firecast.core.simulation import SimulationBuilder
from firecast.config.config import (
    IncomeStep,
    ExpenseStep,
    DeterministicInputs,
    PortfolioRebalance,
    SimulationParameters,
    PlannedContribution,
    PlannedExtraExpense,
)
from firecast.config.correlation_matrix import CorrelationMatrix


@pytest.fixture
def basic_sim_params() -> SimulationParameters:
    """Minimal SimulationParameters for testing."""
    return SimulationParameters(num_simulations=1, random_seed=123)


@pytest.fixture
def basic_assets():
    from firecast.config.config import Asset

    return {
        "stocks": Asset(mu=0.07, sigma=0.15, withdrawal_priority=2),
        "bonds": Asset(mu=0.03, sigma=0.05, withdrawal_priority=1),
        "str": Asset(mu=0.01, sigma=0.01, withdrawal_priority=0),
        "fun": Asset(mu=0.10, sigma=0.30, withdrawal_priority=3),
        "ag": Asset(mu=0.04, sigma=0.10, withdrawal_priority=4),
        "real_estate": Asset(mu=0.03, sigma=0.10),  # Added illiquid asset
        "inflation": Asset(mu=0.02, sigma=0.01),
    }


@pytest.fixture
def basic_det_inputs() -> DeterministicInputs:
    """Minimal DeterministicInputs for testing."""
    return DeterministicInputs(
        initial_bank_balance=5000.0,
        bank_lower_bound=2000.0,
        bank_upper_bound=10000.0,
        years_to_simulate=5,
        monthly_income_steps=[IncomeStep(year=0, monthly_amount=0.0)],
        monthly_expenses_steps=[ExpenseStep(year=0, monthly_amount=0.0)],
        planned_contributions=[PlannedContribution(amount=10000, year=1)],
        annual_fund_fee=0.001,  # 0.1%
        planned_extra_expenses=[PlannedExtraExpense(amount=500, year=2)],
        income_inflation_factor=1.0,
        income_end_year=5,
        monthly_pension=0,
        pension_inflation_factor=1.0,
        pension_start_year=30,
    )


@pytest.fixture
def basic_portfolio_rebalances():
    # Return a simple list of PortfolioRebalance objects, not a PortfolioRebalances wrapper.
    rebalance_event_1 = PortfolioRebalance(
        year=0, weights={"stocks": 0.6, "bonds": 0.3, "str": 0.1, "fun": 0.0}
    )
    rebalance_event_2 = PortfolioRebalance(
        year=2,
        weights={
            "stocks": 0.5,
            "bonds": 0.4,
            "str": 0.05,
            "fun": 0.05,
        },
    )
    return [rebalance_event_1, rebalance_event_2]


@pytest.fixture
def basic_shocks():
    # Return an empty list for shocks in the flat config.
    return []


@pytest.fixture
def basic_correlation_matrix():
    return CorrelationMatrix(
        assets_order=["stocks", "bonds", "str", "fun", "ag", "inflation"],
        matrix=[
            [1.00, -0.20, 0.00, 0.70, 0.60, -0.10],
            [-0.20, 1.00, 0.20, -0.10, 0.10, -0.30],
            [0.00, 0.20, 1.00, 0.00, 0.00, 0.10],
            [0.70, -0.10, 0.00, 1.00, 0.50, -0.05],
            [0.60, 0.10, 0.00, 0.50, 1.00, 0.40],
            [-0.10, -0.30, 0.10, -0.05, 0.40, 1.00],
        ],
    )


@pytest.fixture
def identity_correlation_matrix():
    return CorrelationMatrix(
        assets_order=["stocks", "bonds", "str", "fun", "ag", "inflation"],
        matrix=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )


@pytest.fixture
def complete_builder(
    basic_det_inputs,
    basic_assets,
    basic_correlation_matrix,
    basic_portfolio_rebalances,
    basic_shocks,
    basic_sim_params,
) -> SimulationBuilder:
    """Returns a SimulationBuilder with all required attributes set using real fixtures."""
    builder: SimulationBuilder = SimulationBuilder.new()
    builder.set_det_inputs(basic_det_inputs)
    builder.set_assets(basic_assets)
    builder.set_correlation_matrix(basic_correlation_matrix)
    builder.set_portfolio_rebalances(basic_portfolio_rebalances)
    builder.set_shock_events(basic_shocks)
    builder.set_sim_params(basic_sim_params)
    return builder


@pytest.fixture
def initialized_simulation(
    basic_det_inputs: DeterministicInputs,
    basic_assets,
    basic_correlation_matrix,
    basic_portfolio_rebalances,
    basic_shocks,
    basic_sim_params: SimulationParameters,
):
    """
    Provides a basic, initialized Simulation instance for testing.
    """
    builder = SimulationBuilder.new()
    simulation = (
        builder.set_det_inputs(basic_det_inputs)
        .set_assets(basic_assets)
        .set_correlation_matrix(basic_correlation_matrix)
        .set_portfolio_rebalances(basic_portfolio_rebalances)
        .set_shock_events(basic_shocks)
        .set_sim_params(basic_sim_params)
        .build()
    )
    simulation.init()
    return simulation
