import pytest
from typing import Dict, Any

from firestarter.core.simulation import Simulation, SimulationBuilder
from firestarter.config.config import (
    DeterministicInputs,
    MarketAssumptions,
    PortfolioRebalance,
    PortfolioRebalances,
    Shocks,
    SimulationParameters,
)
from firestarter.core.constants import ASSET_KEYS


@pytest.fixture
def basic_sim_params() -> SimulationParameters:
    """Minimal SimulationParameters for testing."""
    return SimulationParameters(num_simulations=1, random_seed=123)


@pytest.fixture
def basic_initial_assets() -> Dict[str, float]:
    """Minimal initial_assets for testing."""
    assets = {key: 0.0 for key in ASSET_KEYS}
    assets["stocks"] = 100000.0
    assets["real_estate"] = 0.0
    return assets


@pytest.fixture
def basic_det_inputs() -> DeterministicInputs:
    """Minimal DeterministicInputs for testing."""
    return DeterministicInputs(
        initial_investment=100000.0,  # Should match sum of liquid assets in initial_assets
        initial_bank_balance=5000.0,
        bank_lower_bound=2000.0,
        bank_upper_bound=10000.0,
        years_to_simulate=5,
        monthly_salary=0,
        salary_inflation_factor=1.0,
        salary_start_year=0,
        salary_end_year=0,
        monthly_pension=0,
        pension_inflation_factor=1.0,
        pension_start_year=30,
        monthly_investment_contribution=0,
        planned_contributions=[],
        annual_fund_fee=0.001,  # 0.1%
        monthly_expenses=0,
        planned_extra_expenses=[],
        planned_house_purchase_cost=0,
        house_purchase_year=None,
    )


@pytest.fixture
def basic_market_assumptions() -> MarketAssumptions:
    """Minimal MarketAssumptions for testing."""
    return MarketAssumptions(
        stock_mu=0.07,
        stock_sigma=0.15,
        bond_mu=0.03,
        bond_sigma=0.05,
        str_mu=0.01,
        str_sigma=0.01,
        fun_mu=0.10,
        fun_sigma=0.30,
        real_estate_mu=0.04,
        real_estate_sigma=0.10,
        pi_mu=0.02,
        pi_sigma=0.01,
    )


@pytest.fixture
def basic_portfolio_rebalances() -> PortfolioRebalances:
    """Minimal PortfolioRebalances for testing."""
    rebalance_event_1 = PortfolioRebalance(
        year=0, stocks=0.6, bonds=0.3, str=0.1, fun=0.0
    )
    rebalance_event_2 = PortfolioRebalance(
        year=2,
        stocks=0.5,
        bonds=0.4,
        str=0.05,
        fun=0.05,
    )
    return PortfolioRebalances(rebalances=[rebalance_event_1, rebalance_event_2])


@pytest.fixture
def basic_shocks() -> Shocks:
    """Minimal Shocks for testing (no shocks)."""
    return Shocks(events=[])


@pytest.fixture
def initialized_simulation(
    basic_det_inputs: DeterministicInputs,
    basic_market_assumptions: MarketAssumptions,
    basic_portfolio_rebalances: PortfolioRebalances,
    basic_shocks: Shocks,
    basic_initial_assets: Dict[str, float],
    basic_sim_params: SimulationParameters,
) -> Simulation:
    """
    Provides a basic, initialized Simulation instance for testing.
    This fixture is now in conftest.py and available to all tests in this directory.
    """
    builder = SimulationBuilder.new()
    simulation = (
        builder.set_det_inputs(basic_det_inputs)
        .set_market_assumptions(basic_market_assumptions)
        .set_portfolio_rebalances(basic_portfolio_rebalances)
        .set_shock_events(basic_shocks)
        .set_initial_assets(basic_initial_assets)
        .set_sim_params(basic_sim_params)
        .build()
    )
    simulation.init()
    return simulation
