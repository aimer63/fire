"""
Defines the SimulationState dataclass, which encapsulates the mutable state of a
single simulation run.
This includes current asset balances, portfolio weights, precomputed stochastic
sequences, and simulation time tracking.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class SimulationState:
    current_bank_balance: float
    liquid_assets: Dict[str, float]
    current_real_estate_value: float
    current_target_portfolio_weights: Dict[str, float]
    initial_total_wealth: float
    simulation_failed: bool

    # Precomputed stochastic sequences and lookups
    monthly_stocks_returns_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_bonds_returns_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_str_returns_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_fun_returns_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_real_estate_returns_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_inflation_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    monthly_cumulative_inflation_factors: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_returns_lookup: Dict[str, np.ndarray] = field(default_factory=dict)
    monthly_nominal_pension_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_nominal_salary_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    # Tracking current simulation time
    current_month_index: int = 0
    current_year_index: int = 0
