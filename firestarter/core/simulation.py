# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FIRE Simulation Engine

This module implements the core simulation logic for modeling financial independence
and early retirement (FIRE) scenarios. It provides the SimulationBuilder and Simulation
classes, supporting flexible configuration and execution of financial simulations.

Key features:
- Modular builder pattern for simulation setup
- Handles monthly flows: income, contributions, expenses, withdrawals, and asset
  rebalancing
- Evolves asset values monthly according to stochastic returns and inflation
- Supports planned shocks and house purchase
- Records detailed monthly histories of wealth, balances, and asset values
- Marks simulation as failed if withdrawals cannot be covered by liquid assets
- Supports user-defined correlation between asset returns and inflation via a
  validated correlation matrix, enabling realistic multivariate stochastic modeling
  of economic variables

See docs/notes/simulation_engine.md and docs/notes/correlation.md for a detailed
overview.
"""

from typing import (
    Dict,
    Any,
    Optional,
)  # Removed List as Shocks class handles list of events

import numpy as np

# Import the actual types from config.py
from firestarter.config.config import (
    DeterministicInputs,
    MarketAssumptions,
    PortfolioRebalance,
    Asset,
    Shock,
    SimulationParameters,
)

from firestarter.core.sequence_generator import SequenceGenerator
from firestarter.core.simulation_state import SimulationState


class SimulationBuilder:
    def __init__(self):
        self.det_inputs: Optional[DeterministicInputs] = None
        self.assets: Optional[dict[str, Asset]] = None
        self.market_assumptions: Optional[MarketAssumptions] = None
        self.portfolio_rebalances: Optional[list[PortfolioRebalance]] = None
        self.shock_events: Optional[list[Shock]] = None
        self.sim_params: Optional[SimulationParameters] = None

    @classmethod
    def new(cls):
        return cls()

    def set_det_inputs(self, det_inputs):
        self.det_inputs = det_inputs
        return self

    def set_assets(self, assets: dict[str, Asset]):
        self.assets = assets
        return self

    def set_market_assumptions(self, market_assumptions):
        self.market_assumptions = market_assumptions
        return self

    def set_portfolio_rebalances(self, portfolio_rebalances):
        self.portfolio_rebalances = portfolio_rebalances
        return self

    def set_shock_events(self, shock_events: list[Shock]):
        self.shock_events = shock_events
        return self

    def set_sim_params(self, sim_params):
        self.sim_params = sim_params
        return self

    def build(self):
        # Validate all required fields are set
        if self.det_inputs is None:
            raise ValueError("det_inputs must be set before building the simulation.")

        if self.assets is None:
            raise ValueError("assets must be set before building the simulation.")

        if self.market_assumptions is None:
            raise ValueError(
                "market_assumptions must be set before building the simulation."
            )
        if self.portfolio_rebalances is None:
            raise ValueError(
                "portfolio_rebalances must be set before building the simulation."
            )
        if self.shock_events is None:
            raise ValueError(
                "shock_events must be set before building the simulation."  # Corrected message
            )
        if self.sim_params is None:
            raise ValueError(
                "sim_params (SimulationParameters) must be set before building the simulation."
            )

        return Simulation(
            self.det_inputs,
            self.assets,
            self.market_assumptions,
            self.portfolio_rebalances,
            self.shock_events,
            self.sim_params,
        )


class Simulation:
    def __init__(
        self,
        det_inputs: DeterministicInputs,
        assets: dict[str, Asset],
        market_assumptions: MarketAssumptions,
        portfolio_rebalances: list[PortfolioRebalance],
        shock_events: list[Shock],
        sim_params: SimulationParameters,
    ):
        self.det_inputs: DeterministicInputs = det_inputs
        self.assets: dict[str, Asset] = assets
        self.market_assumptions: MarketAssumptions = market_assumptions
        self.portfolio_rebalances: list[PortfolioRebalance] = portfolio_rebalances
        self.shock_events: list[Shock] = shock_events
        self.sim_params: SimulationParameters = sim_params
        self.state: SimulationState = self._initialize_state()
        self.results: Dict[str, Any] = {}

    @property
    def simulation_months(self):
        """
        Total number of months to simulate, based on years_to_simulate.
        """
        return self.det_inputs.years_to_simulate * 12

    def init(self):
        self.state = self._initialize_state()
        self._precompute_sequences()

    def run(self) -> dict:
        """
        Runs the main simulation loop, handling all monthly flows and events.
        """
        total_months = self.simulation_months
        total_months = self.simulation_months

        # Initialize results dictionary dynamically based on assets in the portfolio
        self.results = {
            "wealth_history": [None] * total_months,
            "bank_balance_history": [None] * total_months,
        }
        for asset_name in self.state.portfolio:
            self.results[f"{asset_name}_history"] = [None] * total_months

        for month in range(total_months):
            self.state.current_month_index = month
            self.state.current_year_index = month // 12

            # 1. Income: Add salary and pension for the current year.
            self._process_income(month)

            # 2. Contributions: Apply planned contributions to liquid assets.
            self._handle_contributions(month)

            # 3. Expenses: Deduct regular and extra expenses from the bank account.
            self._handle_expenses(month)

            # 4. House Purchase: If scheduled, withdraw from assets to buy a house.
            self._handle_house_purchase(month)
            if self.state.simulation_failed:
                break  # Exit if house purchase failed

            # 5. Bank Account Management:
            self._handle_bank_account(month)
            if self.state.simulation_failed:
                break  # Exit if bank top-up failed

            # 6. Returns: Apply monthly returns to all assets.
            self._apply_monthly_returns(month)

            # 7. Apply Fund Fee (monthly)
            self._apply_fund_fee()

            # 8. Rebalancing: If scheduled, rebalance liquid assets.
            self._rebalance_if_needed(month)

            # 9. Recording: Save the current state.
            self._record_results(month)

        return self._build_result()

    # --- Helper methods (stubs for now) ---
    def _initialize_state(self):
        """
        Initialize all state variables for the simulation.
        Returns a SimulationState dataclass holding the simulation state.
        """
        # The portfolio is initialized directly from the user-provided initial assets
        initial_portfolio = self.det_inputs.initial_portfolio

        # Find the initial target portfolio weights from the first rebalance event
        first_reb = self.portfolio_rebalances[0]
        initial_target_weights = first_reb.weights

        # Calculate initial total wealth
        initial_total_wealth = self.det_inputs.initial_bank_balance + sum(
            initial_portfolio.values()
        )

        state = SimulationState(
            current_bank_balance=self.det_inputs.initial_bank_balance,
            portfolio=initial_portfolio,
            current_target_portfolio_weights=initial_target_weights,
            initial_total_wealth=initial_total_wealth,
            simulation_failed=False,
        )
        return state

    @staticmethod
    def annual_lognormal_to_monthly(mu_a: float, sigma_a: float) -> tuple[float, float]:
        """
        Convert annual lognormal parameters to monthly lognormal parameters.
        mu_a, sigma_a: annual lognormal parameters (mean and std of the underlying normal distribution)
        Returns (mu_m, sigma_m): monthly lognormal parameters
        """
        mu_m = mu_a / 12
        sigma_m = sigma_a / (12**0.5)
        return mu_m, sigma_m

    def _precompute_sequences(self):
        """
        Precompute all monthly sequences needed for the simulation.
        This version uses the SequenceGenerator to create correlated returns and inflation
        for a single simulation run.
        """
        det_inputs = self.det_inputs
        shock_events = self.shock_events
        total_years = det_inputs.years_to_simulate
        total_months = total_years * 12

        # --- Generate Correlated Sequences using the Generator ---
        generator = SequenceGenerator(
            assets=self.assets,
            market_assumptions=self.market_assumptions,
            num_sequences=1,  # A single simulation run is one sequence
            simulation_years=total_years,
            seed=self.sim_params.random_seed,
        )
        # Squeeze to remove the num_sequences dimension (shape: [1, months, assets] -> [months, assets])
        correlated_returns_array = np.squeeze(
            generator.correlated_monthly_returns, axis=0
        )

        # --- Convert array to dictionary for use in the simulation ---
        self.state.monthly_returns_sequences = {
            asset: correlated_returns_array[:, i]
            for i, asset in enumerate(generator.asset_and_inflation_order)
        }

        # --- Apply shocks (if any) ---
        # A shock's magnitude is an annual rate that replaces the stochastic rate
        # for that year.
        # This annual rate is then converted to a monthly rate and applied to all 12
        # months of that year.

        for shock in shock_events:  # Iterate over the .events attribute
            year_idx = shock.year
            # Iterate over each asset impacted by the shock
            for shock_asset, annual_shock_rate in shock.impact.items():
                if 0 <= year_idx < total_years:
                    # Convert the annual shock rate to an equivalent monthly rate
                    monthly_shock_rate = (
                        (1.0 + annual_shock_rate) ** (1.0 / 12.0)
                    ) - 1.0

                    if shock_asset in self.state.monthly_returns_sequences:
                        target_sequence = self.state.monthly_returns_sequences[
                            shock_asset
                        ]
                        for month_offset in range(12):
                            month_idx_in_simulation = year_idx * 12 + month_offset
                            if 0 <= month_idx_in_simulation < total_months:
                                target_sequence[month_idx_in_simulation] = (
                                    monthly_shock_rate
                                )

        monthly_inflation_sequence = self.state.monthly_returns_sequences["inflation"]

        # --- Cumulative inflation factors (monthly) ---
        monthly_cumulative_inflation_factors = np.ones(
            total_months + 1, dtype=np.float64
        )
        for month_idx in range(total_months):
            monthly_cumulative_inflation_factors[month_idx + 1] = (
                monthly_cumulative_inflation_factors[month_idx]
                * (1.0 + monthly_inflation_sequence[month_idx])
            )

        # --- Precompute nominal pension and salary monthly sequences with partial indexation ---
        monthly_nominal_pension_sequence = np.zeros(total_months, dtype=np.float64)
        monthly_nominal_salary_sequence = np.zeros(total_months, dtype=np.float64)

        pension_start_month_idx = det_inputs.pension_start_year * 12
        salary_start_month_idx = det_inputs.salary_start_year * 12
        salary_end_month_idx = det_inputs.salary_end_year * 12

        # Partial indexation: compound only a fraction of inflation each month
        pension_cumulative = det_inputs.monthly_pension
        salary_cumulative = det_inputs.monthly_salary

        for month_idx in range(total_months):
            # Pension
            if month_idx >= pension_start_month_idx:
                if month_idx == pension_start_month_idx:
                    pension_cumulative = det_inputs.monthly_pension
                else:
                    pension_cumulative *= 1.0 + (
                        monthly_inflation_sequence[month_idx - 1]
                        * det_inputs.pension_inflation_factor
                    )
                monthly_nominal_pension_sequence[month_idx] = pension_cumulative

            # Salary
            if salary_start_month_idx <= month_idx < salary_end_month_idx:
                if month_idx == salary_start_month_idx:
                    salary_cumulative = det_inputs.monthly_salary
                else:
                    salary_cumulative *= 1.0 + (
                        monthly_inflation_sequence[month_idx - 1]
                        * det_inputs.salary_inflation_factor
                    )
                monthly_nominal_salary_sequence[month_idx] = salary_cumulative

        self.state.monthly_cumulative_inflation_factors = (
            monthly_cumulative_inflation_factors
        )
        self.state.monthly_nominal_pension_sequence = monthly_nominal_pension_sequence
        self.state.monthly_nominal_salary_sequence = monthly_nominal_salary_sequence

    def _process_income(self, month):
        """
        For each month, add the precomputed *monthly* salary and pension for the current month.
        These values are now drawn and adjusted monthly.
        """
        income = 0.0

        # Pension (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state.monthly_nominal_pension_sequence):
            income += self.state.monthly_nominal_pension_sequence[month]

        # Salary (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state.monthly_nominal_salary_sequence):
            income += self.state.monthly_nominal_salary_sequence[month]

        self.state.current_bank_balance += float(income)

    def _handle_contributions(self, month):
        """
        Handles planned one-time contributions.
        Planned contribution are all applied the first month of the year
        Contributions are allocated according to the current target portfolio weights,
        but NEVER to real estate (see real_estate.md).
        """
        det_inputs = self.det_inputs

        if month % 12 == 0:
            current_year = month // 12
            for contribution in det_inputs.planned_contributions:
                if contribution.year == current_year:
                    self._invest_in_liquid_assets(contribution.amount)

    def _handle_expenses(self, month):
        """
        Deducts regular monthly expenses and planned extra expenses from the bank balance.
        Expenses are inflation-adjusted. Planned extra expenses are applied at the first month of their year.
        """
        det_inputs = self.det_inputs

        # Regular monthly expenses (inflation-adjusted)
        nominal_monthly_expenses = (
            det_inputs.monthly_expenses
            * self.state.monthly_cumulative_inflation_factors[month]
        )
        total_expenses = nominal_monthly_expenses

        # Planned extra expenses (applied at the first month of the year)
        if month % 12 == 0:
            current_year = month // 12
            for expense in det_inputs.planned_extra_expenses:
                if expense.year == current_year:
                    inflation_factor = self.state.monthly_cumulative_inflation_factors[
                        month
                    ]
                    nominal_amount = expense.amount * inflation_factor
                    total_expenses += nominal_amount

        # Deduct from bank balance
        self.state.current_bank_balance -= float(total_expenses)

    def _handle_house_purchase(self, month):
        """
        Handles the house purchase if scheduled for this month.
        Deducts the (inflation-adjusted) house cost from liquid assets,
        using the unified _withdraw_from_assets method.
        If assets are insufficient, marks the simulation as failed.
        Adds the house value to the portfolio under the 'real_estate' key.
        After purchase, rebalances remaining liquid assets according to current target portfolio weights.
        """
        det_inputs = self.det_inputs
        house_purchase_year = det_inputs.house_purchase_year
        house_cost_real = det_inputs.planned_house_purchase_cost

        if house_purchase_year is None or house_cost_real <= 0:
            return  # No house purchase scheduled

        purchase_month = house_purchase_year * 12

        # Only purchase at the first month of the scheduled year
        if month == purchase_month:
            # Inflation-adjusted nominal house cost
            cumulative_inflation = self.state.monthly_cumulative_inflation_factors[
                month
            ]
            nominal_house_cost = house_cost_real * cumulative_inflation

            # Withdraw funds from liquid assets to cover the cost. This temporarily
            # increases the bank balance.
            self._withdraw_from_assets(float(nominal_house_cost))

            # If the withdrawal failed, the simulation has already been marked as failed.
            # The bank balance will hold whatever could be withdrawn. Exit immediately.
            if self.state.simulation_failed:
                return

            # Subtract the house cost from the bank balance to complete the purchase
            self.state.current_bank_balance -= float(nominal_house_cost)

            # Add house value to the portfolio under the 'real_estate' key
            if "real_estate" in self.state.portfolio:
                self.state.portfolio["real_estate"] += float(nominal_house_cost)
            else:
                self.state.portfolio["real_estate"] = float(nominal_house_cost)

            # --- Rebalance remaining liquid assets according to current target portfolio weights ---
            self._rebalance_liquid_assets()

    def _handle_bank_account(self, month):
        """
        Ensures the bank balance is within [lower_bound, upper_bound].
        - If below lower_bound, withdraw from assets to top up.
        - If above upper_bound, invest excess into assets.
        - If assets are insufficient for top-up, mark simulation as failed.

        The bounds are specified in real terms (today's money) and must be converted
        to nominal using the cumulative inflation factor for the current month.
        """
        cumulative_inflation = self.state.monthly_cumulative_inflation_factors[month]
        lower = self.det_inputs.bank_lower_bound * cumulative_inflation
        upper = self.det_inputs.bank_upper_bound * cumulative_inflation

        # Top up if below lower bound
        if self.state.current_bank_balance < lower:
            shortfall = lower - self.state.current_bank_balance
            self._withdraw_from_assets(float(shortfall))
            # If withdrawal failed, _withdraw_from_assets sets simulation_failed
            if self.state.simulation_failed:
                return
            self.state.current_bank_balance = float(lower)

        # Invest excess if above upper bound
        if self.state.current_bank_balance > upper:
            excess = self.state.current_bank_balance - upper
            self._invest_in_liquid_assets(float(excess))
            self.state.current_bank_balance = float(upper)

    def _apply_monthly_returns(self, month):
        """
        Apply monthly returns to all asset values at the end of the month.
        """
        returns = self.state.monthly_returns_sequences
        for asset in self.state.portfolio:
            # The 'inflation' sequence is in `returns`, but not in the portfolio
            if asset in returns:
                self.state.portfolio[asset] = float(
                    self.state.portfolio[asset] * (1.0 + returns[asset][month])
                )

    def _apply_fund_fee(self) -> None:
        """
        Applies the fund fee to all liquid assets on a monthly basis.
        The annual fee is converted to a monthly equivalent.
        """
        annual_fee_percentage = self.det_inputs.annual_fund_fee
        if annual_fee_percentage > 0:
            # Convert annual fee to a simple monthly fee
            monthly_fee_percentage = annual_fee_percentage / 12.0

            for asset_key, asset_properties in self.assets.items():
                if asset_properties.is_liquid and asset_key in self.state.portfolio:
                    current_value = self.state.portfolio[asset_key]
                    fee_amount = current_value * monthly_fee_percentage
                    self.state.portfolio[asset_key] = current_value - fee_amount

    def _rebalance_if_needed(self, month):
        """
        Rebalance liquid assets (stocks, bonds, str, fun) according to the current portfolio weights,
        if a rebalance is scheduled for this year and this is the first month of the year.
        Real estate is not included in rebalancing.
        Also updates current_target_portfolio_weights if a rebalance occurs.
        """
        current_year = month // 12
        month_in_year = month % 12

        # Check if a rebalance is scheduled for this year and this is the first month
        scheduled_rebalance = None
        for reb in self.portfolio_rebalances:
            if reb.year == current_year and month_in_year == 0:
                scheduled_rebalance = reb
                break

        if scheduled_rebalance is not None:
            # Build a complete weights dict: missing keys get 0.0
            all_liquid_assets = [k for k, v in self.assets.items() if v.is_liquid]
            new_weights = {
                k: scheduled_rebalance.weights.get(k, 0.0) for k in all_liquid_assets
            }
            self.state.current_target_portfolio_weights = new_weights

            # Rebalance liquid assets
            self._rebalance_liquid_assets()

    def _rebalance_liquid_assets(self):
        """
        Rebalances all liquid assets according to the current target portfolio weights.
        """
        weights = self.state.current_target_portfolio_weights
        liquid_asset_keys = weights.keys()

        total_liquid = sum(
            self.state.portfolio.get(asset, 0.0) for asset in liquid_asset_keys
        )

        if total_liquid > 0:
            # Assuming weights sum to 1.0 as validated in config parsing
            for asset, weight in weights.items():
                self.state.portfolio[asset] = total_liquid * weight
        else:
            # If total liquid is zero, zero out all liquid assets
            for asset in liquid_asset_keys:
                if asset in self.state.portfolio:
                    self.state.portfolio[asset] = 0.0

    def _invest_in_liquid_assets(self, amount: float):
        """
        Invests a given amount into liquid assets according to current target weights.
        """
        weights = self.state.current_target_portfolio_weights
        for asset, weight in weights.items():
            self.state.portfolio[asset] = (
                self.state.portfolio.get(asset, 0.0) + amount * weight
            )

    def _withdraw_from_assets(self, amount: float) -> None:
        """
        Withdraws from liquid assets based on their configured withdrawal priority
        to cover a shortfall. If assets are insufficient, marks the simulation as failed.
        The withdrawn amount is added to the bank balance.
        """
        amount_needed = amount
        withdrawn_total = 0.0

        # Dynamically determine withdrawal order from config
        liquid_assets_with_priority = [
            (name, asset.withdrawal_priority)
            for name, asset in self.assets.items()
            if asset.is_liquid and asset.withdrawal_priority is not None
        ]
        # Sort by priority, lowest first
        withdrawal_order = sorted(liquid_assets_with_priority, key=lambda x: x[1])

        for asset_name, _ in withdrawal_order:
            if amount_needed <= 1e-9:  # Effectively zero
                break

            asset_value = self.state.portfolio.get(asset_name, 0.0)
            withdrawal_from_this_asset = min(amount_needed, asset_value)

            self.state.portfolio[asset_name] -= withdrawal_from_this_asset
            withdrawn_total += withdrawal_from_this_asset
            amount_needed -= withdrawal_from_this_asset

        # Add the total withdrawn amount to the bank balance
        self.state.current_bank_balance += withdrawn_total

        # If we couldn't withdraw the full amount, fail the simulation
        if amount_needed > 1e-9:  # Use a small tolerance for floating point
            self.state.simulation_failed = True

    def _record_results(self, month):
        """
        Record the current state of the simulation at the end of the month.
        This includes nominal wealth, bank balance, and all asset values.
        """
        # Record total wealth
        self.results["wealth_history"][month] = self.state.current_bank_balance + sum(
            self.state.portfolio.values()
        )
        # Record bank balance
        self.results["bank_balance_history"][month] = self.state.current_bank_balance

        # Record each asset's value
        for asset_name, value in self.state.portfolio.items():
            self.results[f"{asset_name}_history"][month] = value

    def _build_result(self) -> Dict[str, Any]:
        """
        Return the final simulation results as a dict (result structure).
        Truncates all history arrays to months_lasted to avoid None values.
        For failed simulations, histories are truncated at the failure point (no padding).
        """
        total_months = self.simulation_months
        months_lasted = next(
            (i for i, v in enumerate(self.results["wealth_history"]) if v is None),
            total_months,
        )

        def trunc_only(arr):
            return arr[:months_lasted]

        asset_keys = self.det_inputs.initial_portfolio.keys()

        if months_lasted > 0:
            last_month_idx = months_lasted - 1
            final_nominal_wealth = self.results["wealth_history"][last_month_idx]
            final_cumulative_inflation = (
                self.state.monthly_cumulative_inflation_factors[last_month_idx]
            )
            final_bank_balance = self.results["bank_balance_history"][last_month_idx]
            final_allocations_nominal = {
                key: self.results[f"{key}_history"][last_month_idx]
                for key in asset_keys
            }
        else:  # months_lasted == 0
            final_nominal_wealth = self.state.initial_total_wealth
            final_cumulative_inflation = 1.0
            final_bank_balance = self.det_inputs.initial_bank_balance
            final_allocations_nominal = self.det_inputs.initial_portfolio

        final_investment = final_nominal_wealth - final_bank_balance
        final_real_wealth = final_nominal_wealth / final_cumulative_inflation

        final_allocations_real = {
            k: float(v / final_cumulative_inflation)
            for k, v in final_allocations_nominal.items()
        }

        result = {
            # --- Scalars first ---
            "success": not self.state.simulation_failed,
            "months_lasted": months_lasted,
            "final_investment": final_investment,
            "final_bank_balance": final_bank_balance,
            "final_cumulative_inflation_factor": final_cumulative_inflation,
            "final_nominal_wealth": final_nominal_wealth,
            "final_real_wealth": final_real_wealth,
            # --- Non-state, derived or input data ---
            "final_allocations_nominal": final_allocations_nominal,
            "final_allocations_real": final_allocations_real,
            "initial_total_wealth": self.state.initial_total_wealth,
            # --- State and histories ---
            "monthly_returns_sequences": self.state.monthly_returns_sequences,
            "monthly_cumulative_inflation_factors": trunc_only(
                self.state.monthly_cumulative_inflation_factors
            ),
            "wealth_history": trunc_only(self.results["wealth_history"]),
            "bank_balance_history": trunc_only(self.results["bank_balance_history"]),
        }

        # Add all asset histories dynamically
        for key in asset_keys:
            result[f"{key}_history"] = trunc_only(self.results[f"{key}_history"])

        return result
