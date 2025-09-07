#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
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
- Supports scheduled portfolio rebalances
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
)

import numpy as np

# Import the actual types from config.py
from firecast.config.config import (
    DeterministicInputs,
    PortfolioRebalance,
    Asset,
    Shock,
    SimulationParameters,
)

from firecast.config.correlation_matrix import CorrelationMatrix
from firecast.core.sequences_generator import SequencesGenerator
from firecast.core.simulation_state import SimulationState


def calculate_transaction_fee(amount: float, fee_cfg: dict | None) -> float:
    """
    Calculates the transaction fee for a given amount using the config dict.
    Returns 0.0 if fee_cfg is None.
    """
    if fee_cfg is None:
        return 0.0
    min_fee = float(fee_cfg.get("min", 0.0))
    rate = float(fee_cfg.get("rate", 0.0))
    max_fee = float(fee_cfg.get("max", 0.0))
    fee = max(min_fee, amount * rate)
    if max_fee > 0.0:
        fee = min(fee, max_fee)
    return fee


class SimulationBuilder:
    """
    Builder class for constructing a Simulation instance with validated configuration.

    Use the fluent interface to set all required components, then call build().
    """

    def __init__(self):
        self.det_inputs: Optional[DeterministicInputs] = None
        self.assets: Optional[dict[str, Asset]] = None
        self.correlation_matrix: Optional[Any] = None
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

    def set_correlation_matrix(self, correlation_matrix):
        self.correlation_matrix = correlation_matrix
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

        if self.correlation_matrix is None:
            raise ValueError(
                "correlation_matrix must be set before building the simulation."
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
            self.correlation_matrix,
            self.portfolio_rebalances,
            self.shock_events,
            self.sim_params,
        )


class Simulation:
    """
    Core simulation engine for FIRE scenarios.

    Handles monthly flows, asset evolution, rebalancing, contributions, expenses,
    and records detailed histories. Marks simulation as failed if withdrawals
    cannot be covered by liquid assets.
    """

    def __init__(
        self,
        det_inputs: DeterministicInputs,
        assets: dict[str, Asset],
        correlation_matrix: CorrelationMatrix,
        portfolio_rebalances: list[PortfolioRebalance],
        shock_events: list[Shock],
        sim_params: SimulationParameters,
    ):
        self.det_inputs: DeterministicInputs = det_inputs
        self.assets: dict[str, Asset] = assets
        self.correlation_matrix = correlation_matrix
        self.portfolio_rebalances: list[PortfolioRebalance] = portfolio_rebalances
        self.shock_events: list[Shock] = shock_events
        self.sim_params: SimulationParameters = sim_params
        self.state: SimulationState = SimulationState(
            current_bank_balance=0.0,
            portfolio={},
            current_target_portfolio_weights={},
            initial_total_wealth=0.0,
            simulation_failed=False,
        )  # I t will be properly initialized in self.initialize_state()
        self.results: Dict[str, Any] = {}

    @property
    def simulation_months(self):
        """
        Total number of months to simulate, based on years_to_simulate.
        """
        return self.det_inputs.years_to_simulate * 12

    def _initialize_state(self):
        """
        Initialize all state variables for the simulation.

        This method also applies planned contributions at year 0 by calling
        self._handle_contributions(0), allocating them according to the target
        portfolio weights specified by the mandatory rebalance at year 0.

        Returns:
            SimulationState: Dataclass holding the simulation state.
        """

        # Initialize portfolio with zeros for all assets
        initial_portfolio = {k: 0.0 for k in self.assets.keys()}

        # Find the initial target portfolio weights from the rebalance at year 0
        initial_target_weights = next(
            reb.weights for reb in self.portfolio_rebalances if reb.year == 0
        )

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

        # Apply planned contribution at year 0 to set up initial allocation
        # This must be done here so it is not reapplied in simulation.run
        self.state = state
        self._handle_contributions(0)

        return self.state

    def init(self):
        """
        Initialize the simulation state, precompute sequences, and build rebalance
        schedule.
        """
        self.state = self._initialize_state()
        self._precompute_sequences()
        self._build_rebalance_schedule()

    def _build_rebalance_schedule(self):
        """
        Precompute a mapping from year to the rebalance event to apply, based on
        period logic.
        """
        # List of (year, PortfolioRebalance), sorted by year
        rebalances = sorted(self.portfolio_rebalances, key=lambda r: r.year)
        schedule: dict[int, PortfolioRebalance] = {}

        for idx, reb in enumerate(rebalances):
            start_year = reb.year
            period = reb.period
            # Determine the end year (exclusive) for this rebalance
            if idx + 1 < len(rebalances):
                next_year = rebalances[idx + 1].year
            else:
                next_year = self.simulation_months // 12 + 1  # End after simulation

            if period > 0:
                y = start_year
                while y < next_year:
                    schedule[y] = reb
                    y += period
            else:
                schedule[start_year] = reb

        self._rebalance_schedule = schedule  # year -> PortfolioRebalance

    def run(self) -> dict:
        """
        Run the main simulation loop, handling all monthly flows and events.

        Returns:
            dict: Final simulation results.
        """
        total_months = self.simulation_months
        total_months = self.simulation_months

        # Initialize results dictionary dynamically based on assets in the portfolio
        self.results = {
            "wealth_history": [None] * total_months,
            "bank_balance_history": [None] * total_months,
        }

        for asset_name in self.assets:
            self.results[f"{asset_name}_history"] = [None] * total_months

        for month in range(total_months):
            self.state.current_month_index = month
            self.state.current_year_index = month // 12

            # Contributions: Apply planned contributions to liquid assets.
            if month != 0:
                # To skip the contribution at year 0, alreaty applied in initialize_state
                # to set the initial allocation
                self._handle_contributions(month)

            # Illiquid asset purchases: transfer from liquid assets to illiquid
            self._handle_illiquid_purchases(month)

            # Income: Add income and pension for the current year.
            self._process_income(month)

            # Expenses: Deduct regular and extra expenses from the bank account.
            self._handle_expenses(month)

            # Bank Account Management:
            self._handle_bank_account(month)
            if self.state.simulation_failed:
                break  # Exit if bank top-up failed

            # Apply Fund Fee (monthly)
            self._apply_fund_fee()

            # Returns: Apply monthly returns to all assets.
            self._apply_monthly_returns(month)

            # Rebalancing: If scheduled, rebalance liquid assets.
            self._rebalance_if_needed(month)

            # Recording: Save the current state.
            self._record_results(month)

        return self._build_result()

    def _precompute_sequences(self):
        """
        Precompute all monthly sequences needed for the simulation.

        Uses SequenceGenerator to create correlated returns and inflation for a
        single run.
        """
        det_inputs = self.det_inputs
        shock_events = self.shock_events
        total_years = det_inputs.years_to_simulate
        total_months = total_years * 12

        # --- Generate Correlated Sequences using the Generator ---
        generator = SequencesGenerator(
            assets=self.assets,
            correlation_matrix=self.correlation_matrix,
            num_sequences=1,  # A single simulation run is one sequence
            simulation_years=total_years,
            seed=self.sim_params.random_seed,
        )
        # Squeeze to remove the num_sequences dimension
        # (shape: [1, months, assets] -> [months, assets])
        return_rates_array = np.squeeze(generator.monthly_return_rates, axis=0)

        # --- Convert array to dictionary for use in the simulation ---
        self.state.monthly_return_rates_sequences = {
            asset: return_rates_array[:, i]
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

                    if shock_asset in self.state.monthly_return_rates_sequences:
                        target_sequence = self.state.monthly_return_rates_sequences[
                            shock_asset
                        ]
                        for month_offset in range(12):
                            month_idx_in_simulation = year_idx * 12 + month_offset
                            if 0 <= month_idx_in_simulation < total_months:
                                target_sequence[month_idx_in_simulation] = (
                                    monthly_shock_rate
                                )

        monthly_inflation_sequence = self.state.monthly_return_rates_sequences[
            "inflation"
        ]

        # --- Cumulative inflation factors (monthly) ---
        monthly_cumulative_inflation_factors = np.ones(
            total_months + 1, dtype=np.float64
        )
        for month_idx in range(total_months):
            monthly_cumulative_inflation_factors[month_idx + 1] = (
                monthly_cumulative_inflation_factors[month_idx]
                * (1.0 + monthly_inflation_sequence[month_idx])
            )

        # --- Precompute nominal pension and income monthly sequences with partial indexation ---
        monthly_nominal_pension_sequence = np.zeros(total_months, dtype=np.float64)
        monthly_nominal_income_sequence = np.zeros(total_months, dtype=np.float64)
        monthly_nominal_expenses_sequence = np.zeros(total_months, dtype=np.float64)

        # --- Income steps logic ---
        income_steps = sorted(det_inputs.monthly_income_steps, key=lambda s: s.year)
        income_inflation_factor = det_inputs.income_inflation_factor
        income_end_month_idx = det_inputs.income_end_year * 12

        monthly_nominal_income_sequence = np.zeros(total_months, dtype=np.float64)

        if not income_steps:
            monthly_nominal_income_sequence[:] = 0.0
        else:
            # Build a list of (start_month, monthly_amount) for each step
            income_step_months = [
                (step.year * 12, step.monthly_amount) for step in income_steps
            ]
            step_inflated_amounts = []
            for step_start_month, step_real_amount in income_step_months:
                inflation_factor = monthly_cumulative_inflation_factors[
                    step_start_month
                ]
                step_inflated_amounts.append(step_real_amount * inflation_factor)

            # For all steps except the last, fill with constant value
            for idx in range(len(income_step_months) - 1):
                start = income_step_months[idx][0]
                end = income_step_months[idx + 1][0]
                monthly_nominal_income_sequence[start:end] = step_inflated_amounts[idx]

            # For the last step: grow with inflation and income_inflation_factor
            last_start = income_step_months[-1][0]
            last_income = step_inflated_amounts[-1]
            for month in range(last_start, min(income_end_month_idx, total_months)):
                if month == last_start:
                    monthly_nominal_income_sequence[month] = last_income
                else:
                    prev = monthly_nominal_income_sequence[month - 1]
                    monthly_nominal_income_sequence[month] = prev * (
                        1.0
                        + monthly_inflation_sequence[month - 1]
                        * income_inflation_factor
                    )

            # After income_end_year: zero
            if income_end_month_idx < total_months:
                monthly_nominal_income_sequence[income_end_month_idx:] = 0.0

        # --- Pension logic ---
        if det_inputs.pension_start_year is not None and det_inputs.monthly_pension > 0:
            pension_start_month_idx = det_inputs.pension_start_year * 12
            pension_cumulative = det_inputs.monthly_pension
            for month_idx in range(total_months):
                if month_idx >= pension_start_month_idx:
                    if month_idx == pension_start_month_idx:
                        pension_cumulative = det_inputs.monthly_pension
                    else:
                        pension_cumulative *= 1.0 + (
                            monthly_inflation_sequence[month_idx - 1]
                            * det_inputs.pension_inflation_factor
                        )
                    monthly_nominal_pension_sequence[month_idx] = pension_cumulative

        # --- Expenses steps logic (inflation-adjusted within each step) ---
        expense_steps = sorted(det_inputs.monthly_expenses_steps, key=lambda s: s.year)
        if not expense_steps:
            monthly_nominal_expenses_sequence[:] = 0.0
        else:
            # Build a list of (start_month, monthly_amount) for each step
            expense_step_months = [
                (step.year * 12, step.monthly_amount) for step in expense_steps
            ]
            step_ranges = []
            for idx in range(len(expense_step_months)):
                start = expense_step_months[idx][0]
                if idx + 1 < len(expense_step_months):
                    end = expense_step_months[idx + 1][0]
                else:
                    end = total_months
                step_ranges.append((start, end, expense_step_months[idx][1]))

            for start, end, base_amount in step_ranges:
                base_inflation = monthly_cumulative_inflation_factors[start]
                for month in range(start, min(end, total_months)):
                    inflation = monthly_cumulative_inflation_factors[month]
                    monthly_nominal_expenses_sequence[month] = base_amount * (
                        inflation / base_inflation
                    )

        self.state.monthly_cumulative_inflation_factors = (
            monthly_cumulative_inflation_factors
        )
        self.state.monthly_nominal_pension_sequence = monthly_nominal_pension_sequence
        self.state.monthly_nominal_income_sequence = monthly_nominal_income_sequence
        self.state.monthly_nominal_expenses_sequence = monthly_nominal_expenses_sequence

    def _process_income(self, month):
        """
        Add the precomputed monthly income and pension for the current month.
        """
        income = 0.0

        # Pension (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state.monthly_nominal_pension_sequence):
            income += self.state.monthly_nominal_pension_sequence[month]

        # Income (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state.monthly_nominal_income_sequence):
            income += self.state.monthly_nominal_income_sequence[month]

        self.state.current_bank_balance += float(income)

    def _handle_contributions(self, month):
        """
        Handles planned one-time contributions.
        Each contribution is allocated directly to the specified asset if given,
        otherwise according to current portfolio weights.
        For liquid assets, transaction fee is applied.
        """
        det_inputs = self.det_inputs

        if month % 12 == 0:
            current_year = month // 12
            for contribution in det_inputs.planned_contributions:
                if contribution.year == current_year:
                    amount = contribution.amount
                    asset = contribution.asset
                    if asset is not None:
                        if self.assets[asset].withdrawal_priority is not None:
                            fee_cfg = self.det_inputs.transactions_fee
                            fee = calculate_transaction_fee(amount, fee_cfg)
                            net_amount = amount - fee
                            self.state.portfolio[asset] += net_amount
                        else:
                            self.state.portfolio[asset] += amount
                    else:
                        self._invest_in_liquid_assets(amount)

    def _handle_illiquid_purchases(self, month):
        """
        Handle planned purchases of illiquid assets from liquid assets for the given
        month.
        Withdraws the inflation-adjusted amount from liquid assets and allocates it
        to the illiquid asset.
        """
        det_inputs = self.det_inputs

        if month % 12 == 0:
            current_year = month // 12
            for purchase in det_inputs.planned_illiquid_purchases:
                if purchase.year == current_year:
                    inflation_factor = self.state.monthly_cumulative_inflation_factors[
                        month
                    ]
                    nominal_amount = float(purchase.amount * inflation_factor)
                    self._withdraw_from_assets(nominal_amount)
                    self.state.current_bank_balance -= nominal_amount
                    self.state.portfolio[purchase.asset] += nominal_amount

    def _handle_expenses(self, month):
        """
        Deducts regular monthly expenses and planned extra expenses from the bank balance.
        Expenses are inflation-adjusted.
        Planned extra expenses are applied at the first month of their year.
        """
        det_inputs = self.det_inputs

        # Regular monthly expenses (inflation-adjusted, precomputed)
        nominal_monthly_expenses = 0.0
        if month < len(self.state.monthly_nominal_expenses_sequence):
            nominal_monthly_expenses = self.state.monthly_nominal_expenses_sequence[
                month
            ]
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
            lot_size = self.det_inputs.investment_lot_size
            if lot_size > 0.0:
                invest_amount = (excess // lot_size) * lot_size
            else:
                invest_amount = excess
            if invest_amount > 0.0:
                self._invest_in_liquid_assets(float(invest_amount))
                self.state.current_bank_balance = float(
                    self.state.current_bank_balance - invest_amount
                )
            # Clamp to upper bound if all excess was invested
            if self.state.current_bank_balance > upper and lot_size == 0.0:
                self.state.current_bank_balance = float(upper)

    def _apply_monthly_returns(self, month):
        """
        Apply monthly returns to all asset values at the end of the month.
        """
        returns = self.state.monthly_return_rates_sequences
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

            for asset_key in self.assets:
                if self.assets[asset_key].withdrawal_priority is None:
                    continue
                current_value = self.state.portfolio[asset_key]
                fee_amount = current_value * monthly_fee_percentage
                self.state.portfolio[asset_key] = current_value - fee_amount

    def _rebalance_if_needed(self, month):
        """
        Rebalance liquid assets according to the current
        portfolio weights, if a rebalance is scheduled for this year and this is
        the first month of the year.
        Also updates current_target_portfolio_weights if a rebalance occurs.
        """
        current_year = month // 12
        month_in_year = month % 12

        # Use the precomputed rebalance schedule
        scheduled_rebalance = None
        if month_in_year == 0:
            scheduled_rebalance = self._rebalance_schedule.get(current_year, None)

        if scheduled_rebalance is not None:
            # Only include assets with withdrawal_priority != None
            all_liquid_assets = [
                k
                for k, asset in self.assets.items()
                if asset.withdrawal_priority is not None
            ]
            # If a liquid asset is not included in the rebalance it gets weight = 0.
            new_weights = {
                k: scheduled_rebalance.weights.get(k, 0.0) for k in all_liquid_assets
            }
            self.state.current_target_portfolio_weights = new_weights

            # Rebalance assets
            self._rebalance_liquid_assets()

    def _rebalance_liquid_assets(self):
        """
        Rebalances all liquid assets according to the current target portfolio weights.
        Applies transaction fee to each buy/sell operation.
        """
        weights = self.state.current_target_portfolio_weights
        liquid_asset_keys = [
            k
            for k, asset in self.assets.items()
            if asset.withdrawal_priority is not None
        ]

        total_liquid = sum(self.state.portfolio[k] for k in liquid_asset_keys)
        fee_cfg = self.det_inputs.transactions_fee

        if total_liquid > 0:
            for asset in liquid_asset_keys:
                target_value = total_liquid * weights.get(asset, 0.0)
                current_value = self.state.portfolio[asset]
                delta = target_value - current_value

                if abs(delta) < 1e-9:
                    continue

                if delta > 0:
                    # Buy: pay fee from bank, invest net amount
                    fee = calculate_transaction_fee(delta, fee_cfg)
                    net_invest = delta - fee
                    self.state.current_bank_balance -= delta
                    self.state.portfolio[asset] += net_invest
                else:
                    # Sell: pay fee from proceeds, add net to bank
                    gross_sell = -delta
                    fee = calculate_transaction_fee(gross_sell, fee_cfg)
                    net_proceeds = gross_sell - fee
                    self.state.portfolio[asset] -= gross_sell
                    self.state.current_bank_balance += net_proceeds
        else:
            # If total liquid is zero, zero out all liquid assets
            for asset in liquid_asset_keys:
                self.state.portfolio[asset] = 0.0

    def _invest_in_liquid_assets(self, amount: float):
        """
        Invests a given amount into liquid assets according to current target weights.
        Applies transaction fee if configured.
        """
        fee_cfg = self.det_inputs.transactions_fee
        fee = calculate_transaction_fee(amount, fee_cfg)
        net_amount = amount - fee
        weights = self.state.current_target_portfolio_weights
        for asset, weight in weights.items():
            if self.assets[asset].withdrawal_priority is not None:
                self.state.portfolio[asset] += net_amount * weight

    def _withdraw_from_assets(self, amount: float) -> None:
        """
        Withdraw the specified amount from liquid assets in withdrawal priority order.
        Applies transaction fee if configured.
        """
        amount_needed = amount
        withdrawn_total = 0.0

        liquid_assets_with_priority = [
            (name, asset.withdrawal_priority)
            for name, asset in self.assets.items()
            if asset.withdrawal_priority is not None
        ]
        withdrawal_order = sorted(liquid_assets_with_priority, key=lambda x: x[1])
        fee_cfg = self.det_inputs.transactions_fee

        for asset_name, _ in withdrawal_order:
            if amount_needed <= 1e-9:
                break

            asset_value = self.state.portfolio.get(asset_name, 0.0)
            if asset_value <= 1e-9:
                continue

            # Binary search for gross_withdrawal so that net = amount_needed
            left, right = 0.0, asset_value
            best_gross = 0.0
            epsilon = 1e-8
            while right - left > epsilon:
                mid = (left + right) / 2
                fee = calculate_transaction_fee(mid, fee_cfg)
                net = mid - fee
                if net >= amount_needed:
                    best_gross = mid
                    right = mid
                else:
                    left = mid
            # If best_gross was never updated, withdraw all available from the asset
            if best_gross == 0.0 and amount_needed > 1e-9:
                gross_withdrawal = asset_value
            else:
                gross_withdrawal = min(best_gross, asset_value)
            fee = calculate_transaction_fee(gross_withdrawal, fee_cfg)
            net = gross_withdrawal - fee

            self.state.portfolio[asset_name] -= gross_withdrawal
            withdrawn_total += net
            amount_needed -= net

        self.state.current_bank_balance += withdrawn_total
        if amount_needed > 1e-9:
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
        for asset_name in self.assets:
            value = self.state.portfolio.get(asset_name, 0.0)
            self.results[f"{asset_name}_history"][month] = value

    def _build_result(self) -> Dict[str, Any]:
        """
        Return the final simulation results as a dict (result structure).
        Truncates all history arrays to months_lasted to avoid None values.
        For failed simulations, histories are truncated at the failure point (no padding).

        Returns:
            Dict[str, Any]: Final simulation results.
        """
        total_months = self.simulation_months
        months_lasted = next(
            (i for i, v in enumerate(self.results["wealth_history"]) if v is None),
            total_months,
        )

        def trunc_only(arr):
            return arr[:months_lasted]

        asset_keys = self.assets.keys()

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
            final_allocations_nominal = {k: 0.0 for k in asset_keys}

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
            "monthly_returns_sequences": self.state.monthly_return_rates_sequences,
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
