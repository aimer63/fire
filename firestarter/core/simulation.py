# flake8: noqa=F821
"""
FIRE simulation engine for modeling financial independence and early retirement scenarios.

This module provides the Simulation and SimulationBuilder classes to run customizable
financial simulations. It supports configurable inputs for income, expenses, asset allocation,
portfolio rebalancing, planned contributions, extra expenses, house purchases, and economic assumptions.

Features:
- Modular builder pattern for flexible simulation setup.
- Handles monthly flows: income, contributions, expenses, withdrawals, and asset rebalancing.
- Evolves asset values monthly according to stochastic returns and inflation.
- Supports planned shocks and house purchases.
- Records detailed monthly histories of wealth, balances, and asset values.
- Marks simulation as failed if withdrawals cannot be covered by liquid assets.
"""

from typing import (
    Dict,
    Any,
    Optional,
)  # Removed List as Shocks class handles list of events
from firestarter.core.constants import ASSET_KEYS, WITHDRAWAL_PRIORITY

# Import the actual types from config.py
from firestarter.config.config import (
    DeterministicInputs,
    MarketAssumptions,
    PortfolioRebalances,
    Shocks,
    SimulationParameters,
)


class SimulationBuilder:
    def __init__(self):
        self.det_inputs: Optional[DeterministicInputs] = None
        self.market_assumptions: Optional[MarketAssumptions] = None
        self.portfolio_rebalances: Optional[PortfolioRebalances] = None
        self.shock_events: Optional[Shocks] = None
        self.initial_assets: Optional[Dict[str, float]] = None
        self.sim_params: Optional[SimulationParameters] = None

    @classmethod
    def new(cls):
        return cls()

    def set_det_inputs(self, det_inputs):
        self.det_inputs = det_inputs
        return self

    def set_market_assumptions(self, market_assumptions):
        self.market_assumptions = market_assumptions
        return self

    def set_portfolio_rebalances(self, portfolio_rebalances):
        self.portfolio_rebalances = portfolio_rebalances
        return self

    def set_shock_events(self, shock_events):
        self.shock_events = shock_events
        return self

    def set_initial_assets(self, initial_assets):
        self.initial_assets = initial_assets
        return self

    def set_sim_params(self, sim_params):
        self.sim_params = sim_params
        return self

    def build(self):
        # Validate all required fields are set
        if self.det_inputs is None:
            raise ValueError("det_inputs must be set before building the simulation.")
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
        if self.initial_assets is None:  # Added check for initial_assets
            raise ValueError(
                "initial_assets must be set before building the simulation."
            )
        if self.sim_params is None:
            raise ValueError(
                "sim_params (SimulationParameters) must be set before building the simulation."
            )

        return Simulation(
            self.det_inputs,
            self.market_assumptions,
            self.portfolio_rebalances,
            self.shock_events,
            self.initial_assets,
            self.sim_params,
        )


class Simulation:
    def __init__(
        self,
        det_inputs: DeterministicInputs,
        market_assumptions: MarketAssumptions,
        portfolio_rebalances: PortfolioRebalances,
        shock_events: Shocks,
        initial_assets: Dict[str, float],
        sim_params: SimulationParameters,
    ):
        self.det_inputs: DeterministicInputs = det_inputs
        self.market_assumptions: MarketAssumptions = market_assumptions
        self.portfolio_rebalances: PortfolioRebalances = portfolio_rebalances
        self.shock_events: Shocks = shock_events
        self.initial_assets: Dict[str, float] = initial_assets
        self.sim_params: SimulationParameters = sim_params
        self.state: Dict[str, Any] = {}
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
        self.init()  # Initialize state and precompute sequences
        total_months = self.det_inputs.years_to_simulate * 12

        for month in range(total_months):
            self.state["current_month_index"] = month
            self.state["current_year_index"] = month // 12

            # 1. Income: Add salary and pension for the current year.
            self._process_income(month)

            # 2. Contributions: Apply planned and regular contributions to liquid assets.
            self._handle_contributions(month)

            # 3. Expenses: Deduct regular and extra expenses from the bank account.
            self._handle_expenses(month)

            # 4. House Purchase: If scheduled, withdraw from assets to buy a house.
            self._handle_house_purchase(month)
            if self.state["simulation_failed"]:
                break  # Exit if house purchase failed

            # 5. Bank Account Management:
            self._handle_bank_account(month)
            if self.state["simulation_failed"]:
                break  # Exit if bank top-up failed

            # 6. Returns: Apply monthly returns to all assets.
            self._apply_monthly_returns(month)

            # 7. Apply Fund Fee (monthly)
            self._apply_fund_fee(month)

            # 8. Rebalancing: If scheduled, rebalance liquid assets.
            self._rebalance_if_needed(month)

            # 9. Recording: Save the current state.
            self._record_results(month)

        return self._build_result()

    # --- Helper methods (stubs for now) ---
    def _initialize_state(self):
        """
        Initialize all state variables for the simulation.
        Returns a dictionary or custom object holding the simulation state.
        """
        # Find the initial target portfolio weights (from the first rebalance)
        first_reb = self.portfolio_rebalances.rebalances[0]
        initial_target_weights = {
            asset: getattr(first_reb, asset)
            for asset in ASSET_KEYS
            if asset != "real_estate"
        }

        state = {
            "current_bank_balance": self.det_inputs.initial_bank_balance,
            "liquid_assets": {
                asset: self.initial_assets[asset]
                for asset in ASSET_KEYS
                if asset != "real_estate"
            },
            "current_real_estate_value": self.initial_assets["real_estate"],
            "current_target_portfolio_weights": initial_target_weights,
            # Optionally add more state variables as needed
        }
        self.state = state
        self.state["initial_total_wealth"] = (
            self.state["current_bank_balance"]
            + sum(self.state["liquid_assets"].values())
            + self.state["current_real_estate_value"]
        )
        self.state["simulation_failed"] = False
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
        This version draws returns and inflation for each month, not just each year.
        Correctly applies pension_inflation_factor and salary_inflation_factor.
        """
        import numpy as np

        # Set the random seed for reproducibility if provided
        if self.sim_params.random_seed is not None:
            np.random.seed(self.sim_params.random_seed)
        # If random_seed is None, NumPy's RNG will be seeded from an entropy source
        # by default (or continue with its current state if already initialized).

        det_inputs = self.det_inputs
        market_assumptions = self.market_assumptions
        shock_events = self.shock_events

        lognormal = market_assumptions.lognormal
        total_years = det_inputs.years_to_simulate
        total_months = total_years * 12

        # --- Convert annual lognormal parameters to monthly ---
        mu_log_stocks, sigma_log_stocks = self.annual_lognormal_to_monthly(
            *lognormal["stocks"]
        )
        mu_log_bonds, sigma_log_bonds = self.annual_lognormal_to_monthly(
            *lognormal["bonds"]
        )
        mu_log_str, sigma_log_str = self.annual_lognormal_to_monthly(*lognormal["str"])
        mu_log_fun, sigma_log_fun = self.annual_lognormal_to_monthly(*lognormal["fun"])
        mu_log_real_estate, sigma_log_real_estate = self.annual_lognormal_to_monthly(
            *lognormal["real_estate"]
        )
        mu_log_inflation, sigma_log_inflation = self.annual_lognormal_to_monthly(
            *lognormal["inflation"]
        )

        # --- Draw monthly inflation and returns ---
        monthly_inflations_sequence = (
            np.random.lognormal(
                mu_log_inflation, sigma_log_inflation, total_months
            ).astype(np.float64)
            - 1.0
        )
        monthly_stocks_returns_sequence = (
            np.random.lognormal(mu_log_stocks, sigma_log_stocks, total_months).astype(
                np.float64
            )
            - 1.0
        )
        monthly_bonds_returns_sequence = (
            np.random.lognormal(mu_log_bonds, sigma_log_bonds, total_months).astype(
                np.float64
            )
            - 1.0
        )
        monthly_str_returns_sequence = (
            np.random.lognormal(mu_log_str, sigma_log_str, total_months).astype(
                np.float64
            )
            - 1.0
        )
        monthly_fun_returns_sequence = (
            np.random.lognormal(mu_log_fun, sigma_log_fun, total_months).astype(
                np.float64
            )
            - 1.0
        )
        monthly_real_estate_returns_sequence = (
            np.random.lognormal(
                mu_log_real_estate, sigma_log_real_estate, total_months
            ).astype(np.float64)
            - 1.0
        )

        # --- Apply shocks (if any) ---
        # A shock's magnitude is an annual rate that replaces the stochastic rate
        # for that year.
        # This annual rate is then converted to a monthly rate and applied to all 12
        # months of that year.
        for shock in shock_events.events:  # Iterate over the .events attribute
            year_idx = shock.year
            shock_asset = shock.asset
            annual_shock_rate = shock.magnitude  # This is an annual rate

            if 0 <= year_idx < total_years:
                # Convert the annual shock rate to an equivalent monthly rate
                monthly_shock_rate = (1.0 + annual_shock_rate) ** (1.0 / 12.0) - 1.0

                target_sequence = None
                if shock_asset == "Stocks":
                    target_sequence = monthly_stocks_returns_sequence
                elif shock_asset == "Bonds":
                    target_sequence = monthly_bonds_returns_sequence
                elif shock_asset == "STR":
                    target_sequence = monthly_str_returns_sequence
                elif shock_asset == "Fun":
                    target_sequence = monthly_fun_returns_sequence
                elif shock_asset == "Real Estate":
                    target_sequence = monthly_real_estate_returns_sequence
                elif shock_asset == "Inflation":
                    target_sequence = monthly_inflations_sequence

                if target_sequence is not None:
                    for month_offset in range(12):
                        month_idx_in_simulation = year_idx * 12 + month_offset
                        if 0 <= month_idx_in_simulation < total_months:
                            target_sequence[month_idx_in_simulation] = (
                                monthly_shock_rate
                            )

        # --- Cumulative inflation factors (monthly) ---
        monthly_cumulative_inflation_factors = np.ones(
            total_months + 1, dtype=np.float64
        )
        for month_idx in range(total_months):
            monthly_cumulative_inflation_factors[month_idx + 1] = (
                monthly_cumulative_inflation_factors[month_idx]
                * (1.0 + monthly_inflations_sequence[month_idx])
            )

        # --- Monthly returns lookup ---
        monthly_returns_lookup = {
            "Stocks": monthly_stocks_returns_sequence,
            "Bonds": monthly_bonds_returns_sequence,
            "STR": monthly_str_returns_sequence,
            "Fun": monthly_fun_returns_sequence,
            "Real Estate": monthly_real_estate_returns_sequence,
        }

        # --- Planned contributions and extra expenses (nominal, inflation-adjusted) ---
        planned_contributions = det_inputs.planned_contributions
        planned_extra_expenses = det_inputs.planned_extra_expenses

        nominal_planned_contributions_amounts = []
        for contribution in planned_contributions:
            # Use cumulative inflation up to the first month of the year
            month_idx = contribution.year * 12
            nominal_contribution_amount = float(
                contribution.amount * monthly_cumulative_inflation_factors[month_idx]
            )
            nominal_planned_contributions_amounts.append(
                (nominal_contribution_amount, contribution.year)
            )

        nominal_planned_extra_expenses_amounts = []
        for expense in planned_extra_expenses:
            month_idx = expense.year * 12
            nominal_extra_expense_amount = float(
                expense.amount * monthly_cumulative_inflation_factors[month_idx]
            )
            nominal_planned_extra_expenses_amounts.append(
                (nominal_extra_expense_amount, expense.year)
            )

        # --- Precompute nominal pension and salary monthly sequences with partial indexation ---
        nominal_pension_monthly_sequence = np.zeros(total_months, dtype=np.float64)
        nominal_salary_monthly_sequence = np.zeros(total_months, dtype=np.float64)

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
                        monthly_inflations_sequence[month_idx - 1]
                        * det_inputs.pension_inflation_factor
                    )
                nominal_pension_monthly_sequence[month_idx] = pension_cumulative

            # Salary
            if salary_start_month_idx <= month_idx < salary_end_month_idx:
                if month_idx == salary_start_month_idx:
                    salary_cumulative = det_inputs.monthly_salary
                else:
                    salary_cumulative *= 1.0 + (
                        monthly_inflations_sequence[month_idx - 1]
                        * det_inputs.salary_inflation_factor
                    )
                nominal_salary_monthly_sequence[month_idx] = salary_cumulative

        # --- Store all sequences in self.state ---
        self.state["monthly_inflations_sequence"] = monthly_inflations_sequence
        self.state["monthly_stocks_returns_sequence"] = monthly_stocks_returns_sequence
        self.state["monthly_bonds_returns_sequence"] = monthly_bonds_returns_sequence
        self.state["monthly_str_returns_sequence"] = monthly_str_returns_sequence
        self.state["monthly_fun_returns_sequence"] = monthly_fun_returns_sequence
        self.state["monthly_real_estate_returns_sequence"] = (
            monthly_real_estate_returns_sequence
        )
        self.state["monthly_cumulative_inflation_factors"] = (
            monthly_cumulative_inflation_factors
        )
        self.state["monthly_returns_lookup"] = monthly_returns_lookup
        self.state["nominal_planned_contributions_amounts"] = (
            nominal_planned_contributions_amounts
        )
        self.state["nominal_planned_extra_expenses_amounts"] = (
            nominal_planned_extra_expenses_amounts
        )
        self.state["nominal_pension_monthly_sequence"] = (
            nominal_pension_monthly_sequence
        )
        self.state["nominal_salary_monthly_sequence"] = nominal_salary_monthly_sequence

    def _process_income(self, month):
        """
        For each month, add the precomputed *monthly* salary and pension for the current month.
        These values are now drawn and adjusted monthly.
        """
        income = 0.0

        # Pension (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state["nominal_pension_monthly_sequence"]):
            income += self.state["nominal_pension_monthly_sequence"][month]

        # Salary (precomputed, already inflation/adjustment adjusted)
        if month < len(self.state["nominal_salary_monthly_sequence"]):
            income += self.state["nominal_salary_monthly_sequence"][month]

        self.state["current_bank_balance"] += income

    def _handle_contributions(self, month):
        """
        Handles planned one-time contributions and regular monthly contributions.
        Planned contribution are all applied the first month of the year
        Contributions are allocated according to the current target portfolio weights,
        but NEVER to real estate (see real_estate.md).
        """
        det_inputs = self.det_inputs

        # Planned one-time contributions (applied at the first month of the year)
        for nominal_contribution_amount, year_idx in self.state[
            "nominal_planned_contributions_amounts"
        ]:
            if month == year_idx * 12:
                weights = self.state["current_target_portfolio_weights"]
                increments = {
                    asset: nominal_contribution_amount * weights[asset]
                    for asset in ASSET_KEYS
                    if asset != "real_estate"
                }
                for asset, delta in increments.items():
                    self.state["liquid_assets"][asset] += delta

        # Regular monthly contribution (inflation-adjusted)
        if det_inputs.monthly_investment_contribution > 0.0:
            monthly_contribution = (
                det_inputs.monthly_investment_contribution
                * self.state["monthly_cumulative_inflation_factors"][month]
            )
            weights = self.state["current_target_portfolio_weights"]
            increments = {
                asset: monthly_contribution * weights[asset]
                for asset in ASSET_KEYS
                if asset != "real_estate"
            }
            for asset, delta in increments.items():
                self.state["liquid_assets"][asset] += delta

    def _handle_expenses(self, month):
        """
        Deducts regular monthly expenses and planned extra expenses from the bank balance.
        Expenses are inflation-adjusted. Planned extra expenses are applied at the first month of their year.
        """
        det_inputs = self.det_inputs

        # Regular monthly expenses (inflation-adjusted)
        nominal_monthly_expenses = (
            det_inputs.monthly_expenses
            * self.state["monthly_cumulative_inflation_factors"][month]
        )
        total_expenses = nominal_monthly_expenses

        # Planned extra expenses (applied at the first month of the year)
        for nominal_extra_expense_amount, year_idx in self.state[
            "nominal_planned_extra_expenses_amounts"
        ]:
            if month == year_idx * 12:
                total_expenses += nominal_extra_expense_amount

        # Deduct from bank balance
        self.state["current_bank_balance"] -= total_expenses

    def _handle_house_purchase(self, month):
        """
        Handles the house purchase if scheduled for this month.
        Deducts the (inflation-adjusted) house cost from liquid assets (STR, Bonds, Stocks, Fun) in order,
        using the unified _withdraw_from_assets method.
        If assets are insufficient, marks the simulation as failed.
        Adds the house value to real estate holdings.
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
            cumulative_inflation = self.state["monthly_cumulative_inflation_factors"][
                month
            ]
            nominal_house_cost = house_cost_real * cumulative_inflation

            # Use the unified withdrawal logic, but do NOT increase bank balance
            old_bank_balance = self.state["current_bank_balance"]
            self._withdraw_from_assets(nominal_house_cost)
            if self.state["simulation_failed"]:
                return
            # Remove the artificial bank increase (since we don't want to increase bank, just pay for house)
            self.state["current_bank_balance"] = old_bank_balance

            # Add house value to real estate
            self.state["current_real_estate_value"] += nominal_house_cost

            # --- Rebalance remaining liquid assets according to current target portfolio weights ---
            total_liquid = sum(self.state["liquid_assets"].values())
            weights = self.state["current_target_portfolio_weights"]
            if total_liquid > 0:
                for asset, weight in weights.items():
                    self.state["liquid_assets"][asset] = total_liquid * weight
            else:
                # If total liquid is zero, zero out all liquid assets
                for asset in self.state["liquid_assets"]:
                    self.state["liquid_assets"][asset] = 0.0

    def _handle_bank_account(self, month):
        """
        Ensures the bank balance is within [lower_bound, upper_bound].
        - If below lower_bound, withdraw from assets to top up.
        - If above upper_bound, invest excess into assets.
        - If assets are insufficient for top-up, mark simulation as failed.

        The bounds are specified in real terms (today's money) and must be converted
        to nominal using the cumulative inflation factor for the current month.
        """
        cumulative_inflation = self.state["monthly_cumulative_inflation_factors"][month]
        lower = self.det_inputs.bank_lower_bound * cumulative_inflation
        upper = self.det_inputs.bank_upper_bound * cumulative_inflation

        # Top up if below lower bound
        if self.state["current_bank_balance"] < lower:
            shortfall = lower - self.state["current_bank_balance"]
            self._withdraw_from_assets(shortfall)
            # If withdrawal failed, _withdraw_from_assets sets simulation_failed
            if self.state["simulation_failed"]:
                return
            self.state["current_bank_balance"] = lower

        # Invest excess if above upper bound
        if self.state["current_bank_balance"] > upper:
            excess = self.state["current_bank_balance"] - upper
            weights = self.state["current_target_portfolio_weights"]
            increments = {
                asset: excess * weights[asset] for asset in self.state["liquid_assets"]
            }
            for asset, delta in increments.items():
                self.state["liquid_assets"][asset] += delta
            self.state["current_bank_balance"] = upper

    def _apply_monthly_returns(self, month):
        """
        Apply monthly returns to all asset values at the end of the month.
        """
        returns = self.state["monthly_returns_lookup"]
        asset_to_returns_key = {
            "stocks": "Stocks",
            "bonds": "Bonds",
            "str": "STR",
            "fun": "Fun",
        }
        for asset in ASSET_KEYS:
            if asset != "real_estate":
                self.state["liquid_assets"][asset] *= (
                    1.0 + returns[asset_to_returns_key[asset]][month]
                )
        self.state["current_real_estate_value"] *= 1.0 + returns["Real Estate"][month]

    def _apply_fund_fee(self, month: int) -> None:
        """
        Applies the fund fee to all liquid assets on a monthly basis.
        The annual fee is converted to a monthly equivalent.
        """
        annual_fee_percentage = self.det_inputs.annual_fund_fee
        if annual_fee_percentage > 0:
            # Convert annual fee to a simple monthly fee
            monthly_fee_percentage = annual_fee_percentage / 12.0

            for asset_key in self.state["liquid_assets"].keys():
                current_value = self.state["liquid_assets"][asset_key]
                fee_amount = current_value * monthly_fee_percentage
                self.state["liquid_assets"][asset_key] = current_value - fee_amount

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
        for reb in self.portfolio_rebalances.rebalances:
            if reb.year == current_year and month_in_year == 0:
                scheduled_rebalance = reb
                break

        if scheduled_rebalance is not None:
            # Update current_target_portfolio_weights in state
            self.state["current_target_portfolio_weights"] = {
                asset: getattr(scheduled_rebalance, asset)
                for asset in ASSET_KEYS
                if asset != "real_estate"
            }

            # Calculate total liquid assets
            total_liquid = sum(self.state["liquid_assets"].values())
            weights = self.state["current_target_portfolio_weights"]
            # Assuming weights sum to 1.0 as validated in config parsing
            if total_liquid > 0:
                for asset, weight in weights.items():
                    self.state["liquid_assets"][asset] = total_liquid * weight
            else:
                # If total liquid is zero, zero out all liquid assets
                for asset in self.state["liquid_assets"]:
                    self.state["liquid_assets"][asset] = 0.0

    def _withdraw_from_assets(self, amount: float) -> None:
        """
        Withdraws from liquid assets in priority order (STR, Bonds, Stocks, Fun)
        to cover a bank shortfall. If assets are insufficient, marks the simulation as failed.
        """
        shortfall = amount
        for asset in WITHDRAWAL_PRIORITY:
            asset_value = self.state["liquid_assets"][asset]
            if asset_value >= shortfall:
                self.state["liquid_assets"][asset] -= shortfall
                self.state["current_bank_balance"] += shortfall
                return
            else:
                self.state["current_bank_balance"] += asset_value
                shortfall -= asset_value
                self.state["liquid_assets"][asset] = 0.0

        # If still shortfall after all liquid assets, mark simulation as failed
        self.state["simulation_failed"] = True

    def _record_results(self, month):
        """
        Record the current state of the simulation at the end of the month.
        This includes nominal wealth, bank balance, and all asset values.
        """
        if self.results == {}:
            total_months = self.simulation_months
            self.results = {
                "wealth_history": [None] * total_months,
                "bank_balance_history": [None] * total_months,
                "stocks_history": [None] * total_months,
                "bonds_history": [None] * total_months,
                "str_history": [None] * total_months,
                "fun_history": [None] * total_months,
                "real_estate_history": [None] * total_months,
            }

        self.results["wealth_history"][month] = (
            self.state["current_bank_balance"]
            + sum(self.state["liquid_assets"].values())
            + self.state["current_real_estate_value"]
        )
        self.results["bank_balance_history"][month] = self.state["current_bank_balance"]
        self.results["stocks_history"][month] = self.state["liquid_assets"]["stocks"]
        self.results["bonds_history"][month] = self.state["liquid_assets"]["bonds"]
        self.results["str_history"][month] = self.state["liquid_assets"]["str"]
        self.results["fun_history"][month] = self.state["liquid_assets"]["fun"]
        self.results["real_estate_history"][month] = self.state[
            "current_real_estate_value"
        ]

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

        final_nominal_wealth = (
            self.results["wealth_history"][months_lasted - 1]
            # if months_lasted > 0
            # else 0.0
        )
        final_cumulative_inflation = (
            self.state["monthly_cumulative_inflation_factors"][months_lasted - 1]
            # if months_lasted > 0
            # else 1.0
        )
        final_real_wealth = (
            final_nominal_wealth / final_cumulative_inflation
            # if final_cumulative_inflation
            # else 0.0
        )

        final_investment = (
            sum(self.state["liquid_assets"].values())
            + self.state["current_real_estate_value"]
        )
        final_bank_balance = self.state["current_bank_balance"]

        final_allocations_nominal = {
            "Stocks": self.state["liquid_assets"]["stocks"],
            "Bonds": self.state["liquid_assets"]["bonds"],
            "STR": self.state["liquid_assets"]["str"],
            "Fun": self.state["liquid_assets"]["fun"],
            "Real Estate": self.state["current_real_estate_value"],
        }
        final_allocations_real = {
            k: float(v / final_cumulative_inflation)
            for k, v in final_allocations_nominal.items()
        }

        result = {
            # --- Scalars first ---
            "success": not self.state["simulation_failed"],
            "months_lasted": months_lasted,
            "final_investment": final_investment,
            "final_bank_balance": final_bank_balance,
            "final_cumulative_inflation_factor": final_cumulative_inflation,
            "final_nominal_wealth": final_nominal_wealth,
            "final_real_wealth": final_real_wealth,
            # --- Non-state, derived or input data ---
            "final_allocations_nominal": final_allocations_nominal,
            "final_allocations_real": final_allocations_real,
            "initial_total_wealth": self.state["initial_total_wealth"],
            # --- State and histories ---
            "monthly_inflations_sequence": self.state["monthly_inflations_sequence"],
            "monthly_cumulative_inflation_factors": trunc_only(
                self.state["monthly_cumulative_inflation_factors"]
            ),
            "wealth_history": trunc_only(self.results["wealth_history"]),
            "bank_balance_history": trunc_only(self.results["bank_balance_history"]),
            "stocks_history": trunc_only(self.results["stocks_history"]),
            "bonds_history": trunc_only(self.results["bonds_history"]),
            "str_history": trunc_only(self.results["str_history"]),
            "fun_history": trunc_only(self.results["fun_history"]),
            "real_estate_history": trunc_only(self.results["real_estate_history"]),
        }

        return result
