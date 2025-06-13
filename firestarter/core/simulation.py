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

from typing import TypedDict, Dict, Any
import numpy as np
from firestarter.core.constants import ASSET_KEYS, WITHDRAWAL_PRIORITY


class SimulationBuilder:
    def __init__(self):
        self.det_inputs = None
        self.market_assumptions = None
        self.portfolio_rebalances = None
        self.shock_events = None
        self.initial_assets = None

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

    def build(self):
        # Validate all required fields are set
        if self.det_inputs is None:
            raise ValueError("det_inputs must be set before building the simulation.")
        if self.market_assumptions is None:
            raise ValueError("econ_assumptions must be set before building the simulation.")
        if self.portfolio_rebalances is None:
            raise ValueError("portfolio_rebalances must be set before building the simulation.")
        if self.shock_events is None:
            raise ValueError("shock_events must be set before building the simulation.")
        if self.initial_assets is None:
            raise ValueError("initial_assets must be set before building the simulation.")

        return Simulation(
            self.det_inputs,
            self.market_assumptions,
            self.portfolio_rebalances,
            self.shock_events,
            self.initial_assets,
        )


class Simulation:
    def __init__(
        self, det_inputs, econ_assumptions, portfolio_rebalances, shock_events, initial_assets
    ):
        self.det_inputs = det_inputs
        self.econ_assumptions = econ_assumptions
        self.portfolio_rebalances = portfolio_rebalances
        self.shock_events = shock_events
        self.initial_assets = initial_assets
        self.state = None
        self.results = None

    @property
    def simulation_months(self):
        """
        Total number of months to simulate, based on years_to_simulate.
        """
        return self.det_inputs.years_to_simulate * 12

    def init(self):
        self.state = self.initialize_state()
        self.precompute_sequences()

    def run(self):
        """
        Main simulation loop.
        Processes all monthly flows, then ensures the bank account is within bounds.
        Exits early if a shortfall cannot be covered by liquid assets.
        """
        for month in range(self.simulation_months):
            self.process_income(month)
            self.handle_contributions(month)
            self.handle_expenses(month)
            self.handle_house_purchase(month)
            self.handle_bank_account(
                month
            )  # Ensures bank is within bounds, handles withdrawals/investments

            if self.state.get("simulation_failed"):
                break  # Exit early if a shortfall could not be covered

            self.apply_monthly_returns(month)
            self.rebalance_if_needed(month)
            self.record_results(month)
        return self.build_result()

    # --- Helper methods (stubs for now) ---
    def initialize_state(self):
        """
        Initialize all state variables for the simulation.
        Returns a dictionary or custom object holding the simulation state.
        """
        # Find the initial target portfolio weights (from the first rebalance)
        first_reb = self.portfolio_rebalances.rebalances[0]
        initial_target_weights = {
            asset: getattr(first_reb, asset) for asset in ASSET_KEYS if asset != "real_estate"
        }

        state = {
            "current_bank_balance": self.det_inputs.initial_bank_balance,
            "liquid_assets": {
                asset: self.initial_assets[asset] for asset in ASSET_KEYS if asset != "real_estate"
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

    def precompute_sequences(self):
        """
        Precompute all annual and monthly sequences needed for the simulation.

        Salary & Pension Logic:
        - For each year, we precompute the *monthly* salary and monthly pension amount
          for that year, already adjusted for inflation and any adjustment factor.
        - These values are stored in `nominal_salary_annual_sequence` and `nominal_pension_annual_sequence`.
        - During the simulation, for each month, we add the value for the current year to income.
        - This means salary and pension are constant within a year, but can change annually.

        Planned Contributions & Expenses:
        - Planned contributions and extra expenses are specified as (real_amount, year).
        - We convert these to nominal values for each year using cumulative inflation factors.
        - These are stored as lists of (nominal_amount, year_idx) for use during the simulation.

        This approach matches the legacy simulation logic for equivalence.
        """
        import numpy as np
        from firestarter.core.helpers import annual_to_monthly_compounded_rate

        det_inputs = self.det_inputs
        econ_assumptions = self.econ_assumptions
        portfolio_rebalances = self.portfolio_rebalances
        shock_events = self.shock_events

        lognormal = econ_assumptions.lognormal
        mu_log_stocks, sigma_log_stocks = lognormal["stocks"]
        mu_log_bonds, sigma_log_bonds = lognormal["bonds"]
        mu_log_str, sigma_log_str = lognormal["str"]
        mu_log_fun, sigma_log_fun = lognormal["fun"]
        mu_log_real_estate, sigma_log_real_estate = lognormal["real_estate"]
        mu_log_inflation, sigma_log_inflation = lognormal["inflation"]

        total_years = det_inputs.years_to_simulate
        total_months = total_years * 12

        # Annual sequences
        annual_inflations_sequence = (
            np.random.lognormal(mu_log_inflation, sigma_log_inflation, total_years).astype(
                np.float64
            )
            - 1.0
        )
        annual_stocks_returns_sequence = (
            np.random.lognormal(mu_log_stocks, sigma_log_stocks, total_years).astype(np.float64)
            - 1.0
        )
        annual_bonds_returns_sequence = (
            np.random.lognormal(mu_log_bonds, sigma_log_bonds, total_years).astype(np.float64) - 1.0
        )
        annual_str_returns_sequence = (
            np.random.lognormal(mu_log_str, sigma_log_str, total_years).astype(np.float64) - 1.0
        )
        annual_fun_returns_sequence = (
            np.random.lognormal(mu_log_fun, sigma_log_fun, total_years).astype(np.float64) - 1.0
        )
        annual_real_estate_returns_sequence = (
            np.random.lognormal(mu_log_real_estate, sigma_log_real_estate, total_years).astype(
                np.float64
            )
            - 1.0
        )

        # Apply shocks
        for shock in shock_events:
            shock_year = shock.year
            shock_asset = shock.asset
            shock_magnitude = shock.magnitude
            if 0 <= shock_year < total_years:
                if shock_asset == "Stocks":
                    annual_stocks_returns_sequence[shock_year] = shock_magnitude
                elif shock_asset == "Bonds":
                    annual_bonds_returns_sequence[shock_year] = shock_magnitude
                elif shock_asset == "STR":
                    annual_str_returns_sequence[shock_year] = shock_magnitude
                elif shock_asset == "Fun":
                    annual_fun_returns_sequence[shock_year] = shock_magnitude
                elif shock_asset == "Real Estate":
                    annual_real_estate_returns_sequence[shock_year] = shock_magnitude
                elif shock_asset == "Inflation":
                    annual_inflations_sequence[shock_year] = shock_magnitude

        # Cumulative inflation factors (annual)
        annual_cumulative_inflation_factors = np.ones(total_years + 1, dtype=np.float64)
        for year_idx in range(total_years):
            annual_cumulative_inflation_factors[year_idx + 1] = annual_cumulative_inflation_factors[
                year_idx
            ] * (1.0 + annual_inflations_sequence[year_idx])

        # --- Uniform monthly inflation rate array and cumulative factors ---
        monthly_cumulative_inflation_factors = np.ones(total_months + 1, dtype=np.float64)
        for month_idx in range(total_months):
            # Compute monthly rate on the fly
            year_idx = month_idx // 12
            monthly_rate = annual_to_monthly_compounded_rate(annual_inflations_sequence[year_idx])
            monthly_cumulative_inflation_factors[month_idx + 1] = (
                monthly_cumulative_inflation_factors[month_idx] * (1.0 + monthly_rate)
            )

        # Monthly returns lookup
        monthly_returns_lookup = {
            "Stocks": np.zeros(total_months, dtype=np.float64),
            "Bonds": np.zeros(total_months, dtype=np.float64),
            "STR": np.zeros(total_months, dtype=np.float64),
            "Fun": np.zeros(total_months, dtype=np.float64),
            "Real Estate": np.zeros(total_months, dtype=np.float64),
        }
        for year_idx in range(total_years):
            monthly_stocks_rate = annual_to_monthly_compounded_rate(
                annual_stocks_returns_sequence[year_idx]
            )
            monthly_bonds_rate = annual_to_monthly_compounded_rate(
                annual_bonds_returns_sequence[year_idx]
            )
            monthly_str_rate = annual_to_monthly_compounded_rate(
                annual_str_returns_sequence[year_idx]
            )
            monthly_fun_rate = annual_to_monthly_compounded_rate(
                annual_fun_returns_sequence[year_idx]
            )
            monthly_real_estate_rate = annual_to_monthly_compounded_rate(
                annual_real_estate_returns_sequence[year_idx]
            )

            start_month = year_idx * 12
            end_month = min((year_idx + 1) * 12, total_months)
            monthly_returns_lookup["Stocks"][start_month:end_month] = monthly_stocks_rate
            monthly_returns_lookup["Bonds"][start_month:end_month] = monthly_bonds_rate
            monthly_returns_lookup["STR"][start_month:end_month] = monthly_str_rate
            monthly_returns_lookup["Fun"][start_month:end_month] = monthly_fun_rate
            monthly_returns_lookup["Real Estate"][start_month:end_month] = monthly_real_estate_rate

        # Planned contributions and extra expenses (nominal, inflation-adjusted)
        planned_contributions = det_inputs.planned_contributions
        planned_extra_expenses = det_inputs.planned_extra_expenses

        nominal_planned_contributions_amounts = []
        for real_amount, year_idx in planned_contributions:
            nominal_contribution_amount = float(
                real_amount * annual_cumulative_inflation_factors[year_idx]
            )
            nominal_planned_contributions_amounts.append((nominal_contribution_amount, year_idx))

        nominal_planned_extra_expenses_amounts = []
        local_planned_extra_expenses = list(planned_extra_expenses)
        for real_amount, year_idx in local_planned_extra_expenses:
            nominal_extra_expense_amount = float(
                real_amount * annual_cumulative_inflation_factors[year_idx]
            )
            nominal_planned_extra_expenses_amounts.append((nominal_extra_expense_amount, year_idx))

        # Precompute nominal pension and salary annual sequences
        nominal_pension_annual_sequence = np.zeros(total_years, dtype=np.float64)
        nominal_salary_annual_sequence = np.zeros(total_years, dtype=np.float64)

        pension_start_year_idx = det_inputs.pension_start_year
        salary_start_year_idx = det_inputs.salary_start_year
        salary_end_year_idx = det_inputs.salary_end_year

        for year_idx in range(total_years):
            # Pension
            if year_idx >= pension_start_year_idx:
                if year_idx > pension_start_year_idx:
                    pension_adjusted_inflations = (
                        annual_inflations_sequence[pension_start_year_idx:year_idx]
                        * det_inputs.pension_inflation_factor
                    )
                    pension_factor = float(np.prod(1.0 + pension_adjusted_inflations))
                else:
                    pension_factor = 1.0
                nominal_pension_annual_sequence[year_idx] = (
                    det_inputs.monthly_pension
                    * annual_cumulative_inflation_factors[pension_start_year_idx]
                    * pension_factor
                )
            # Salary
            if salary_start_year_idx <= year_idx < salary_end_year_idx:
                if year_idx > salary_start_year_idx:
                    salary_adjusted_inflations = (
                        annual_inflations_sequence[salary_start_year_idx:year_idx]
                        * det_inputs.salary_inflation_factor
                    )
                    salary_factor = float(np.prod(1.0 + salary_adjusted_inflations))
                else:
                    salary_factor = 1.0
                nominal_salary_annual_sequence[year_idx] = (
                    det_inputs.monthly_salary
                    * annual_cumulative_inflation_factors[salary_start_year_idx]
                    * salary_factor
                )

        # Store all sequences in self.state
        self.state["annual_inflations_sequence"] = annual_inflations_sequence
        self.state["annual_stocks_returns_sequence"] = annual_stocks_returns_sequence
        self.state["annual_bonds_returns_sequence"] = annual_bonds_returns_sequence
        self.state["annual_str_returns_sequence"] = annual_str_returns_sequence
        self.state["annual_fun_returns_sequence"] = annual_fun_returns_sequence
        self.state["annual_real_estate_returns_sequence"] = annual_real_estate_returns_sequence
        self.state["annual_cumulative_inflation_factors"] = annual_cumulative_inflation_factors
        self.state["monthly_cumulative_inflation_factors"] = monthly_cumulative_inflation_factors
        self.state["monthly_returns_lookup"] = monthly_returns_lookup
        self.state["nominal_planned_contributions_amounts"] = nominal_planned_contributions_amounts
        self.state["nominal_planned_extra_expenses_amounts"] = (
            nominal_planned_extra_expenses_amounts
        )
        self.state["nominal_pension_annual_sequence"] = nominal_pension_annual_sequence
        self.state["nominal_salary_annual_sequence"] = nominal_salary_annual_sequence

    def process_income(self, month):
        """
        For each month, add the precomputed *monthly* salary and pension for the current year.
        These values are constant within a year, but can change annually due to inflation/adjustment.
        This matches the legacy simulation logic.
        """
        income = 0.0
        year = month // 12

        # Pension (precomputed, already inflation/adjustment adjusted)
        if year < len(self.state["nominal_pension_annual_sequence"]):
            income += self.state["nominal_pension_annual_sequence"][year]

        # Salary (precomputed, already inflation/adjustment adjusted)
        if year < len(self.state["nominal_salary_annual_sequence"]):
            income += self.state["nominal_salary_annual_sequence"][year]

        self.state["current_bank_balance"] += income

    def handle_contributions(self, month):
        """
        Handles planned one-time contributions and regular monthly contributions.
        Contributions are allocated according to the current target portfolio weights,
        but NEVER to real estate (see real_estate.md).
        """
        det_inputs = self.det_inputs
        current_year = month // 12
        month_in_year = month % 12

        # Planned one-time contributions (applied at the first month of the year)
        for nominal_contribution_amount, year_idx in self.state[
            "nominal_planned_contributions_amounts"
        ]:
            if current_year == year_idx and month_in_year == 0:
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
                * self.state["annual_cumulative_inflation_factors"][current_year]
            )
            weights = self.state["current_target_portfolio_weights"]
            increments = {
                asset: monthly_contribution * weights[asset]
                for asset in ASSET_KEYS
                if asset != "real_estate"
            }
            for asset, delta in increments.items():
                self.state["liquid_assets"][asset] += delta

    def handle_expenses(self, month):
        """
        Deducts regular monthly expenses and planned extra expenses from the bank balance.
        Expenses are inflation-adjusted. Planned extra expenses are applied at the first month of their year.
        """
        det_inputs = self.det_inputs
        current_year = month // 12
        month_in_year = month % 12

        # Regular monthly expenses (inflation-adjusted)
        nominal_monthly_expenses = (
            det_inputs.monthly_expenses
            * self.state["annual_cumulative_inflation_factors"][current_year]
        )
        total_expenses = nominal_monthly_expenses

        # Planned extra expenses (applied at the first month of the year)
        for nominal_extra_expense_amount, year_idx in self.state[
            "nominal_planned_extra_expenses_amounts"
        ]:
            if current_year == year_idx and month_in_year == 0:
                total_expenses += nominal_extra_expense_amount

        # Deduct from bank balance
        self.state["current_bank_balance"] -= total_expenses

    def handle_house_purchase(self, month):
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

        current_year = month // 12
        month_in_year = month % 12

        # Only purchase at the first month of the scheduled year
        if current_year == house_purchase_year and month_in_year == 0:
            # Inflation-adjusted nominal house cost
            cumulative_inflation = self.state["annual_cumulative_inflation_factors"][
                house_purchase_year
            ]
            nominal_house_cost = house_cost_real * cumulative_inflation

            # Use the unified withdrawal logic, but do NOT increase bank balance
            old_bank_balance = self.state["current_bank_balance"]
            self._withdraw_from_assets(nominal_house_cost)
            if self.state.get("simulation_failed"):
                return
            # Remove the artificial bank increase (since we don't want to increase bank, just pay for house)
            self.state["current_bank_balance"] = old_bank_balance

            # Add house value to real estate
            self.state["current_real_estate_value"] += nominal_house_cost

            # --- Rebalance remaining liquid assets according to current target portfolio weights ---
            total_liquid = sum(self.state["liquid_assets"].values())
            weights = self.state["current_target_portfolio_weights"]
            if total_liquid > 0:
                for asset in ASSET_KEYS:
                    if asset != "real_estate":
                        self.state["liquid_assets"][asset] = total_liquid * weights[asset]
            else:
                for asset in ASSET_KEYS:
                    if asset != "real_estate":
                        self.state["liquid_assets"][asset] = 0.0

    def handle_bank_account(self, month):
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
            if self.state.get("simulation_failed"):
                return
            self.state["current_bank_balance"] = lower

        # Invest excess if above upper bound
        if self.state["current_bank_balance"] > upper:
            excess = self.state["current_bank_balance"] - upper
            weights = self.state["current_target_portfolio_weights"]
            increments = {asset: excess * weights[asset] for asset in self.state["liquid_assets"]}
            for asset, delta in increments.items():
                self.state["liquid_assets"][asset] += delta
            self.state["current_bank_balance"] = upper

    def apply_monthly_returns(self, month):
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

    def rebalance_if_needed(self, month):
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
            sum_weights = sum(weights.values())
            if sum_weights > 0 and total_liquid > 0:
                # Normalize weights and rebalance
                for asset in ASSET_KEYS:
                    if asset != "real_estate":
                        self.state["liquid_assets"][asset] = total_liquid * (
                            weights[asset] / sum_weights
                        )
            else:
                for asset in ASSET_KEYS:
                    if asset != "real_estate":
                        self.state["liquid_assets"][asset] = 0.0

    def record_results(self, month):
        """
        Record the current state of the simulation at the end of the month.
        This includes nominal wealth, bank balance, and all asset values.
        """
        if self.results is None:
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
        self.results["real_estate_history"][month] = self.state["current_real_estate_value"]

    def build_result(self) -> Dict[str, Any]:
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
            self.results["wealth_history"][months_lasted - 1] if months_lasted > 0 else 0.0
        )
        final_cumulative_inflation = (
            self.state["monthly_cumulative_inflation_factors"][months_lasted - 1]
            if months_lasted > 0
            else 1.0
        )
        final_real_wealth = (
            final_nominal_wealth / final_cumulative_inflation if final_cumulative_inflation else 0.0
        )

        final_investment = (
            sum(self.state["liquid_assets"].values()) + self.state["current_real_estate_value"]
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
            k: float(v / final_cumulative_inflation) for k, v in final_allocations_nominal.items()
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
            "annual_inflations_sequence": self.state["annual_inflations_sequence"],
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
