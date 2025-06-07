# flake8: noqa=F821
"""
simulation_v1.py

First refactored version of the simulation engine using the builder pattern.
This file is intended for experimentation and future development.
"""


class SimulationBuilder:
    def __init__(self):
        self.det_inputs = None
        self.econ_assumptions = None
        self.portfolio_rebalances = None
        self.shock_events = None
        self.initial_assets = None

    @classmethod
    def new(cls):
        return cls()

    def set_det_inputs(self, det_inputs):
        self.det_inputs = det_inputs
        return self

    def set_econ_assumptions(self, econ_assumptions):
        self.econ_assumptions = econ_assumptions
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

        return Simulation(
            self.det_inputs,
            self.econ_assumptions,
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

    def init(self):
        self.state = self.initialize_state()
        self.precompute_sequences()

    def run(self):
        for month in range(self.det_inputs.simulation_months):
            self.process_income(month)
            self.handle_contributions(month)
            self.handle_expenses(month)
            self.handle_bank_top_up(month)
            self.handle_withdrawals(month)
            self.handle_house_purchase(month)
            self.rebalance_if_needed(month)
            self.record_results(month)
        return self.build_result()

    # --- Helper methods (stubs for now) ---
    def initialize_state(self):
        """
        Initialize all state variables for the simulation.
        Returns a dictionary or custom object holding the simulation state.
        """
        state = {
            "current_bank_balance": self.det_inputs.initial_bank_balance,
            "current_stocks_value": self.initial_assets["stocks"],
            "current_bonds_value": self.initial_assets["bonds"],
            "current_str_value": self.initial_assets["str"],
            "current_fun_value": self.initial_assets["fun"],
            "current_real_estate_value": self.initial_assets["real_estate"],
            # Optionally add more state variables as needed
        }
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
        cumulative_inflation_factors_annual = np.ones(total_years + 1, dtype=np.float64)
        for year_idx in range(total_years):
            cumulative_inflation_factors_annual[year_idx + 1] = cumulative_inflation_factors_annual[
                year_idx
            ] * (1.0 + annual_inflations_sequence[year_idx])

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
                real_amount * cumulative_inflation_factors_annual[year_idx]
            )
            nominal_planned_contributions_amounts.append((nominal_contribution_amount, year_idx))

        nominal_planned_extra_expenses_amounts = []
        local_planned_extra_expenses = list(planned_extra_expenses)
        for real_amount, year_idx in local_planned_extra_expenses:
            nominal_extra_expense_amount = float(
                real_amount * cumulative_inflation_factors_annual[year_idx]
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
                    * cumulative_inflation_factors_annual[pension_start_year_idx]
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
                    * cumulative_inflation_factors_annual[salary_start_year_idx]
                    * salary_factor
                )

        # Store all sequences in self.state
        self.state["annual_inflations_sequence"] = annual_inflations_sequence
        self.state["annual_stocks_returns_sequence"] = annual_stocks_returns_sequence
        self.state["annual_bonds_returns_sequence"] = annual_bonds_returns_sequence
        self.state["annual_str_returns_sequence"] = annual_str_returns_sequence
        self.state["annual_fun_returns_sequence"] = annual_fun_returns_sequence
        self.state["annual_real_estate_returns_sequence"] = annual_real_estate_returns_sequence
        self.state["cumulative_inflation_factors_annual"] = cumulative_inflation_factors_annual
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
        Contributions are allocated according to the current portfolio weights,
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
                weights = self._get_current_portfolio_weights(current_year)
                self.state["current_stocks_value"] += (
                    nominal_contribution_amount * weights["stocks"]
                )
                self.state["current_bonds_value"] += nominal_contribution_amount * weights["bonds"]
                self.state["current_str_value"] += nominal_contribution_amount * weights["str"]
                self.state["current_fun_value"] += nominal_contribution_amount * weights["fun"]
                # Do NOT allocate to real estate

        # Regular monthly contribution (inflation-adjusted)
        if det_inputs.monthly_investment_contribution > 0.0:
            monthly_contribution = (
                det_inputs.monthly_investment_contribution
                * self.state["cumulative_inflation_factors_annual"][current_year]
            )
            weights = self._get_current_portfolio_weights(current_year)
            self.state["current_stocks_value"] += monthly_contribution * weights["stocks"]
            self.state["current_bonds_value"] += monthly_contribution * weights["bonds"]
            self.state["current_str_value"] += monthly_contribution * weights["str"]
            self.state["current_fun_value"] += monthly_contribution * weights["fun"]
            # Do NOT allocate to real estate

    def handle_expenses(self, month):
        pass

    def handle_bank_top_up(self, month):
        pass

    def handle_withdrawals(self, month):
        pass

    def handle_house_purchase(self, month):
        pass

    def rebalance_if_needed(self, month):
        pass

    def record_results(self, month):
        pass

    def build_result(self):
        # Return final simulation results
        pass

    def _get_current_portfolio_weights(self, year_idx):
        """
        Helper to get the current portfolio weights for contributions.
        Uses the initial rebalance weights (Phase 1) or the current phase if dynamic.
        Real estate is excluded from liquid allocations.
        """
        reb = self.portfolio_rebalances.rebalances[0]
        return {
            "stocks": reb.stocks,
            "bonds": reb.bonds,
            "str": reb.str,
            "fun": reb.fun,
            # Do NOT include real estate
        }


# Usage example (with placeholder variables):
det_inputs = ...  # Replace with DeterministicInputs instance
econ_assumptions = ...  # Replace with EconomicAssumptions instance
portfolio_rebalances = ...  # Replace with PortfolioRebalances instance
shock_events = ...  # Replace with list[ShockEvent]
initial_assets = ...  # Replace with dict[str, float]

builder = SimulationBuilder.new()
simulation = (
    builder.set_det_inputs(det_inputs)
    .set_econ_assumptions(econ_assumptions)
    .set_portfolio_rebalances(portfolio_rebalances)
    .set_shock_events(shock_events)
    .set_initial_assets(initial_assets)
    .build()
)
simulation.init()
simulation.run()
