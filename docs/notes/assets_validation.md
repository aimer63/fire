# Refactoring Plan: `shocks` Configuration and Validation

This plan details the steps to refactor the `shocks` configuration to a more concise format
and to introduce a top-level `Config` model to enable robust, cross-sectional validation.

---

## 1. Pydantic Model Overhaul (`firestarter/config/config.py`)

The data models will be restructured to introduce a single container and enforce consistency.

- **Create Top-Level `Config` Model:**

  - A new `Config` class will be created to act as a container for the entire
    configuration file.
  - It will hold all other configuration models as attributes (e.g., `deterministic_inputs`,
    `market_assumptions`, `shocks`).

- **Redefine `Shock` Model:**

  - The model will be changed from representing a single-asset shock to a multi-asset
    shock event for a specific year.
  - It will contain `year`, an optional `description`, and a flexible `impact`
    dictionary mapping asset names (`str`) to their shock magnitudes (`float`).

- **Implement `Config` Validator:**
  - A `model_validator` will be added to the new `Config` class to perform checks
    after the entire file has been parsed.
  - **Source of Truth:** The validator will establish the set of keys from the
    `market_assumptions.assets` dictionary as the definitive list of all valid assets.
  - **Correlation Matrix Validation:** It will ensure the `assets` list within the
    `correlation_matrix` is perfectly identical to the source-of-truth asset set.
  - **Shock Validation:** It will iterate through all shock events and verify that
    every key in each shock's `impact` dictionary is a valid, defined asset.

---

## 2. Update Main Script (`firestarter/main.py`)

The main script will be simplified to use the new `Config` model.

- The script will be modified to load the entire TOML file into the new `Config` model
  with a single command: `config = Config(**config_data)`.
- All subsequent code will be updated to access configuration sections via the `config`
  object (e.g., `config.deterministic_inputs`, `config.shocks`).

---

## 3. Update Simulation Logic (`firestarter/core/simulation.py`)

The `_precompute_sequences` method will be adapted to process the new shock structure.

- The method will now iterate through the list of new multi-asset `Shock` objects.
- For each `Shock` object, it will loop through the `impact` dictionary.
- The existing logic for converting an annual shock magnitude to an equivalent monthly
  rate and overwriting the return sequence will be applied for each asset within the
  `impact` dictionary.

---

## 4. Update Configuration Files (`configs/test_config.toml`)

The TOML configuration files will be updated to use the new, more compact format.

- The existing `[[shocks]]` entries will be replaced with the new structure. For example:

  ```toml
  # New format for a multi-asset shock in a single year
  [[shocks]]
  year = 10
  description = "October 1929 equivalent"
  impact = { stocks = -0.35, bonds = 0.02, inflation = -0.023 }
  ```

---

## 5. Update and Create Tests

Testing will be updated to cover the new models and validation logic.

- **Rewrite `tests/simulation/test_shocks.py`:**

  - The tests will be modified to use the new `Shock` model structure.
  - A single test will be able to verify that a multi-asset shock event correctly
    modifies the return sequences for all targeted assets in the specified year.

- **Update `tests/config/test_config.py`:**
  - It will include tests that confirm:
    - A valid, consistent configuration file loads successfully.
    - A configuration with a shock targeting an undefined asset fails validation.
    - A configuration with an incorrect `assets` list in the correlation matrix fails
      validation.
