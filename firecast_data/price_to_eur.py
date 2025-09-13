#!/usr/bin/env python3
#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com>
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Converts asset prices from a USD-denominated Excel file to EUR.

This script takes two Excel files as input:
1. An asset price file where all prices are in USD.
2. An exchange rate file containing historical EUR/USD rates.

It aligns the two datasets by date, converts all asset prices to EUR,
and saves the result to a new Excel file.

The script assumes the exchange rate is quoted as EUR/USD (e.g., 1.08 means
1 EUR = 1.08 USD). Therefore, conversion is done by division:
Price in EUR = Price in USD / (EUR/USD Rate)

Usage
-----
python price_to_eur.py \\
    --prices-file my_prices_usd.xlsx \\
    --exchange-rate-file eur_usd_rates.xlsx \\
    --output-file my_prices_eur.xlsx
"""

import argparse

import pandas as pd


def convert_prices_to_eur(
    prices_file: str, exchange_rate_file: str, output_file: str
) -> None:
    """
    Loads USD prices and EUR/USD rates, converts prices to EUR, and saves.

    Args:
        prices_file: Path to the Excel file with asset prices in USD.
        exchange_rate_file: Path to the Excel file with EUR/USD rates.
        output_file: Path to save the new Excel file with prices in EUR.
    """
    # --- Load and Prepare Data ---
    print(f"Loading USD prices from '{prices_file}'...")
    prices_df = pd.read_excel(prices_file)
    if "Date" not in prices_df.columns:
        raise ValueError("Price file must contain a 'Date' column.")

    print(f"Loading EUR/USD exchange rates from '{exchange_rate_file}'...")
    rates_df = pd.read_excel(exchange_rate_file)
    if "Date" not in rates_df.columns:
        raise ValueError("Exchange rate file must contain a 'Date' column.")

    # Assume the first column after 'Date' is the rate
    rate_col_name = rates_df.columns[1]
    print(f"Using '{rate_col_name}' as the exchange rate column.")
    rates_df = rates_df[["Date", rate_col_name]].copy()
    rates_df.rename(columns={rate_col_name: "Rate"}, inplace=True)

    # Convert 'Date' columns to datetime and set as index for alignment
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    prices_df.set_index("Date", inplace=True)
    rates_df["Date"] = pd.to_datetime(rates_df["Date"])
    rates_df.set_index("Date", inplace=True)

    initial_rows = len(prices_df)

    # --- Align and Convert ---
    # Get the valid date range from the exchange rate data
    first_rate_date = rates_df.index.min()
    last_rate_date = rates_df.index.max()

    # Filter prices to the available rate range
    prices_df = prices_df.loc[first_rate_date:last_rate_date]

    # Combine the dataframes and forward-fill internal gaps (weekends/holidays)
    combined_df = prices_df.join(rates_df, how="left")
    combined_df["Rate"] = combined_df["Rate"].ffill()

    # Drop any remaining NaNs, which should only be at the very start if any
    combined_df.dropna(subset=["Rate"], inplace=True)

    if len(combined_df) < initial_rows:
        print(
            f"Warning: Dropped {initial_rows - len(combined_df)} rows from price data that fell outside the exchange rate's date range."
        )

    # --- Convert Prices and Save ---
    asset_cols = [col for col in prices_df.columns]
    for col in asset_cols:
        combined_df[col] = combined_df[col] / combined_df["Rate"]

    # Prepare final DataFrame for output
    output_df = combined_df[asset_cols].reset_index()
    output_df["Date"] = output_df["Date"].dt.strftime("%Y-%m-%d")

    print(f"Saving converted EUR prices to '{output_file}'...")
    output_df.to_excel(output_file, index=False)
    print("Conversion complete.")


def main() -> None:
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert asset prices from USD to EUR using an exchange rate file."
    )
    parser.add_argument(
        "-p",
        "--prices-file",
        type=str,
        required=True,
        help="Path to the Excel file with asset prices in USD.",
    )
    parser.add_argument(
        "-e",
        "--exchange-rate-file",
        type=str,
        required=True,
        help="Path to the Excel file with EUR/USD exchange rates.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Path for the output Excel file with prices in EUR.",
    )
    args = parser.parse_args()

    convert_prices_to_eur(args.prices_file, args.exchange_rate_file, args.output_file)


if __name__ == "__main__":
    main()
