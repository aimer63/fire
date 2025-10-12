#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com>
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Yahoo Finance Data Downloader.

This script downloads historical price data for a list of tickers from Yahoo
Finance and saves it into a single Excel file. The output is formatted to be
compatible with the `portfolios.py` and `data_metrics.py` analysis scripts.

The script reads a list of tickers from a text file. This file supports
full-line and inline comments.

Key Features
------------
- Downloads historical 'Adj Close' prices for multiple tickers.
- Reads tickers from a text file, ignoring comments and blank lines.
- Allows specifying a date range and data interval (daily, weekly, monthly).
- Saves data to a clean Excel file with a 'Date' column.
- Reports any tickers that failed to download.

Dependencies
------------
This script requires pandas and yfinance. Install them with::

    pip install pandas yfinance openpyxl

Usage
-----
Download daily data for tickers listed in 'tickers.txt' and save to 'market_data.xlsx'::

    python download_data.py -i tickers.txt -o market_data.xlsx

Download data for a specific period (e.g., from 2010 to 2023)::

    python download_data.py -i tickers.txt -o market_data.xlsx -s 2010-01-01 -e 2023-12-31
"""

import argparse
import sys
from typing import List

import yfinance as yf

# Setup CLI argument parsing
parser = argparse.ArgumentParser(
    description="Download historical market data from Yahoo Finance."
)
parser.add_argument(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help="Path to the input file containing the list of tickers.",
)
parser.add_argument(
    "-o",
    "--output-file",
    type=str,
    required=True,
    help="Path for the output Excel file (e.g., 'market_data.xlsx').",
)
parser.add_argument(
    "-s",
    "--start-date",
    type=str,
    default=None,
    help="The start date for the data download in YYYY-MM-DD format.",
)
parser.add_argument(
    "-e",
    "--end-date",
    type=str,
    default=None,
    help="The end date for the data download in YYYY-MM-DD format.",
)
parser.add_argument(
    "--interval",
    type=str,
    default="1d",
    choices=["1d", "1wk", "1mo"],
    help="The data interval. Default is '1d' (daily).",
)


def parse_ticker_file(filepath: str) -> List[str]:
    """Parses a text file to extract a list of tickers, ignoring comments."""
    tickers = []
    with open(filepath) as f:
        for line in f:
            # Remove inline comments and strip whitespace
            ticker = line.split("#", 1)[0].strip()
            if ticker:  # Add to list if not an empty string
                tickers.append(ticker)
    return tickers


def main() -> None:
    """Main function to download and save the data."""
    args = parser.parse_args()

    try:
        tickers = parse_ticker_file(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    if not tickers:
        print("No tickers found in the input file. Exiting.")
        sys.exit(0)

    print(f"Downloading data for {len(tickers)} tickers: {', '.join(tickers)}")

    # Download data from yfinance. Use period="max" if no start date is given.
    period = "max" if args.start_date is None else None
    data = yf.download(
        tickers,
        start=args.start_date,
        end=args.end_date,
        interval=args.interval,
        period=period,
        progress=True,
    )

    if data is None or data.empty:
        print("No data downloaded. Check tickers and date range.")
        sys.exit(1)

    # Select 'Adj Close' and drop levels if necessary
    adj_close = data["Close"]

    # Report any tickers that failed to download (all NaN columns)
    failed_tickers = adj_close.columns[adj_close.isna().all()].tolist()
    if failed_tickers:
        print(f"\nWarning: Failed to download data for: {', '.join(failed_tickers)}")
        # Drop failed tickers from the DataFrame
        adj_close = adj_close.drop(columns=failed_tickers)

    if adj_close.empty:
        print("No valid data remains after removing failed tickers. Exiting.")
        sys.exit(1)

    # Reset index to turn 'Date' into a column, format it, and save
    output_df = adj_close.reset_index()
    output_df["Date"] = output_df["Date"].dt.strftime("%Y-%m-%d")
    output_df.to_excel(args.output_file, index=False)
    print(f"\nSuccessfully saved data to '{args.output_file}'")


if __name__ == "__main__":
    main()
