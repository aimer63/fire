#!/usr/bin/env python3
#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Convert daily price data to monthly by taking last available value per month, forward-filling missing months per asset, and reporting filled months."
    )
    parser.add_argument(
        "--daily-file",
        "-f",
        required=True,
        help="Input daily price file (Excel or CSV)",
    )
    parser.add_argument(
        "--monthly-file",
        "-m",
        required=True,
        help="Output monthly price file (Excel or CSV)",
    )
    args = parser.parse_args()

    # Load daily data
    try:
        if args.daily_file.endswith(".xlsx"):
            df = pd.read_excel(args.daily_file)
        else:
            df = pd.read_csv(args.daily_file)
    except FileNotFoundError:
        print(
            f"Error: The file '{args.daily_file}' was not found. Please check the path and try again."
        )
        return

    if "Date" not in df.columns:
        raise ValueError("Input file must have a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    monthly_df = pd.DataFrame(
        index=pd.date_range(df.index.min(), df.index.max(), freq="ME")
    )
    filling_report = {}

    for col in df.columns:
        monthly_col = df[col].resample("ME").last()
        monthly_df[col] = monthly_col.reindex(monthly_df.index)
        missing_months = monthly_df[col][monthly_df[col].isna()].index
        if len(missing_months) > 0:
            filling_report[col] = [d.strftime("%Y-%m") for d in missing_months]

    monthly_df.reset_index(inplace=True)
    monthly_df.rename(columns={"index": "Date"}, inplace=True)
    monthly_df["Date"] = monthly_df["Date"].dt.strftime("%Y-%m-%d")

    if args.monthly_file.endswith(".xlsx"):
        monthly_df.to_excel(args.monthly_file, index=False)
    else:
        monthly_df.to_csv(args.monthly_file, index=False)

    print("\nMissing months per asset (left empty in output):")
    for col, months in filling_report.items():
        print(f"- {col}: {', '.join(months)}")


if __name__ == "__main__":
    main()
