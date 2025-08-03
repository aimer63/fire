import pandas as pd

# Load the monthly data
df_monthly = pd.read_excel("Ethereum-monthly.xlsx")
df_monthly["Date"] = (
    pd.to_datetime(df_monthly["Date"]).dt.to_period("M").dt.to_timestamp()
)
df_monthly.set_index("Date", inplace=True)
df_monthly = df_monthly.rename(columns={"Ethereum/USD": "Monthly_Price"})

# Load and process the daily data using a type-safe method
df_daily = pd.read_excel("Ethereum-daily.xlsx")
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.drop_duplicates(subset=["Date"], keep="last")
# Set index, resample, then bring the index back to a column for safe manipulation
df_daily = df_daily.set_index("Date").resample("ME").last().reset_index()
# Now normalize the 'Date' column, which is guaranteed to be a Series
df_daily["Date"] = df_daily["Date"].dt.to_period("M").dt.to_timestamp()
# Finally, set the clean column as the index
df_daily.set_index("Date", inplace=True)
df_daily = df_daily.rename(columns={"Ethereum/USD": "Daily_Resampled_Price"})

# Combine them into one DataFrame
comparison_df = pd.concat([df_monthly, df_daily], axis=1)
comparison_df.dropna(inplace=True)  # Ensure we only compare the common date range

# Find the differences
comparison_df["Difference"] = (
    comparison_df["Monthly_Price"] - comparison_df["Daily_Resampled_Price"]
).abs()

# Show the rows where the prices do NOT match
print("Rows with differing prices:")
print(comparison_df[comparison_df["Difference"] > 0.01])
