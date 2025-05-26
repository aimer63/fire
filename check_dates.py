import pandas as pd
import os

excel_file_name = 'data/MSCI World Historical Data.xlsx' # Keep this as it is

try:
    print(f"Current Working Directory: {os.getcwd()}")

    # --- ADD THESE LINES TO DEBUG FILE PATHS ---
    # data_dir_path = os.path.join(os.getcwd(), 'data')
    # print(f"Expected 'data' directory: {data_dir_path}")

    # if not os.path.exists(data_dir_path):
    #     print(f"ERROR: The 'data' directory DOES NOT EXIST at: {data_dir_path}")
    # else:
    #     print(f"Contents of 'data' directory: {os.listdir(data_dir_path)}")
    #     # Construct the full absolute path Python is trying to open
    #     full_file_path = os.path.join(os.getcwd(), excel_file_name)
    #     print(f"Full absolute file path Python is looking for: {full_file_path}")

    #     if not os.path.exists(full_file_path):
    #         print(f"ERROR: The file '{excel_file_name}' DOES NOT EXIST at the full path: {full_file_path}")
    #     else:
    #         print(f"CONFIRMED: File '{excel_file_name}' EXISTS at: {full_file_path}")
            # --- END ADDITION ---


    df = pd.read_excel(excel_file_name)

    print("\n--- Original DataFrame Info ---")
    df.info()
    print("\n--- First 5 rows of 'Date' column ---")
    print(df['Date'].head())

    # Attempt to convert to datetime
    df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')

    print("\n--- Parsed 'Date_Parsed' Column Info ---")
    df['Date_Parsed'].info()
    print("\n--- First 5 rows of 'Date_Parsed' column ---")
    print(df['Date_Parsed'].head())

    # Check if any dates failed to parse
    if df['Date_Parsed'].isnull().any():
        print("\nWARNING: Some dates could not be parsed. Check rows where 'Date_Parsed' is NaT (Not a Time).")
        print(df[df['Date_Parsed'].isnull()])
    else:
        print("\nSUCCESS: All dates appear to be parsed correctly!")

except FileNotFoundError:
    print(f"Error: A FileNotFoundError occurred for '{excel_file_name}'. This usually means the path is still incorrect or inaccessible.")
except KeyError:
    print("Error: 'Date' column not found. Please ensure your date column header is exactly 'Date' (case-sensitive).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")