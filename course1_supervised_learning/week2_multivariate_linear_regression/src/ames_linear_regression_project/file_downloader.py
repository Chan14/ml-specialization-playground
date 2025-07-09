from pathlib import Path

import pandas as pd

# URL for the raw Ames Housing dataset.
# This URL points to a raw CSV file often used for this dataset.
data_url = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"

# Define the columns we want to select.
# We'll pick a few numerical features that are typically useful for regression,
# plus the target variable 'SalePrice'.
# Column names have been adjusted to match the dataset from the specified URL.
selected_columns = [
    "Lot Area",  # Lot size in square feet (changed from LotArea)
    "Gr Liv Area",  # Above grade (ground) living area square feet (changed from GrLivArea)
    "Overall Qual",  # Overall material and finish quality (changed from OverallQual)
    "Total Bsmt SF",  # Total square feet of basement area (changed from TotalBsmtSF)
    "Year Built",  # Original construction date (changed from YearBuilt)
    "Garage Cars",  # Size of garage in car capacity (changed from GarageCars)
    "SalePrice",  # Target variable
]

try:
    # 1. Download the full dataset using pandas
    print(f"Attempting to download data from: {data_url}")
    df_full = pd.read_csv(data_url)
    print("Data loaded successfully.")

    # --- DEBUGGING STEP: Print all actual column names to verify ---
    # print(f"\nActual columns in the downloaded dataframe:")
    # print(df_full.columns.tolist())

    # 2. Select the specified columns
    # It's good practice to check if all selected columns exist in the downloaded data.
    # If a column is missing, this will raise a KeyError, which the try-except block will catch.
    missing_columns = [col for col in selected_columns if col not in df_full.columns]
    if missing_columns:
        raise ValueError(
            f"The following selected columns are not found in the downloaded data: {missing_columns}. Please check the 'Actual columns in the downloaded DataFrame:' output above to correct them."
        )
    df_subset = df_full[selected_columns]

    # 3. Take a subset of rows (e.g., the first 90 rows) to ensure less than 100
    # This also helps in keeping the file small for quick experimentation.
    df_small = df_subset.head(
        90
    ).copy()  # Using .copy() to avoid SettingWithCopyWarning

    # Handle any potential missing values by dropping rows (simple approach for this exercise)
    # For real analysis, we'd use more sophisticated imputation techniques.
    original_rows = df_small.shape[0]
    df_small.dropna(inplace=True)
    if df_small.shape[0] < original_rows:
        print(
            f"Dropped {original_rows - df_small.shape[0]} rows due to missing values."
        )

    # Define the data_dir and the output file name
    file_name = Path(__file__).resolve().parent / "data" / "ames_housing_subset.csv"

    # 4. Save the processed data to a new CSV file
    df_small.to_csv(file_name, index=False)
    print(f"\nCleaned Ames Housing data saved to '{file_name}'")

    # 5. Display information about the generated file
    print(f"\nFile details:")
    # print(f"  - Shape of data (rows, columns): {df_small.shape}")
    # print(f"  - Column info and types: \n  {df_small.info()}")
    # print(f"  - Descriptive strategies: \n  {df_small.describe().T}")
    print(f"  - Number of rows: {df_small.shape[0]}")
    print(f"  - Number of columns: {df_small.shape[1]}")
    print(f"  - Column names: {df_small.columns.tolist()}")

    print("\nFirst 5 rows of the generated data:")
    print(df_small.head())

except Exception as e:
    print(f"An error occurred: {e}")
    print(
        "Please check the URL, your internet connection, or if the selected columns exist in the dataset."
    )
