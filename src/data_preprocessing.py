import pandas as pd
from IPython.display import display

def load_and_preprocess_data(csv_path):
    """Load and preprocess the e-commerce CSV data."""
    try:
        # Load CSV with low_memory=False for large files
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        display(df.head(3))

        # Preprocess data
        processed_df = preprocess_csv(df)
        print(f"Preprocessed {len(processed_df)} rows of e-commerce data")

        # Display data types
        print("\nData types after preprocessing:")
        print(processed_df.dtypes.to_string())

        # Display sample of processed columns
        if "Purchase_Amount" in processed_df.columns:
            print("\nSample of processed Purchase_Amount column:")
            print(processed_df["Purchase_Amount"].head())
        if "Time_of_Purchase" in processed_df.columns:
            print("\nSample of processed Time_of_Purchase column:")
            print(processed_df["Time_of_Purchase"].head())

        # Check for missing values
        missing_values = processed_df.isnull().sum()
        print("\nMissing values:")
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")

        return processed_df

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        raise
    except Exception as e:
        print(f"Error during data loading/preprocessing: {str(e)}")
        raise

def preprocess_csv(df):
    """Clean and prepare the CSV data."""
    processed_df = df.copy()

    # Clean string columns in one go
    str_columns = processed_df.select_dtypes(include=["object"]).columns
    for col in str_columns:
        processed_df[col] = processed_df[col].str.strip()

    # Convert Purchase_Amount to numeric
    if "Purchase_Amount" in processed_df.columns:
        processed_df["Purchase_Amount"] = (
            processed_df["Purchase_Amount"]
            .str.replace("$", "", regex=False)
            .str.strip()
            .astype(float)
        )

    # Convert Time_of_Purchase to datetime
    if "Time_of_Purchase" in processed_df.columns:
        processed_df["Time_of_Purchase"] = pd.to_datetime(
            processed_df["Time_of_Purchase"], format="%m/%d/%Y", errors="coerce"
        )

    return processed_df
