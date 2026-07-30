"""
data_loader.py
--------------
Loads the cellphone price dataset and runs basic structural checks
(shape, dtypes, missing values, duplicates).
"""

import os
import pandas as pd

DEFAULT_PATHS = [
    "data/train.csv",
    "data/datasets_11167_15520_train.csv",
    "train.csv",
]


def load_data(path: str = None) -> pd.DataFrame:
    """
    Loads the mobile price dataset from `path`, or from the first match in
    DEFAULT_PATHS if no path is given. Raises FileNotFoundError if nothing
    is found, so the pipeline fails loudly instead of silently using fake data.
    """
    candidates = [path] if path else DEFAULT_PATHS
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            df = pd.read_csv(candidate)
            print(f"[data_loader] Loaded dataset from '{candidate}' "
                  f"-> shape {df.shape}")
            return df

    raise FileNotFoundError(
        "Could not find the dataset. Place 'train.csv' inside the 'data/' folder "
        "(or pass an explicit path to load_data())."
    )


def basic_checks(df: pd.DataFrame) -> dict:
    """
    Runs the 'Basic Checks' step of the analysis: shape, dtypes, missing
    values, and duplicate rows. Prints a readable summary and returns the
    same information as a dict for downstream use / logging.
    """
    print("\n===== BASIC CHECKS =====")
    print(f"Rows (phones): {df.shape[0]}")
    print(f"Columns (features incl. target): {df.shape[1]}")

    print("\nData types:")
    print(df.dtypes.value_counts())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    print("None found." if missing.empty else missing)

    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"Duplicates removed -> new shape {df.shape}")

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_total": int(missing.sum()),
        "duplicates_removed": int(dup_count),
    }


if __name__ == "__main__":
    df = load_data()
    basic_checks(df)
