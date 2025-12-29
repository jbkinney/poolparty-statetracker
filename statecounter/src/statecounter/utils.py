"""Utility functions for statecounter."""
import pandas as pd


def clean_df_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float columns that contain only integers/NaN to int/None.
    
    A column is converted only if ALL values are either NaN or whole numbers.
    Columns with actual float values (e.g., 3.14) are left unchanged.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            notna_mask = df[col].notna()
            # Check if all non-NaN values are whole numbers
            if notna_mask.any():
                notna_vals = df.loc[notna_mask, col]
                is_all_whole = (notna_vals == notna_vals.astype(int)).all()
            else:
                # All NaN column - don't convert
                is_all_whole = False
            if is_all_whole:
                df[col] = df[col].astype(object)
                df.loc[notna_mask, col] = df.loc[notna_mask, col].astype(int)
                df.loc[~notna_mask, col] = None
    return df
