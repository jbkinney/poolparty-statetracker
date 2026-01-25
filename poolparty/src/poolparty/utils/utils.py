"""Utility functions for poolparty."""
from ..types import Sequence
import pandas as pd


def validate_iteration_order(iteration_order: Sequence[int], n: int) -> None:
    """Validate that iteration_order is a permutation of [0, 1, ..., n-1].
    
    Args:
        iteration_order: Sequence of integers specifying iteration order.
        n: Expected length (number of items being ordered).
    
    Raises:
        ValueError: If iteration_order is not a valid permutation.
    """
    if len(iteration_order) != n:
        raise ValueError(
            f"iteration_order must have length {n} (number of items), "
            f"got {len(iteration_order)}"
        )
    expected = set(range(n))
    actual = set(iteration_order)
    if actual != expected:
        raise ValueError(
            f"iteration_order must contain exactly the integers 0 to {n-1}, "
            f"got {sorted(actual)}"
        )


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

