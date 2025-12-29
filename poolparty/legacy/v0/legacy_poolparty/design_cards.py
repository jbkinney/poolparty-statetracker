"""DesignCards class for columnar storage of sequence metadata."""

from typing import List, Dict, Any, Optional


class DesignCards:
    """Columnar storage for design card data.
    
    Provides efficient storage and retrieval of sequence metadata in a
    column-oriented format, suitable for large-scale sequence generation.
    
    Example:
        >>> dc = DesignCards(['sequence_id', 'promoter_index', 'promoter_value'])
        >>> dc.append_row({'sequence_id': 0, 'promoter_index': 1, 'promoter_value': 'AAAA'})
        >>> dc['promoter_index']
        [1]
        >>> dc.to_dataframe()
           sequence_id  promoter_index promoter_value
        0            0               1           AAAA
    """
    
    def __init__(self, keys: List[str]):
        """Initialize with fixed schema (column names).
        
        Args:
            keys: List of column names for the design cards.
        """
        self._keys = list(keys)
        self._data: Dict[str, List[Any]] = {key: [] for key in keys}
        self._num_rows = 0
    
    def append_row(self, values: Dict[str, Any]) -> None:
        """Append a row. Missing keys get None.
        
        Args:
            values: Dictionary mapping column names to values.
                    Keys not in schema are ignored.
                    Missing keys get None as value.
        """
        for key in self._keys:
            self._data[key].append(values.get(key))
        self._num_rows += 1
    
    def __getitem__(self, key: str) -> List:
        """Get column by name.
        
        Args:
            key: Column name.
            
        Returns:
            List of values for that column.
            
        Raises:
            KeyError: If key is not in schema.
        """
        if key not in self._data:
            raise KeyError(f"Column '{key}' not found. Available columns: {self._keys}")
        return self._data[key]
    
    def __len__(self) -> int:
        """Number of rows."""
        return self._num_rows
    
    def __contains__(self, key: str) -> bool:
        """Check if column exists."""
        return key in self._data
    
    def to_dataframe(self):
        """Convert to pandas DataFrame.
        
        Returns:
            pandas.DataFrame with columns in schema order.
            
        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
        
        return pd.DataFrame(self._data, columns=self._keys)
    
    def get_row(self, idx: int) -> Dict[str, Any]:
        """Get single row as dictionary.
        
        Args:
            idx: Row index (0-based).
            
        Returns:
            Dictionary mapping column names to values for that row.
            
        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f"Row index {idx} out of range [0, {self._num_rows})")
        
        return {key: self._data[key][idx] for key in self._keys}
    
    @property
    def keys(self) -> List[str]:
        """List of column names."""
        return list(self._keys)
    
    def __repr__(self) -> str:
        return f"DesignCards({self._num_rows} rows, {len(self._keys)} columns)"

