import numpy as np
import pandas as pd
from typing import Union, Literal, Dict, Any
from .pool import Pool


class MotifPool(Pool):
    """A class for generating sequences by sampling from a position-specific probability matrix.
    
    Each position in the sequence is sampled independently from its corresponding probability
    distribution defined in the probability_df DataFrame. This class is compatible with
    Logomaker's probability matrix format. Has infinite states and only supports random mode.
    
    Supports forward, reverse complement, or randomly sampled orientation for strand-aware
    motif scanning experiments (e.g., MPRA assays testing TF binding on both strands).
    """
    
    # Complement mapping for reverse complement
    _COMPLEMENT = str.maketrans('ACGTacgt', 'TGCAtgca')
    
    def __init__(self, 
                 probability_df: pd.DataFrame,
                 orientation: Literal['forward', 'reverse', 'both'] = 'forward',
                 forward_prob: float = 0.5,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a MotifPool.
        
        Args:
            probability_df: A pandas DataFrame where rows represent positions and columns
                represent characters. Each entry is the probability of that character at
                that position. Row sums must be close to 1.0. Column names must be single
                characters and unique. This follows the Logomaker probability matrix format.
            orientation: How to handle strand orientation:
                - 'forward': Use sampled sequence as-is (default)
                - 'reverse': Always reverse complement the sampled sequence
                - 'both': Randomly choose orientation per sample using forward_prob
            forward_prob: Probability of forward orientation when orientation='both'.
                Ignored when orientation is 'forward' or 'reverse'. Default: 0.5
            mode: Must be 'random' (default). Sequential mode is not supported.
            iteration_order: Order for iteration (default: auto-assigned based on creation order)
            name: Optional name for this pool
            metadata: Metadata level ('core', 'features', 'complete')
                
        Raises:
            ValueError: If mode is not 'random', if probability_df is empty,
                if column names are not single characters or not unique,
                if row sums are not close to 1.0, if orientation is invalid,
                if forward_prob is not between 0 and 1
        """
        # Enforce random mode only
        if mode != 'random':
            raise ValueError(
                f"MotifPool only supports mode='random', got mode='{mode}'. "
                "Sequential iteration is not available for this pool type."
            )
        
        # Validate orientation
        if orientation not in ('forward', 'reverse', 'both'):
            raise ValueError(
                f"orientation must be 'forward', 'reverse', or 'both', got '{orientation}'"
            )
        
        # Validate forward_prob
        if not 0 <= forward_prob <= 1:
            raise ValueError(f"forward_prob must be between 0 and 1, got {forward_prob}")
        
        # Validate probability_df is not empty
        if probability_df.empty:
            raise ValueError("probability_df must be a non-empty DataFrame")
        
        if len(probability_df) == 0:
            raise ValueError("probability_df must have at least one row (position)")
        
        if len(probability_df.columns) == 0:
            raise ValueError("probability_df must have at least one column (character)")
        
        # Validate column names are single characters
        for col in probability_df.columns:
            if not isinstance(col, str):
                raise ValueError(
                    f"All column names must be strings, but column '{col}' "
                    f"is {type(col).__name__}"
                )
            if len(col) != 1:
                raise ValueError(
                    f"All column names must be single characters, but column "
                    f"'{col}' has length {len(col)}"
                )
        
        # Validate column names are unique
        if len(probability_df.columns) != len(set(probability_df.columns)):
            duplicates = [col for col in set(probability_df.columns) 
                         if list(probability_df.columns).count(col) > 1]
            raise ValueError(
                f"All column names must be unique, but found duplicates: {duplicates}"
            )
        
        # Validate no NaN values
        if probability_df.isna().any().any():
            raise ValueError("probability_df must not contain NaN values")
        
        # Validate all probabilities are non-negative
        if np.any(probability_df.values < 0):
            raise ValueError("All probability values must be non-negative")
        
        # Validate row sums are close to 1.0
        row_sums = probability_df.sum(axis=1)
        if not np.all(np.isclose(row_sums, 1.0)):
            bad_rows = []
            for idx, (row_idx, row_sum) in enumerate(row_sums.items()):
                if not np.isclose(row_sum, 1.0):
                    bad_rows.append((row_idx, row_sum))
            
            error_msg = "All rows in probability_df must sum to approximately 1.0. "
            error_msg += f"Found {len(bad_rows)} row(s) with invalid sums:\n"
            for row_idx, row_sum in bad_rows[:5]:  # Show first 5 bad rows
                error_msg += f"  Row {row_idx}: sum = {row_sum}\n"
            if len(bad_rows) > 5:
                error_msg += f"  ... and {len(bad_rows) - 5} more\n"
            raise ValueError(error_msg.rstrip())
        
        # Store the probability matrix
        self.probability_df = probability_df.copy()
        
        # Store the alphabet (column names)
        self.alphabet = list(probability_df.columns)
        
        # Pre-convert to numpy arrays for fast vectorized sampling
        # This eliminates expensive pandas DataFrame indexing in _compute_seq
        self._prob_matrix = probability_df.values.astype(np.float64)  # Shape: (length, num_chars)
        self._cumprobs = np.cumsum(self._prob_matrix, axis=1)  # Cumulative probs for vectorized sampling
        
        # Store orientation settings
        self.orientation = orientation
        self.forward_prob = forward_prob
        
        # Design cards: cached orientation from last _compute_seq call
        self._cached_orientation: str | None = None
        self._cached_state: int | None = None
        
        super().__init__(op='motif', mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
    
    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return the reverse complement of a DNA sequence.
        
        Args:
            seq: DNA sequence string (supports ACGT and acgt)
            
        Returns:
            Reverse complement of the input sequence
        """
        return seq.translate(MotifPool._COMPLEMENT)[::-1]
    
    def _calculate_num_internal_states(self) -> float:
        """MotifPool has infinite internal states (different random samples from distributions)."""
        return float('inf')
    
    def _calculate_seq_length(self) -> int:
        """Sequence length equals the number of rows (positions) in probability_df."""
        return len(self.probability_df)
    
    def _compute_seq(self) -> str:
        """Compute sequence by sampling each position from its probability distribution.
        
        Uses vectorized numpy operations for fast sampling:
        1. Generate all random values at once
        2. Use cumulative probabilities and broadcasting to select characters
        
        Applies orientation transformation based on the orientation setting:
        - 'forward': Returns sampled sequence as-is
        - 'reverse': Returns reverse complement of sampled sequence
        - 'both': Randomly chooses orientation using forward_prob
        
        Returns:
            A sequence string sampled from the position-specific probability matrix,
            potentially reverse complemented based on orientation setting.
        """
        # Create a numpy random generator with the current state as seed
        rng = np.random.Generator(np.random.PCG64(self.get_state()))
        
        # Vectorized sampling: generate all random values at once
        length = len(self._prob_matrix)
        random_vals = rng.random(length)
        
        # Use cumulative probabilities to find character indices
        # For each position, find the first index where cumprob >= random_val
        # Broadcasting: random_vals[:, None] compares against each cumprob row
        indices = (random_vals[:, np.newaxis] < self._cumprobs).argmax(axis=1)
        
        # Convert indices to characters
        seq = ''.join(self.alphabet[i] for i in indices)
        
        # Apply orientation transformation
        if self.orientation == 'reverse':
            seq = self._reverse_complement(seq)
            self._cached_orientation = 'reverse'
        elif self.orientation == 'both':
            # Use the same RNG to decide orientation (deterministic per state)
            if rng.random() < self.forward_prob:
                self._cached_orientation = 'forward'
            else:
                seq = self._reverse_complement(seq)
                self._cached_orientation = 'reverse'
        else:  # 'forward'
            self._cached_orientation = 'forward'
        
        self._cached_state = self.get_state()
        return seq
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this MotifPool at the current state.
        
        Extends base Pool metadata with orientation information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + orientation (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Ensure cached values are current (recompute if state changed)
        if self._cached_state != self.get_state():
            _ = self.seq  # Trigger computation to populate cache
        
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add MotifPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['orientation'] = self._cached_orientation
        
        return metadata
    
    def __repr__(self) -> str:
        shape_str = f"{len(self.probability_df)}x{len(self.probability_df.columns)}"
        orientation_str = f", orientation='{self.orientation}'" if self.orientation != 'forward' else ""
        prob_str = f", forward_prob={self.forward_prob}" if self.orientation == 'both' and self.forward_prob != 0.5 else ""
        return f"MotifPool(shape={shape_str}, alphabet={self.alphabet}{orientation_str}{prob_str})"
