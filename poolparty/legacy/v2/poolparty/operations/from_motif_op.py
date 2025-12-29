"""FromMotif operation - generate sequences by sampling from a position probability matrix."""

from ..types import Optional, Sequence, ModeType, AlphabetType, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import validate_alphabet
import numpy as np
import pandas as pd


@beartype
class FromMotifOp(Operation):
    """Sample sequences from a position probability matrix."""
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        motif_df: pd.DataFrame,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        # Validate alphabet
        self.alphabet = validate_alphabet(alphabet)
        
        # Validate and store motif
        self.motif_df = _validate_motif_df(motif_df, self.alphabet)
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=-1,  # Unknown number of states (random sampling)
            mode=mode,
            seq_length=len(motif_df),
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results(
        self, 
        input_strings_lists: Sequence[Sequence[str]], 
        sequential_states: Sequence[int],
    ) -> None:
        """Generate all sequences at once using vectorized sampling."""
        num_seqs = len(sequential_states)
        length = len(self.motif_df)
        random_vals = self.rng.random((num_seqs, length))  # (num_seqs, length)
        random_vals_expanded = random_vals[:, :, np.newaxis]  # (num_seqs, length, 1)
        cumprobs = np.cumsum(self.motif_df.values, axis=1)  # (length, alpha)
        cumprobs_expanded = cumprobs[np.newaxis, :, :]  # (1, length, alpha)
        indices = (random_vals_expanded < cumprobs_expanded).argmax(axis=2)  # (num_seqs, length)
        alphabet_arr = np.array(list(self.alphabet))  # (alpha,)
        char_matrix = alphabet_arr[indices]           # (num_seqs, length)
        seqs = char_matrix.view(f'U{length}').ravel().tolist()  
        self._results_df = pd.DataFrame({'seq': seqs})

#########################################################
# Public factory function
#########################################################

@beartype
def from_motif(
    motif_df: pd.DataFrame,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'random',
    name: str = 'from_motif',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that samples sequences from a position probability matrix.
    
    Args:
        motif_df: DataFrame with probability values for each position.
            Columns should be alphabet characters (e.g., 'A', 'C', 'G', 'T').
            Rows represent positions. Values are probabilities (auto-normalized).
        alphabet: Alphabet to use ('dna', 'rna', or custom list).
        mode: Only 'random' is supported.
        name: Name for the pool.
        design_card_keys: Keys to include in design cards.
    """
    # Enforce random mode only
    if mode != 'random':
        raise ValueError(
            f"from_motif only supports mode='random', got mode='{mode}'. "
            "Sequential iteration is not available for this operation."
        )
    
    return Pool(
        operation=FromMotifOp(
            motif_df=motif_df,
            alphabet=alphabet,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        ),
    )

def _validate_motif_df(motif_df: pd.DataFrame, alphabet: Sequence[str]) -> pd.DataFrame:
    """Validate and normalize a motif probability DataFrame.
    
    Args:
        motif_df: DataFrame with columns that are a subset of alphabet characters.
            Missing alphabet columns are filled with zeros.
        alphabet: Sequence of alphabet characters (e.g., ['A', 'C', 'G', 'T']).
        
    Returns:
        Normalized DataFrame with rows summing to 1 and columns ordered by alphabet.
        
    Raises:
        ValueError: If motif_df is empty, has columns not in alphabet, contains NaN,
            contains negative values, or has rows summing to zero.
    """
    # Validate motif_df is not empty
    if motif_df.empty or len(motif_df) == 0:
        raise ValueError("motif_df must be a non-empty DataFrame")
    
    # Validate columns are a subset of alphabet (reject extra columns)
    expected_cols = set(alphabet)
    actual_cols = set(motif_df.columns)
    extra = actual_cols - expected_cols
    if extra:
        raise ValueError(
            f"motif_df columns must be alphabet characters. "
            f"Extra: {sorted(extra)}."
        )
    
    # Validate no NaN values in provided columns
    if motif_df.isna().any().any():
        raise ValueError("motif_df must not contain NaN values")
    
    # Validate all values are non-negative
    if np.any(motif_df.values < 0):
        raise ValueError("All motif_df values must be >= 0")
    
    # Add missing columns with zeros
    for col in alphabet:
        if col not in motif_df.columns:
            motif_df = motif_df.copy()
            motif_df[col] = 0.0
    
    # Copy and normalize rows to sum to 1
    prob_matrix = motif_df[alphabet].values.astype(np.float64).copy()
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("motif_df rows must not sum to zero")
    prob_matrix /= row_sums
    
    return pd.DataFrame(prob_matrix, columns=alphabet)