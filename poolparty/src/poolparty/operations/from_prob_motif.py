"""FromProbMotif operation - generate sequences by sampling from a position probability matrix."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import Alphabet
from ..party import get_active_party
import numpy as np
import pandas as pd


@beartype
def from_prob_motif(
    prob_df: pd.DataFrame,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """
    Create a Pool that samples sequences from a position probability matrix.

    Parameters
    ----------
    prob_df : pd.DataFrame
        DataFrame with probability values for each position.
        Columns should be alphabet characters (e.g., 'A', 'C', 'G', 'T').
        Rows represent positions. Values are probabilities (auto-normalized).
    mode : ModeType, default='random'
        Sequence selection mode: 'random' or 'hybrid'.
    num_hybrid_states : Optional[int], default=None
        Number of pool states when using 'hybrid' mode.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool_type
        A Pool yielding sequences sampled from the probability matrix.
    """
    if mode not in ('random', 'hybrid'):
        raise ValueError(
            f"from_prob_motif only supports mode='random' or mode='hybrid', got mode='{mode}'. "
            "Sequential iteration is not available for probability-based sampling."
        )
    op = FromProbMotifOp(
        prob_df=prob_df,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class FromProbMotifOp(Operation):
    """Sample sequences from a position probability matrix."""
    factory_name = "from_prob_motif"
    design_card_keys = ['prob_state']

    def __init__(
        self,
        prob_df: pd.DataFrame,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize FromProbMotifOp."""
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")

        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_prob_motif requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        self.alphabet = party.alphabet

        # Validate and store probability matrix
        self.prob_df = _validate_prob_df(prob_df, self.alphabet)
        self._cumprobs = np.cumsum(self.prob_df.values, axis=1)

        match mode:
            case 'hybrid':
                num_states = num_hybrid_states
            case _:
                num_states = 1

        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=len(self.prob_df),
            name=name,
            iter_order=iter_order,
        )

    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with sampled position indices."""
        if rng is None:
            raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
        length = len(self.prob_df)
        random_vals = rng.random(length)
        indices = (random_vals[:, np.newaxis] < self._cumprobs).argmax(axis=1)
        return {'prob_state': indices.tolist()}

    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the sequence based on design card indices."""
        indices = card['prob_state']
        alphabet_chars = self.alphabet.chars
        seq = ''.join(alphabet_chars[i] for i in indices)
        return {'seq_0': seq}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'prob_df': self.prob_df.copy(),
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }


def _validate_prob_df(prob_df: pd.DataFrame, alphabet: Alphabet) -> pd.DataFrame:
    """Validate and normalize a probability DataFrame.

    Args:
        prob_df: DataFrame with columns that are a subset of alphabet characters.
            Missing alphabet columns are filled with zeros.
        alphabet: Alphabet object from the active Party.

    Returns:
        Normalized DataFrame with rows summing to 1 and columns ordered by alphabet.

    Raises:
        ValueError: If prob_df is empty, has columns not in alphabet, contains NaN,
            contains negative values, or has rows summing to zero.
    """
    if prob_df.empty or len(prob_df) == 0:
        raise ValueError("prob_df must be a non-empty DataFrame")

    # Validate columns are a subset of alphabet chars
    alphabet_chars = alphabet.chars
    expected_cols = set(alphabet_chars)
    actual_cols = set(prob_df.columns)
    extra = actual_cols - expected_cols
    if extra:
        raise ValueError(
            f"prob_df columns must be alphabet characters ({alphabet_chars}). "
            f"Extra columns: {sorted(extra)}."
        )

    if prob_df.isna().any().any():
        raise ValueError("prob_df must not contain NaN values")

    if np.any(prob_df.values < 0):
        raise ValueError("All prob_df values must be >= 0")

    # Add missing columns with zeros
    result_df = prob_df.copy()
    for col in alphabet_chars:
        if col not in result_df.columns:
            result_df[col] = 0.0

    # Normalize rows to sum to 1
    prob_matrix = result_df[alphabet_chars].values.astype(np.float64).copy()
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("prob_df rows must not sum to zero")
    prob_matrix /= row_sums

    return pd.DataFrame(prob_matrix, columns=alphabet_chars)
