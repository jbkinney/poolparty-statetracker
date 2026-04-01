"""FromMotif operation - generate sequences by sampling from a position probability matrix."""

from numbers import Real

import numpy as np
import pandas as pd

from ..dna_pool import DnaPool
from ..operation import Operation
from ..party import get_active_party
from ..pool import Pool
from ..types import CardsType, Integral, ModeType, Optional, Pool_type, RegionType, Seq, Union, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


@beartype
def from_motif(
    prob_df: pd.DataFrame,
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
    cards: CardsType = None,
) -> DnaPool:
    """
    Create a Pool that samples sequences from a position probability matrix.

    Parameters
    ----------
    prob_df : pd.DataFrame
        DataFrame with probability values for each position.
        Columns should be alphabet characters (e.g., 'A', 'C', 'G', 'T').
        Rows represent positions. Values are probabilities (auto-normalized).
    pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, generated sequence
        replaces the region content.
    region : RegionType, default=None
        Region to replace in pool. Can be a marker name or [start, stop] interval.
        Required if pool is provided.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Sequence selection mode: 'random'.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to generated sequences (e.g., 'red', 'blue bold').
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'prob_state'``.

    Returns
    -------
    Pool_type
        A Pool yielding sequences sampled from the probability matrix.

    Raises
    ------
    ValueError
        If pool is provided without region.
    """
    if mode != "random":
        raise ValueError(
            f"from_motif only supports mode='random', got mode='{mode}'. "
            "Sequential iteration is not available for probability-based sampling."
        )
    from ..fixed_ops.from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = FromMotifOp(
        prob_df=prob_df,
        parent_pool=pool_obj,
        region=region,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        style=style,
        cards=cards,
    )
    result_pool = DnaPool(operation=op)
    return result_pool


class FromMotifOp(Operation):
    """Sample sequences from a position probability matrix."""

    factory_name = "from_motif"
    design_card_keys = ["prob_state"]

    def __init__(
        self,
        prob_df: pd.DataFrame,
        parent_pool: Optional[Pool] = None,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize FromMotifOp."""

        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_motif requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )

        # Validate parent_pool/region combination
        if parent_pool is not None and region is None:
            raise ValueError(
                "region is required when parent_pool is provided. "
                "Specify which region of parent_pool to replace with the generated sequence."
            )

        # Validate and store probability matrix
        self.prob_df = _validate_prob_df(prob_df)
        self._cumprobs = np.cumsum(self.prob_df.values, axis=1)
        self._style = style

        match mode:
            case "random":
                # num_states stays None for pure random mode
                pass
            case _:
                num_states = 1

        insert_length = len(self.prob_df)
        if parent_pool is None:
            seq_length = insert_length
        else:
            parent_seq_length = parent_pool.seq_length
            if isinstance(region, str):
                try:
                    region_obj = party.get_region_by_name(region)
                    region_length = region_obj.seq_length
                except ValueError:
                    region_length = None
            else:
                region_length = region[1] - region[0] if region is not None else None

            if parent_seq_length is None or region_length is None:
                seq_length = None
            else:
                seq_length = parent_seq_length - region_length + insert_length

        parent_pools_list = [parent_pool] if parent_pool is not None else []
        super().__init__(
            parent_pools=parent_pools_list,
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq and design card."""
        if rng is None:
            raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
        length = len(self.prob_df)
        random_vals = rng.random(length)
        indices = (random_vals[:, np.newaxis] < self._cumprobs).argmax(axis=1)
        indices_list = indices.tolist()
        seq_string = "".join(dna_utils.BASES[i] for i in indices_list)

        # Apply styling if requested
        from ..utils.style_utils import SeqStyle, styles_suppressed

        if styles_suppressed():
            output_seq = DnaSeq(seq_string, None)
        else:
            output_style = SeqStyle.full(len(seq_string), self._style)
            output_seq = DnaSeq(seq_string, output_style)

        return output_seq, {
            "prob_state": indices_list,
        }


def _validate_prob_df(prob_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a probability DataFrame.

    Args:
        prob_df: DataFrame with columns that are a subset of DNA bases (A, C, G, T).
            Missing columns are filled with zeros.

    Returns:
        Normalized DataFrame with rows summing to 1 and columns ordered by dna_utils.BASES.

    Raises:
        ValueError: If prob_df is empty, has columns not in BASES, contains NaN,
            contains negative values, or has rows summing to zero.
    """
    if prob_df.empty or len(prob_df) == 0:
        raise ValueError("prob_df must be a non-empty DataFrame")

    # Validate columns are a subset of DNA bases
    expected_cols = set(dna_utils.BASES)
    actual_cols = set(prob_df.columns)
    extra = actual_cols - expected_cols
    if extra:
        raise ValueError(
            f"prob_df columns must be DNA bases ({dna_utils.BASES}). "
            f"Extra columns: {sorted(extra)}."
        )

    if prob_df.isna().any().any():
        raise ValueError("prob_df must not contain NaN values")

    if np.any(prob_df.values < 0):
        raise ValueError("All prob_df values must be >= 0")

    # Add missing columns with zeros
    result_df = prob_df.copy()
    for col in dna_utils.BASES:
        if col not in result_df.columns:
            result_df[col] = 0.0

    # Normalize rows to sum to 1
    prob_matrix = result_df[dna_utils.BASES].values.astype(np.float64).copy()
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("prob_df rows must not sum to zero")
    prob_matrix /= row_sums

    return pd.DataFrame(prob_matrix, columns=dna_utils.BASES)
