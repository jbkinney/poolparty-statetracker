"""Filter operation for rejecting sequences based on a predicate."""

from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import Callable, CardsType, NullSeq, Optional, Pool_type, Seq, Sequence, beartype
from ..utils.parsing_utils import strip_all_tags


class FilterOp(Operation):
    """Operation that filters sequences based on a predicate function.

    If the predicate returns False for a sequence, the operation returns
    NullSeq, which propagates through downstream operations.
    """

    factory_name: str = "filter"
    design_card_keys: Sequence[str] = ["passed"]

    def __init__(
        self,
        parent_pool: Pool_type,
        predicate: Callable[[str], bool],
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize FilterOp."""
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,  # Filter is always fixed mode
            mode="fixed",
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            cards=cards,
        )
        self._predicate = predicate

    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[Seq, dict]:
        """Apply predicate and return Seq or NullSeq."""
        parent_seq = parents[0]

        # Get tag-free sequence content for the predicate
        if parent_seq.clean:
            seq_str = parent_seq.clean
        elif '<' in parent_seq.string:
            seq_str = strip_all_tags(parent_seq.string)
        else:
            seq_str = parent_seq.string
        passes = self._predicate(seq_str)

        if passes:
            return parent_seq, {"passed": True}
        else:
            return NullSeq(), {"passed": False}


@beartype
def filter(
    pool: Pool_type,
    predicate: Callable[[str], bool],
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    cards: CardsType = None,
) -> Pool:
    """Filter sequences based on a predicate function.

    Sequences for which the predicate returns False are replaced with NullSeq,
    which propagates through downstream operations. Use generate_library with
    discard_null_seqs=True to exclude filtered sequences from output.

    Parameters
    ----------
    pool : Pool_type
        Input pool to filter.
    predicate : Callable[[str], bool]
        Function taking sequence string (clean, no tags), returning True to keep.
    name : Optional[str]
        Optional name for the operation.
    prefix : Optional[str]
        Prefix for sequence names in the resulting Pool.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'passed'``.

    Returns
    -------
    Pool
        New pool that may contain NullSeq for filtered sequences.
    """
    op = FilterOp(parent_pool=pool, predicate=predicate, name=name, prefix=prefix, cards=cards)
    pool_class = type(pool)
    return pool_class(operation=op)
