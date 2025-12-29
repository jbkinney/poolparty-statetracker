"""Slice operation - extract a subsequence from a pool."""

import numpy as np

from ..types import Union, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party


class SliceOp(Operation):
    """Extract a subsequence using Python slice notation.
    
    This is a fixed-mode operation - it has no internal variability.
    Supports both integer indexing and slice objects.
    """
    
    design_card_keys = []  # No additional metadata
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool,
        key: Union[int, slice],
        name: str = 'slice',
    ) -> None:
        """Initialize SliceOp.
        
        Args:
            parent_pool: Input pool to slice
            key: Integer index or slice object
            name: Operation name
        """
        self.key = key
        
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,  # Fixed mode
            mode='fixed',
            name=name,
        )
        
        # Register with active party
        party = get_active_party()
        if party is not None:
            party._register_operation(self)
    
    @beartype
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Apply slice to parent sequence."""
        seq = parent_seqs[0]
        result = seq[self.key]
        
        # Handle single character indexing (returns str, but ensure it's a string)
        if isinstance(result, str):
            sliced = result
        else:
            sliced = str(result)
        
        return {
            'seq_0': sliced,
        }


@beartype
def subseq(
    parent: Pool,
    key: Union[int, slice],
    name: str = 'slice',
) -> Pool:
    """Extract a subsequence from a pool.
    
    Args:
        parent: Input pool
        key: Integer index or slice object
        name: Operation name
    
    Returns:
        Pool with sliced sequences
    
    Example:
        >>> pool = from_seqs(['ACGTACGT'])
        >>> first_half = subseq(pool, slice(0, 4))  # 'ACGT'
        >>> last_char = subseq(pool, -1)  # 'T'
        >>> # Or using Pool indexing:
        >>> first_half = pool[0:4]
        >>> last_char = pool[-1]
    """
    op = SliceOp(parent, key=key, name=name)
    return Pool(operation=op, output_index=0)
