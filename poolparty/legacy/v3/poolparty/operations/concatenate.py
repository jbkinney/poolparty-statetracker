"""Concatenate operation - join multiple sequences together."""

import numpy as np

from ..types import Union, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party


class ConcatenateOp(Operation):
    """Concatenate multiple sequences.
    
    This is a fixed-mode operation - it has no internal variability.
    The variability comes from its parent pools.
    """
    
    design_card_keys = []  # No additional metadata
    
    @beartype
    def __init__(
        self,
        parent_pools: list[Pool],
        name: str = 'concat',
    ) -> None:
        """Initialize ConcatenateOp.
        
        Args:
            parent_pools: Pools to concatenate
            name: Operation name
        """
        super().__init__(
            parent_pools=parent_pools,
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
        """Concatenate parent sequences."""
        return {
            'seq_0': ''.join(parent_seqs),
        }


@beartype
def concatenate(
    items: list[Union[Pool, str]],
    name: str = 'concat',
) -> Pool:
    """Concatenate multiple pools and/or strings.
    
    Args:
        items: List of Pools and/or literal strings
        name: Operation name
    
    Returns:
        Pool containing concatenated sequences
    
    Example:
        >>> oligo = concatenate([left, '...', right])
        >>> # Or using operators:
        >>> oligo = left + '...' + right
    """
    from .from_seqs import from_seqs
    
    # Convert strings to pools
    parent_pools = []
    for item in items:
        if isinstance(item, str):
            # Create a fixed pool from the literal string
            pool = from_seqs([item], mode='sequential', name='literal')
            parent_pools.append(pool)
        else:
            parent_pools.append(item)
    
    op = ConcatenateOp(parent_pools, name=name)
    return Pool(operation=op, output_index=0)
