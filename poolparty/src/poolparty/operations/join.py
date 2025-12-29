"""Join operation - join multiple sequences together."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class JoinOp(Operation):
    """Join multiple sequences."""
    factory_name = "join"
    design_card_keys = []
    
    @beartype
    def __init__(
        self,
        parent_pools: list,
        spacer_str: str = '',
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize JoinOp.
        
        Args:
            parent_pools: List of parent pools to join.
            spacer_str: String to insert between joined sequences.
            name: Optional operation name.
            op_iteration_order: Iteration order for this operation's counter.
        """
        self.spacer_str = spacer_str
        
        # Compute seq_length as sum of parent lengths plus spacers if all are known
        parent_lengths = [p.seq_length for p in parent_pools]
        if all(L is not None for L in parent_lengths):
            n_spacers = max(0, len(parent_pools) - 1)
            seq_length = sum(parent_lengths) + len(spacer_str) * n_spacers
        else:
            seq_length = None
        super().__init__(
            parent_pools=parent_pools,
            num_states=1,
            seq_length=seq_length,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return empty design card (no design decisions)."""
        return {}
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Join parent sequences."""
        return {'seq_0': self.spacer_str.join(parent_seqs)}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'spacer_str': self.spacer_str,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def join(
    items: list[Union[Pool_type, str]],
    spacer_str: str = '',
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Join multiple pools and/or strings.
    
    Args:
        items: List of pools and/or strings to join.
        spacer_str: String to insert between joined sequences.
        pool_iteration_order: Sort key for the result pool (default 0).
        op_iteration_order: Sort key for this operation's counter (default 0).
        op_name: Optional operation name.
        name: Optional pool name.
    
    Returns:
        A pool that joins all input sequences.
    """
    from .from_seqs import from_seqs
    parent_pools = []
    for item in items:
        if isinstance(item, str):
            pool = from_seqs([item], mode='fixed')
            parent_pools.append(pool)
        else:
            parent_pools.append(item)
    op = JoinOp(parent_pools, spacer_str=spacer_str, name=op_name,
                op_iteration_order=op_iteration_order)
    pool = Pool(operation=op, output_index=0)
    pool.iter_order = pool_iteration_order
    if name is not None:
        pool.name = name
    return pool

