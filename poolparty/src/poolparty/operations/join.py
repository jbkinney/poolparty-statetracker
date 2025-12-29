"""Join operation - join multiple sequences together."""
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
    ) -> None:
        """Initialize JoinOp.
        
        Args:
            parent_pools: List of parent pools to join.
            spacer_str: String to insert between joined sequences.
            name: Optional operation name.
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
        }


@beartype
def join(
    items: list[Union[Pool_type, str]],
    spacer_str: str = '',
    iteration_order: int = 0,
    op_name: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> Pool_type:
    """Join multiple pools and/or strings.
    
    Args:
        items: List of pools and/or strings to join.
        spacer_str: String to insert between joined sequences.
        iteration_order: Sort key for this operation's counter (default 0).
        op_name: Optional operation name.
        pool_name: Optional pool name.
    
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
    op = JoinOp(parent_pools, spacer_str=spacer_str, name=op_name)
    op.counter.iteration_order = iteration_order
    pool = Pool(operation=op, output_index=0)
    if pool_name is not None:
        pool.name = pool_name
    return pool

