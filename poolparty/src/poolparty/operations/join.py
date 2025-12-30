"""Join operation - join multiple sequences together."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
class JoinOp(Operation):
    """Join multiple sequences."""
    factory_name = "join"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pools: list,
        spacer_str: str = '',
        name: Optional[str] = None,
        iter_order: Real = 0,
    ) -> None:
        """Initialize JoinOp."""
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
            iter_order=iter_order,
        )
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return empty design card (no design decisions)."""
        return {}
    
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
            'iter_order': self.iter_order,
        }


@beartype
def join(
    segment_pools: Sequence[Union[Pool_type, str]],
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool_type:
    """
    Concatenate multiple Pools or string sequences into a single Pool.

    Parameters
    ----------
    segment_pools : Sequence[Union[Pool_type, str]]
        List of Pool objects and/or strings to be joined in order.
        Any provided string is automatically converted to a constant Pool.
    spacer_str : str, default=''
        String to insert between joined sequences.
    name : Optional[str], default=None
        Name to assign to the resulting Pool.
    op_name : Optional[str], default=None
        Name to assign to the internal JoinOp operation.
    iter_order : Real, default=0
        Iteration priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration priority for the internal JoinOp operation (typically unused).

    Returns
    -------
    Pool_type
        A Pool whose states yield joined sequences from the specified inputs.
    """
    from .from_seq import from_seq
    parent_pools = [from_seq(item) if isinstance(item, str) else item for item in segment_pools]
    op = JoinOp(parent_pools, spacer_str=spacer_str, name=op_name,
                iter_order=op_iter_order)
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool
