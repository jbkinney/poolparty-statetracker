"""SwapCase operation - swap case of sequence characters."""
from numbers import Real
import statecounter as sc
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def swap_case(
    pool: Union[Pool_type, str],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool containing case-swapped sequences from the input pool.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to swap case.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing case-swapped sequences.
    """
    from .from_seq import from_seq
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = SwapCaseOp(pool_obj, name=op_name, iter_order=op_iter_order)
    result_pool = Pool(operation=op, iter_order=iter_order, name=name)
    return result_pool


@beartype
class SwapCaseOp(Operation):
    """Swap case of sequences from the parent pool."""
    factory_name = "swap_case"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool: Pool,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize SwapCaseOp."""
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> sc.Counter:
        """Return parent counter directly (no state added)."""
        return parent_pools[0].counter
    
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
        """Swap case of parent sequence."""
        seq = parent_seqs[0]
        return {'seq_0': seq.swapcase()}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'name': None,
            'iter_order': self.iter_order,
        }
