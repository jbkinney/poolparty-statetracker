"""SeqSlice operation - slice SEQUENCES (string slicing)."""
from numbers import Real
import statecounter as sc
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class SeqSliceOp(Operation):
    """Slice sequences using Python slice notation."""
    factory_name = "seq_slice"
    design_card_keys = []
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool_type,
        key: Union[int, slice],
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize SeqSliceOp."""
        self.key = key
        # Compute seq_length from slice params and parent length
        parent_len = parent_pool.seq_length
        if parent_len is not None:
            if isinstance(key, int):
                seq_length = 1
            else:
                start, stop, step = key.indices(parent_len)
                seq_length = max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)
        else:
            seq_length = None
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            seq_length=seq_length,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> sc.Counter:
        """Return parent counter directly (no state added)."""
        return parent_pools[0].counter
    
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
        """Apply slice to parent sequence."""
        seq = parent_seqs[0]
        result = seq[self.key]
        return {'seq_0': result}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'key': self.key,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def seq_slice(
    parent: Pool_type,
    key: Union[int, slice],
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Slice sequences from a pool."""
    op = SeqSliceOp(parent, key=key, name=op_name, op_iteration_order=op_iteration_order)
    result_pool = Pool(operation=op, output_index=0)
    result_pool.iter_order = pool_iteration_order
    if name is not None:
        result_pool.name = name
    return result_pool
