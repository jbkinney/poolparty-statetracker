"""StateShuffle operation - randomly permute a pool's states."""
from numbers import Real
import statetracker as st
from ..types import Optional, Sequence, Integral, Real, beartype, SeqStyle
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_shuffle(
    pool: Pool,
    seed: Optional[Integral] = None,
    permutation: Optional[Sequence[Integral]] = None,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with randomly permuted states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool whose states will be shuffled.
    seed : Optional[Integral], default=None
        Random seed for deterministic shuffling. If None, a random seed is generated.
    permutation : Optional[Sequence[Integral]], default=None
        Custom permutation to use. If provided, seed must not be specified.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool containing the same states as the input but in a randomly permuted order.
    """
    op = StateShuffleOp(pool, seed=seed, permutation=permutation, prefix=prefix, name=None, iter_order=iter_order)
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class StateShuffleOp(Operation):
    """Randomly permute a pool's states."""
    factory_name = "state_shuffle"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool: Pool,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateShuffleOp."""
        self.seed = seed
        self.permutation = permutation
        super().__init__(
            parent_pools=[parent_pool],
            num_values=1,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool counter using st.shuffle."""
        return st.shuffle(
            parent_pools[0].state,
            seed=self.seed,
            permutation=self.permutation,
        )
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[SeqStyle] | None = None,
    ) -> dict:
        """Return parent sequence (state mapping handled by counter)."""
        seq = parent_seqs[0]
        output_style = SeqStyle.from_parent(parent_styles, 0, len(seq))
        return {'seq': seq, 'style': output_style}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'seed': self.seed,
            'permutation': self.permutation,
            'prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
