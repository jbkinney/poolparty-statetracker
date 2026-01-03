"""StateShuffle operation - randomly permute a pool's states."""
from numbers import Real
import statecounter as sc
from ..types import Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_shuffle(
    pool: Pool,
    seed: Optional[Integral] = None,
    permutation: Optional[Sequence[Integral]] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying state shuffle Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing the same states as the input but in a randomly permuted order.
    """
    op = StateShuffleOp(pool, seed=seed, permutation=permutation, name=op_name, iter_order=op_iter_order)
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
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
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateShuffleOp."""
        self.seed = seed
        self.permutation = permutation
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            name=name,
            iter_order=iter_order,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> sc.Counter:
        """Build pool counter using sc.shuffle."""
        return sc.shuffle(
            parent_pools[0].counter,
            seed=self.seed,
            permutation=self.permutation,
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
        """Return the parent sequence (state mapping handled by counter)."""
        return {'seq_0': parent_seqs[0]}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'seed': self.seed,
            'permutation': self.permutation,
            'name': None,
            'iter_order': self.iter_order,
        }
