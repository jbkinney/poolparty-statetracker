"""StateSample operation - sample states from a pool."""
from numbers import Real
import statecounter as sc
from ..types import Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_sample(
    pool: Pool,
    num_states: Optional[Integral] = None,
    sampled_states: Optional[Sequence[Integral]] = None,
    seed: Optional[Integral] = None,
    with_replacement: bool = True,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with sampled states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool to sample states from.
    num_states : Optional[Integral], default=None
        Number of states to sample. Mutually exclusive with sampled_states.
    sampled_states : Optional[Sequence[Integral]], default=None
        Explicit list of states to sample. Mutually exclusive with num_states.
    seed : Optional[Integral], default=None
        Random seed for deterministic sampling. Only used with num_states.
    with_replacement : bool, default=True
        If False, num_states must be <= pool.num_states (no duplicates).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying state sample Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing the sampled states from the input Pool.
    """
    op = StateSampleOp(
        pool,
        num_states=num_states,
        sampled_states=sampled_states,
        seed=seed,
        with_replacement=with_replacement,
        name=op_name,
        iter_order=op_iter_order,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool


@beartype
class StateSampleOp(Operation):
    """Sample states from a pool."""
    factory_name = "state_sample"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool: Pool,
        num_states: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateSampleOp."""
        self._num_states = num_states
        self.sampled_states = sampled_states
        self.seed = seed
        self.with_replacement = with_replacement
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
        """Build pool counter using sc.sample."""
        return sc.sample(
            parent_pools[0].counter,
            num_states=self._num_states,
            sampled_states=self.sampled_states,
            seed=self.seed,
            with_replacement=self.with_replacement,
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
            'num_states': self._num_states,
            'sampled_states': self.sampled_states,
            'seed': self.seed,
            'with_replacement': self.with_replacement,
            'name': None,
            'iter_order': self.iter_order,
        }
