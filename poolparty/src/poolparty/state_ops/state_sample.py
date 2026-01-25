"""StateSample operation - sample states from a pool."""
from numbers import Real
import statetracker as st
from ..types import Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_sample(
    pool: Pool,
    num_values: Optional[Integral] = None,
    sampled_states: Optional[Sequence[Integral]] = None,
    seed: Optional[Integral] = None,
    with_replacement: bool = True,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with sampled states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool to sample states from.
    num_values : Optional[Integral], default=None
        Number of states to sample. Mutually exclusive with sampled_states.
    sampled_states : Optional[Sequence[Integral]], default=None
        Explicit list of states to sample. Mutually exclusive with num_values.
    seed : Optional[Integral], default=None
        Random seed for deterministic sampling. Only used with num_values.
    with_replacement : bool, default=True
        If False, num_values must be <= pool.num_states (no duplicates).
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool containing the sampled states from the input Pool.
    """
    op = StateSampleOp(
        pool,
        num_values=num_values,
        sampled_states=sampled_states,
        seed=seed,
        with_replacement=with_replacement,
        prefix=prefix,
        name=None,
        iter_order=iter_order,
    )
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class StateSampleOp(Operation):
    """Sample states from a pool."""
    factory_name = "state_sample"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool: Pool,
        num_values: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateSampleOp."""
        self._num_values = num_values
        self.sampled_states = sampled_states
        self.seed = seed
        self.with_replacement = with_replacement
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
        """Build pool counter using st.sample."""
        return st.sample(
            parent_pools[0].state,
            num_values=self._num_values,
            sampled_states=self.sampled_states,
            seed=self.seed,
            with_replacement=self.with_replacement,
        )
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list | None = None,
    ) -> dict:
        """Return parent sequence (state mapping handled by counter)."""
        output_styles = parent_styles[0] if parent_styles and len(parent_styles) > 0 else []
        return {'seq': parent_seqs[0], 'style': output_styles}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_values': self._num_values,
            'sampled_states': self.sampled_states,
            'seed': self.seed,
            'with_replacement': self.with_replacement,
            'prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
