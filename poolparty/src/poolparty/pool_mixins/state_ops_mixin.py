"""State operation mixins for Pool class."""
from ..types import Pool_type, Optional, Real, Integral, Sequence, Union, beartype


class StateOpsMixin:
    """Mixin providing state operation methods for Pool."""
    
    def repeat_states(
        self,
        times: Integral,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..state_ops.repeat import repeat
        return repeat(
            pool=self,
            times=times,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def sample_states(
        self,
        num_values: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..state_ops.state_sample import state_sample
        return state_sample(
            pool=self,
            num_values=num_values,
            sampled_states=sampled_states,
            seed=seed,
            with_replacement=with_replacement,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def shuffle_states(
        self,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..state_ops.state_shuffle import state_shuffle
        return state_shuffle(
            pool=self,
            seed=seed,
            permutation=permutation,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def slice_states(
        self,
        key: Union[Integral, slice],
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..state_ops.state_slice import state_slice
        return state_slice(
            pool=self,
            key=key,
            prefix=prefix,
            iter_order=iter_order,
        )
