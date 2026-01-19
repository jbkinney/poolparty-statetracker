"""SampleOp - Sample values from a parent state."""
from ..imports import beartype, Optional, Integral, Sequence, State_type
from ..operation import Operation
import random


@beartype
def sample(
    state: State_type,
    num_values: Optional[Integral] = None,
    sampled_states: Optional[Sequence[Integral]] = None,
    seed: Optional[Integral] = None,
    with_replacement: bool = True,
    name: Optional[str] = None,
):
    """
    Create a State that samples values from the provided parent State.

    Parameters
    ----------
    state : State_type
        Parent State whose values will be sampled.
    num_values : Optional[Integral], default=None
        Number of values to sample from the parent. Mutually exclusive with 'sampled_states'.
    sampled_states : Optional[Sequence[Integral]], default=None
        Explicit sequence of parent value indices to use as samples. Mutually exclusive with 'num_values'.
    seed : Optional[Integral], default=None
        Random seed for sampling. Only relevant if sampling with 'num_values' and not supplying 'sampled_states'.
    with_replacement : bool, default=True
        Whether to sample with replacement (True) or without replacement (False).
    name : Optional[str], default=None
        Name for the resulting sampled State.

    Returns
    -------
    State_type
        A State whose values are a sampled subset of those of the parent State.
    """
    from ..state import State
    return State(
        _parents=(state,),
        _op=SampleOp(
            state.num_values,
            num_values=num_values,
            sampled_states=sampled_states,
            seed=seed,
            with_replacement=with_replacement,
        ),
        name=name,
    )


@beartype
class SampleOp(Operation):
    """Sample values from parent state."""
    
    def __init__(
        self,
        num_parent_values: Integral,
        num_values: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
    ):
        # Validate mutually exclusive args
        match (num_values, sampled_states):
            case (None, None):
                raise ValueError("Must specify either 'num_values' or 'sampled_states'.")
            case (_, None):
                # Sample num_values from parent using seed
                if not with_replacement and num_values > num_parent_values:
                    raise ValueError(
                        f"num_values ({num_values}) exceeds parent.num_values ({num_parent_values}) "
                        f"and with_replacement=False."
                    )
                self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
                rng = random.Random(self.seed)
                if with_replacement:
                    self.sampled_states = tuple(rng.choices(range(num_parent_values), k=num_values))
                else:
                    self.sampled_states = tuple(rng.sample(range(num_parent_values), k=num_values))
            case (None, _):
                # Explicit sampled_states provided
                if seed is not None:
                    raise ValueError("Cannot specify 'seed' with 'sampled_states'.")
                # Validate states are in valid range
                for s in sampled_states:
                    if s < 0 or s >= num_parent_values:
                        raise ValueError(f"Value {s} out of range [0, {num_parent_values}).")
                self.seed = None
                self.sampled_states = tuple(sampled_states)
            case (_, _):
                raise ValueError("Cannot specify both 'num_values' and 'sampled_states'.")
    
    def compute_num_states(self, parent_num_values):
        return len(self.sampled_states)
    
    def decompose(self, value, parent_num_values):
        if value is None:
            return (None,)
        return (self.sampled_states[value],)
