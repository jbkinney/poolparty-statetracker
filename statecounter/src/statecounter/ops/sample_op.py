"""SampleOp - Sample states from a parent counter."""
from ..imports import beartype, Optional, Integral, Sequence, Counter_type
from ..operation import Operation
import random


@beartype
class SampleOp(Operation):
    """Sample states from parent counter."""
    
    def __init__(
        self,
        num_parent_states: Integral,
        num_states: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
    ):
        # Validate mutually exclusive args
        match (num_states, sampled_states):
            case (None, None):
                raise ValueError("Must specify either 'num_states' or 'sampled_states'.")
            case (_, None):
                # Sample num_states from parent using seed
                if not with_replacement and num_states > num_parent_states:
                    raise ValueError(
                        f"num_states ({num_states}) exceeds parent.num_states ({num_parent_states}) "
                        f"and with_replacement=False."
                    )
                self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
                rng = random.Random(self.seed)
                if with_replacement:
                    self.sampled_states = tuple(rng.choices(range(num_parent_states), k=num_states))
                else:
                    self.sampled_states = tuple(rng.sample(range(num_parent_states), k=num_states))
            case (None, _):
                # Explicit sampled_states provided
                if seed is not None:
                    raise ValueError("Cannot specify 'seed' with 'sampled_states'.")
                # Validate states are in valid range
                for s in sampled_states:
                    if s < 0 or s >= num_parent_states:
                        raise ValueError(f"State {s} out of range [0, {num_parent_states}).")
                self.seed = None
                self.sampled_states = tuple(sampled_states)
            case (_, _):
                raise ValueError("Cannot specify both 'num_states' and 'sampled_states'.")
    
    def compute_num_states(self, parent_num_states):
        return len(self.sampled_states)
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return (None,)
        return (self.sampled_states[state],)


@beartype
def sample(
    counter: Counter_type,
    num_states: Optional[Integral] = None,
    sampled_states: Optional[Sequence[Integral]] = None,
    seed: Optional[Integral] = None,
    with_replacement: bool = True,
    name: Optional[str] = None,
):
    """Sample states from a counter.
    
    Args:
        counter: The parent counter to sample from.
        num_states: Number of states to sample (mutually exclusive with sampled_states).
        sampled_states: Explicit list of states to sample (mutually exclusive with num_states).
        seed: Random seed for reproducibility (only used with num_states).
        with_replacement: If False, num_states must be <= parent.num_states. Default True.
        name: Optional name for the new counter.
    
    Returns:
        A new Counter with sampled states from the parent.
    """
    from ..counter import Counter
    return Counter(
        _parents=(counter,),
        _op=SampleOp(
            counter.num_states,
            num_states=num_states,
            sampled_states=sampled_states,
            seed=seed,
            with_replacement=with_replacement,
        ),
        name=name,
    )
