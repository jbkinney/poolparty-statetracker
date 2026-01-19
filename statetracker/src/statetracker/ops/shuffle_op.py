"""ShuffleOp - Randomly shuffle state values given a seed."""
from ..imports import beartype, Optional, Integral, Sequence, State_type
from ..operation import Operation
import random


@beartype
def shuffle(
    state: State_type,
    seed: Optional[Integral] = None,
    permutation: Optional[Sequence[Integral]] = None,
    name: Optional[str] = None,
):
    """
    Create a State with the values of a parent State randomly shuffled.

    Parameters
    ----------
    state : State_type
        The State whose values will be shuffled.
    seed : Optional[Integral], default=None
        Random seed to generate the shuffle permutation. Cannot be provided with 'permutation'.
    permutation : Optional[Sequence[Integral]], default=None
        Explicit permutation of parent value indices, as a sequence of length parent.num_values. Cannot be provided with 'seed'.
    name : Optional[str], default=None
        Name for the resulting shuffled State.

    Returns
    -------
    State_type
        A State whose value order corresponds to a permutation of the parent's values.
    """
    from ..state import State
    result = State(
        _parents=(state,),
        _op=ShuffleOp(state.num_values, seed=seed, permutation=permutation),
        name=name,
    )
    return result


@beartype
class ShuffleOp(Operation):
    """Randomly shuffle state values using a deterministic seed."""
    
    def __init__(
        self,
        num_parent_values: Integral,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
    ):
        match (seed, permutation):
            case (None, None):
                self.seed = random.randint(0, 2**32 - 1)
                indices = list(range(num_parent_values))
                random.Random(self.seed).shuffle(indices)
                self.permutation = tuple(indices)
            case (_, None):
                self.seed = seed
                indices = list(range(num_parent_values))
                random.Random(seed).shuffle(indices)
                self.permutation = tuple(indices)
            case (None, _):
                if len(permutation) != num_parent_values:
                    raise ValueError(
                        f"permutation has length {len(permutation)}, expected {num_parent_values}."
                    )
                if set(permutation) != set(range(num_parent_values)):
                    raise ValueError(
                        f"permutation must contain exactly the integers 0 to {num_parent_values - 1}."
                    )
                self.seed = None
                self.permutation = tuple(permutation)
            case (_, _):
                raise ValueError("Cannot specify both 'seed' and 'permutation'; they are mutually exclusive.")
    
    def compute_num_states(self, parent_num_values):
        return parent_num_values[0]
    
    def decompose(self, value, parent_num_values):
        if value is None:
            return (None,)
        return (self.permutation[value],)
