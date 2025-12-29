"""ShuffleOp - Randomly shuffle counter states given a seed."""
from ..imports import beartype, Optional, Integral, Counter_type
from ..operation import Operation
import random


@beartype
class ShuffleOp(Operation):
    """Randomly shuffle counter states using a deterministic seed."""
    
    def __init__(self, seed: Integral, num_parent_states: Integral):
        self.seed = seed
        indices = list(range(num_parent_states))
        random.Random(seed).shuffle(indices)
        self.permutation = tuple(indices)
    
    def compute_num_states(self, parent_num_states):
        return parent_num_states[0]
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return (None,)
        return (self.permutation[state],)


@beartype
def shuffle(counter: Counter_type, seed: Optional[Integral] = None, name: Optional[str] = None):
    """Create a shuffled counter with randomized state order."""
    from ..counter import Counter
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    result = Counter(_parents=(counter,), _op=ShuffleOp(seed, counter.num_states))
    if name is not None:
        result.name = name
    return result
