"""ShuffleCoOp - Randomly shuffle counter states given a seed."""
from ..counter_operation import CounterOperation
import random


class ShuffleCoOp(CounterOperation):
    """Randomly shuffle counter states using a deterministic seed."""
    
    def __init__(self, seed, num_parent_states):
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


def shuffle_counter(counter, seed=None, name=None):
    """Create a shuffled counter with randomized state order."""
    from ..counter import Counter
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    result = Counter(_parents=(counter,), _op=ShuffleCoOp(seed, counter.num_states))
    if name is not None:
        result.name = name
    return result
