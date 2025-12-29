"""
ShuffleCoOp - Randomly shuffle counter states given a seed.
"""
import random
from ..counter_operation import CounterOperation


class ShuffleCoOp(CounterOperation):
    """Randomly shuffle counter states using a deterministic seed."""
    
    def __init__(self, seed, num_parent_states):
        self.seed = seed
        # Generate permutation at construction time
        indices = list(range(num_parent_states))
        random.Random(seed).shuffle(indices)
        self.permutation = tuple(indices)
    
    def compute_num_states(self, parent_num_states):
        return parent_num_states[0]  # Same as parent
    
    def decompose(self, state, parent_num_states):
        if state == -1:
            return (-1,)
        return (self.permutation[state],)


def shuffle_counter(counter, seed=None, name=None):
    """Create a shuffled counter with randomized state order.
    
    Creates a new counter that visits the same states as the parent counter
    but in a randomly shuffled order determined by the seed.
    
    Args:
        counter: The Counter to shuffle.
        seed: Random seed for reproducibility. If None, uses a random seed.
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter with the same num_states but shuffled order.
    
    Examples:
        A = Counter(5)
        B = shuffle_counter(A, seed=42)  # Shuffled order, reproducible
        for _ in B:
            print(A.state)  # Prints states in shuffled order
    """
    from ..counter import Counter
    
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    result = Counter(_parents=(counter,), _op=ShuffleCoOp(seed, counter.num_states))
    if name is not None:
        result.name = name
    return result

