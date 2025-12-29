"""ProductOp - Cartesian product of N counters."""
from ..operation import Operation


class ProductOp(Operation):
    """C = A * B * ... : Cartesian product of N counters."""
    
    def compute_num_states(self, parent_num_states):
        result = 1
        for n in parent_num_states:
            result *= n
        return result
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return tuple(None for _ in parent_num_states)
        result = []
        for n in parent_num_states:
            result.append(state % n)
            state //= n
        return tuple(result)


def product(counters, name=None):
    """Create product counter from 0 or more counters."""
    from ..counter import Counter
    if len(counters) == 0:
        result = Counter(1)
    else:
        for c in counters:
            if not isinstance(c, Counter):
                raise TypeError(f"Expected Counter, got {type(c)}")
        result = Counter(_parents=counters, _op=ProductOp())
    if name is not None:
        result.name = name
    return result
