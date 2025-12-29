"""StackOp - Disjoint union of N counters."""
from ..operation import Operation


class StackOp(Operation):    
    def compute_num_states(self, parent_num_states):
        return sum(parent_num_states)
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return tuple(None for _ in parent_num_states)
        cumsum = 0
        for i, n in enumerate(parent_num_states):
            if state < cumsum + n:
                return tuple(
                    state - cumsum if j == i else None
                    for j in range(len(parent_num_states))
                )
            cumsum += n
        raise ValueError(f"Invalid state {state}")


def stack(counters, name=None):
    from ..counter import Counter
    if len(counters) == 0:
        result = Counter(0)
    else:
        for c in counters:
            if not isinstance(c, Counter):
                raise TypeError(f"Expected Counter, got {type(c)}")
        result = Counter(_parents=counters, _op=StackOp(), deduplicate_parents=False, sort_parents=False)
    if name is not None:
        result.name = name
    return result
