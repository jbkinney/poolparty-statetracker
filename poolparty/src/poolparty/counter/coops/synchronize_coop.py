"""SynchronizeCoOp - Keep N counters in lockstep."""
from ..counter_operation import CounterOperation


class SynchronizeCoOp(CounterOperation):
    """Keep N counters in lockstep."""
    
    def compute_num_states(self, parent_num_states):
        if len(parent_num_states) == 0:
            return 1
        if len(set(parent_num_states)) != 1:
            raise ValueError(
                f"Cannot sync counters with different num_states: {parent_num_states}"
            )
        return parent_num_states[0]
    
    def decompose(self, state, parent_num_states):
        return tuple(state for _ in parent_num_states)


def synchronize_counters(*counters, name=None):
    """Create sync counter from 0 or more counters."""
    from ..counter import Counter
    if len(counters) == 0:
        result = Counter(1)
    else:
        for c in counters:
            if not isinstance(c, Counter):
                raise TypeError(f"Expected Counter, got {type(c)}")
        result = Counter(_parents=counters, _op=SynchronizeCoOp())
    if name is not None:
        result.name = name
    return result
