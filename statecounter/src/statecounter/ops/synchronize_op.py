"""SynchronizeOp - Keep N counters in lockstep."""
from ..imports import beartype, Sequence, Optional, Counter_type
from ..operation import Operation


@beartype
class SyncOp(Operation):
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


@beartype
def sync(counters: Sequence[Counter_type], name: Optional[str] = None):
    """Create sync counter from 0 or more counters."""
    from ..counter import Counter
    if len(counters) == 0:
        result = Counter(1)
    else:
        result = Counter(_parents=counters, _op=SyncOp())
    if name is not None:
        result.name = name
    return result
