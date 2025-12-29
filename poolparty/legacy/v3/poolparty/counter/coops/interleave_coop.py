"""
InterleaveCoOp - Interleave states from N counters.
"""
from ..counter_operation import CounterOperation


class InterleaveCoOp(CounterOperation):
    """Interleave states from N counters with equal num_states.
    
    Unlike SumCoOp which visits all states of A, then all of B, etc.,
    InterleaveCoOp alternates: A[0], B[0], C[0], A[1], B[1], C[1], ...
    """
    
    def compute_num_states(self, parent_num_states):
        if len(set(parent_num_states)) != 1:
            raise ValueError(
                f"Cannot interleave counters with different num_states: {parent_num_states}"
            )
        return parent_num_states[0] * len(parent_num_states)
    
    def decompose(self, state, parent_num_states):
        if state == -1:
            return tuple(-1 for _ in parent_num_states)
        k = len(parent_num_states)
        active_idx = state % k
        parent_state = state // k
        return tuple(parent_state if i == active_idx else -1 
                     for i in range(k))


def interleave_counters(*counters, name=None):
    """Create an interleaved counter from multiple counters.
    
    Given counters A, B, C with equal num_states=N, creates a counter with
    N*k total states that alternates between them:
    - State 0: A=0, B=-1, C=-1
    - State 1: A=-1, B=0, C=-1
    - State 2: A=-1, B=-1, C=0
    - State 3: A=1, B=-1, C=-1
    - ...
    
    Args:
        *counters: Two or more Counter objects (must have same num_states).
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter that interleaves all parent states.
    
    Examples:
        A = Counter(3, name='A')
        B = Counter(3, name='B')
        I = interleave_counters(A, B, name='I')
        for _ in I:
            print(f"A={A.state}, B={B.state}")
        # Prints: A=0,B=-1 then A=-1,B=0 then A=1,B=-1 then A=-1,B=1 ...
    """
    from ..counter import Counter
    
    if len(counters) < 2:
        raise ValueError("interleave_counters() requires at least 2 counters")
    for c in counters:
        if not isinstance(c, Counter):
            raise TypeError(f"Expected Counter, got {type(c)}")
    result = Counter(_parents=counters, _op=InterleaveCoOp())
    if name is not None:
        result.name = name
    return result
