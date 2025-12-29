"""
Counter module with unidirectional state propagation.

Counters can be composed via product (*), sum (+), and sync operations.
State flows from composite counters DOWN to their parent counters only.
"""
from .counter_manager import CounterManager
from .counter_operation import CounterOperation
from .coops import (
    MultiplyCoOp,
    SumCoOp,
    SynchronizeCoOp,
    SliceCoOp,
    RepeatCoOp,
    ShuffleCoOp,
    InterleaveCoOp,
    multiply_counters,
    sum_counters,
    synchronize_counters,
    slice_counter,
    repeat_counter,
    shuffle_counter,
    split_counter,
    interleave_counters,
)
from .counter import Counter

__all__ = [
    'Counter',
    'CounterManager',
    'CounterOperation',
    'MultiplyCoOp',
    'SumCoOp',
    'SynchronizeCoOp',
    'SliceCoOp',
    'RepeatCoOp',
    'ShuffleCoOp',
    'InterleaveCoOp',
    'multiply_counters',
    'sum_counters',
    'synchronize_counters',
    'slice_counter',
    'repeat_counter',
    'shuffle_counter',
    'split_counter',
    'interleave_counters',
]
