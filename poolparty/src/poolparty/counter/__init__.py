"""Counter module with unidirectional state propagation."""
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
    PassthroughCoOp,
    multiply_counters,
    sum_counters,
    synchronize_counters,
    slice_counter,
    repeat_counter,
    shuffle_counter,
    split_counter,
    interleave_counters,
    passthrough_counter,
)
from .counter import Counter, ConflictingStateAssignmentError

__all__ = [
    'Counter', 'CounterManager', 'CounterOperation', 'ConflictingStateAssignmentError',
    'MultiplyCoOp', 'SumCoOp', 'SynchronizeCoOp', 'SliceCoOp',
    'RepeatCoOp', 'ShuffleCoOp', 'InterleaveCoOp', 'PassthroughCoOp',
    'multiply_counters', 'sum_counters', 'synchronize_counters',
    'slice_counter', 'repeat_counter', 'shuffle_counter',
    'split_counter', 'interleave_counters', 'passthrough_counter',
]
