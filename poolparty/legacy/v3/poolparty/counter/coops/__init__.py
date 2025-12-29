"""
Counter operations (CoOps) for composing counters.

This package contains all concrete CounterOperation subclasses
and their factory functions.
"""
from .multiply_coop import MultiplyCoOp, multiply_counters
from .sum_coop import SumCoOp, sum_counters
from .synchronize_coop import SynchronizeCoOp, synchronize_counters
from .slice_coop import SliceCoOp, slice_counter
from .repeat_coop import RepeatCoOp, repeat_counter
from .shuffle_coop import ShuffleCoOp, shuffle_counter
from .split_coop import split_counter
from .interleave_coop import InterleaveCoOp, interleave_counters

__all__ = [
    'MultiplyCoOp', 'multiply_counters',
    'SumCoOp', 'sum_counters',
    'SynchronizeCoOp', 'synchronize_counters',
    'SliceCoOp', 'slice_counter',
    'RepeatCoOp', 'repeat_counter',
    'ShuffleCoOp', 'shuffle_counter',
    'split_counter',
    'InterleaveCoOp', 'interleave_counters',
]

