"""State operations for poolparty."""
from .stack import stack, StackOp
from .sync import sync
from .state_slice import state_slice, StateSliceOp
from .state_sample import state_sample, StateSampleOp
from .state_shuffle import state_shuffle, StateShuffleOp
from .repeat import repeat, RepeatOp

__all__ = [
    'stack', 'StackOp',
    'sync',
    'state_slice', 'StateSliceOp',
    'state_sample', 'StateSampleOp',
    'state_shuffle', 'StateShuffleOp',
    'repeat', 'RepeatOp',
]
