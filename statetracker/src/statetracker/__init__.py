"""StateTracker - Composable states with unidirectional value propagation."""
from .manager import Manager
from .operation import Operation
from .ops import (
    ProductOp,
    StackOp,
    SliceOp,
    RepeatOp,
    ShuffleOp,
    SampleOp,
    InterleaveOp,
    PassthroughOp,
    product,
    ordered_product,
    set_product_order_mode,
    get_product_order_mode,
    stack,
    sync,
    slice,
    repeat,
    shuffle,
    sample,
    split,
    interleave,
    passthrough,
    synced_to,
)
from .state import State, ConflictingValueAssignmentError

__version__ = "0.1.0"

__all__ = [
    'State', 'Manager', 'Operation', 'ConflictingValueAssignmentError',
    'ProductOp', 'StackOp', 'SliceOp',
    'RepeatOp', 'ShuffleOp', 'SampleOp', 'InterleaveOp', 'PassthroughOp',
    'product', 'ordered_product', 'set_product_order_mode', 'get_product_order_mode',
    'stack', 'sync', 'slice', 'repeat', 'shuffle', 'sample',
    'split', 'interleave', 'passthrough', 'synced_to',
]
