"""StateCounter - Composable counters with unidirectional state propagation."""
from .manager import Manager
from .operation import Operation
from .ops import (
    ProductOp,
    StackOp,
    SyncOp,
    SliceOp,
    RepeatOp,
    ShuffleOp,
    InterleaveOp,
    PassthroughOp,
    product,
    ordered_product,
    stack,
    sync,
    slice,
    repeat,
    shuffle,
    split,
    interleave,
    passthrough,
)
from .counter import Counter, ConflictingStateAssignmentError

__version__ = "0.1.0"

__all__ = [
    'Counter', 'Manager', 'Operation', 'ConflictingStateAssignmentError',
    'ProductOp', 'StackOp', 'SyncOp', 'SliceOp',
    'RepeatOp', 'ShuffleOp', 'InterleaveOp', 'PassthroughOp',
    'product', 'ordered_product', 'stack', 'sync',
    'slice', 'repeat', 'shuffle',
    'split', 'interleave', 'passthrough',
]
