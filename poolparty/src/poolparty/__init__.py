"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.3.0"

from .party import Party, get_active_party
from .pool import Pool
from .operation import Operation
from .generate_seqs import generate_seqs
from .counter import Counter, CounterManager
from .alphabet import get_alphabet, NAMED_ALPHABETS
from .operations import (
    from_seq, FromSeqOp,
    from_seqs, FromSeqsOp,
    get_kmers, GetKmersOp,
    join, JoinOp,
    seq_slice, SeqSliceOp,
    mutation_scan, MutationScanOp,
    breakpoint_scan, BreakpointScanOp,
    insertion_scan,
    replacement_scan,
    deletion_scan,
    stack, StackOp,
    repeat, RepeatOp,
    state_slice, StateSliceOp,
    sync,
)

__all__ = [
    '__version__',
    'Party', 'get_active_party',
    'Pool', 'Operation', 'Counter', 'CounterManager', 'generate_seqs',
    'get_alphabet', 'NAMED_ALPHABETS',
    'from_seq', 'FromSeqOp',
    'from_seqs', 'FromSeqsOp',
    'get_kmers', 'GetKmersOp',
    'join', 'JoinOp',
    'seq_slice', 'SeqSliceOp',
    'mutation_scan', 'MutationScanOp',
    'breakpoint_scan', 'BreakpointScanOp',
    'insertion_scan',
    'replacement_scan',
    'deletion_scan',
    'stack', 'StackOp',
    'repeat', 'RepeatOp',
    'state_slice', 'StateSliceOp',
    'sync',
]
