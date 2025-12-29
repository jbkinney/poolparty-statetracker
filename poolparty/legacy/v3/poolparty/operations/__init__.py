"""Operations for poolparty.

This module provides factory functions that create Pools from Operations.
"""

from .from_seqs import from_seqs, FromSeqsOp
from .get_kmers import get_kmers, GetKmersOp
from .concatenate import concatenate, ConcatenateOp
from .mutation_scan import mutation_scan, MutationScanOp
from .breakpoint_scan import breakpoint_scan, BreakpointScanOp
from .slice_op import subseq, SliceOp

__all__ = [
    'from_seqs', 'FromSeqsOp',
    'get_kmers', 'GetKmersOp',
    'concatenate', 'ConcatenateOp',
    'mutation_scan', 'MutationScanOp',
    'breakpoint_scan', 'BreakpointScanOp',
    'subseq', 'SliceOp',
]

