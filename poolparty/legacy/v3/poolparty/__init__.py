"""poolparty - A Python package for designing oligonucleotide sequence libraries.

Usage:
    import poolparty as pp
    
    with pp.Party() as party:
        seq = pp.from_seqs(['ACGTACGT'])
        mutants = pp.mutation_scan(seq, k=1)
        barcode = pp.get_kmers(length=10)
        oligo = mutants + '...' + barcode
        party.output(oligo, name='oligo')
    
    df = party.generate(num_seqs=100, seed=42)
"""

__version__ = "0.3.0"

# Core classes
from .party import Party, get_active_party
from .pool import Pool
from .operation import Operation, reset_op_id_counter
from .counter import Counter, CounterManager

# Alphabet utilities
from .alphabet import get_alphabet, NAMED_ALPHABETS

# Operations (factory functions)
from .operations import (
    from_seqs, FromSeqsOp,
    get_kmers, GetKmersOp,
    concatenate, ConcatenateOp,
    mutation_scan, MutationScanOp,
    breakpoint_scan, BreakpointScanOp,
    subseq, SliceOp,
)

__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'Party',
    'get_active_party',
    'Pool',
    'Operation',
    'reset_op_id_counter',
    'Counter',
    'CounterManager',
    
    # Alphabet
    'get_alphabet',
    'NAMED_ALPHABETS',
    
    # Operations
    'from_seqs', 'FromSeqsOp',
    'get_kmers', 'GetKmersOp',
    'concatenate', 'ConcatenateOp',
    'mutation_scan', 'MutationScanOp',
    'breakpoint_scan', 'BreakpointScanOp',
    'subseq', 'SliceOp',
]

