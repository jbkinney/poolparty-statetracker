"""poolparty - A Python package for generating oligonucleotide sequence pools.

This package provides a function-based API for creating and manipulating
oligonucleotide sequences with lazy evaluation and combinatorial iteration.

Example:
    >>> from poolparty import from_seqs, get_kmers, mutation_scan
    >>> 
    >>> # Create a pool from sequences
    >>> variants = from_seqs(['AAA', 'TTT', 'GGG'])
    >>> 
    >>> # Generate k-mers
    >>> barcode = get_kmers(length=10, alphabet='dna')
    >>> 
    >>> # Apply k mutations
    >>> mutants = mutation_scan('ACGTACGT', k=1, alphabet='dna')
    >>> 
    >>> # Compose pools
    >>> library = mutants + '...' + barcode
"""

__version__ = "0.2.0"

# Core classes
from .pool import Pool, MultiPool, OutputSelectorOp
from .operation import Operation

# Factory functions (the primary API)
from .operations import (
    # Core generators
    from_seqs,
    FromSeqsOp,
    from_motif,
    FromMotifOp,
    get_kmers,
    GetKmersOp,
    flip_flop,
    FlipFlopOp,
    mix,
    MixOp,
    from_iupac,
    FromIupacOp,
    filter,
    FilterOp,
    # get_barcodes_op,  # STASHED
    
    # Simple transformers
    # subseq_op,  # STASHED
    # shuffle_op,  # STASHED
    # mutagenize_op,  # STASHED
    mutation_scan,
    MutationScanOp,
    
    # Scan operations
    subseq_scan,
    SubseqScanOp,
    breakpoint_scan,
    BreakpointScanOp,
    # deletion_scan_op,  # STASHED
    # insertion_scan_op,  # STASHED
    # shuffle_scan_op,  # STASHED
    # multiinsertion_scan_op,  # STASHED
    
    # Composition operations
    concatenate,
    ConcatenateOp,
    repeat,
    RepeatOp,
    subseq,
    SliceOp,
)

# Utilities
from .utils import reset_op_id_counter
from .alphabet import (
    validate_alphabet,
    get_alphabet,
    named_alphabets_dict,
)

__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'Pool',
    'MultiPool',
    'OutputSelectorOp',
    'Operation',
    
    # Core generators
    'from_seqs',
    'FromSeqsOp',
    'from_motif',
    'FromMotifOp',
    'flip_flop',
    'FlipFlopOp',
    'get_kmers',
    'GetKmersOp',
    'mix',
    'MixOp',
    'from_iupac',
    'FromIupacOp',
    'filter',
    'FilterOp',
    # 'get_barcodes_op',  # STASHED
    
    # Simple transformers
    # 'subseq_op',  # STASHED
    # 'shuffle_op',  # STASHED
    # 'mutagenize_op',  # STASHED
    'mutation_scan',
    'MutationScanOp',
    
    # Scan operations
    'subseq_scan',
    'SubseqScanOp',
    'breakpoint_scan',
    'BreakpointScanOp',
    # 'deletion_scan_op',  # STASHED
    # 'insertion_scan_op',  # STASHED
    # 'shuffle_scan_op',  # STASHED
    # 'multiinsertion_scan_op',  # STASHED
    
    # Composition operations
    'concatenate',
    'ConcatenateOp',
    'repeat',
    'RepeatOp',
    'subseq',
    'SliceOp',
    
    # Utilities
    'validate_alphabet',
    'get_alphabet',
    'named_alphabets_dict',
    'reset_op_id_counter',
]
