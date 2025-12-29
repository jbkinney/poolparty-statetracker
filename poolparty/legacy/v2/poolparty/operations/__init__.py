"""Operations for poolparty.

Each operation is a factory function that returns a Pool object.
"""

# Core generators
from .from_seqs_op import from_seqs, FromSeqsOp
from .get_kmers_op import get_kmers, GetKmersOp
from .from_motif_op import from_motif, FromMotifOp
from .mix_op import mix, MixOp
from .from_iupac_op import from_iupac, FromIupacOp
from .filter_op import filter, FilterOp
# from .get_barcodes_op import get_barcodes_op, GetBarcodesOp  # STASHED

# Simple transformers
# from .subseq_op import subseq_op, SubseqOp  # STASHED
# from .shuffle_op import shuffle_op, ShuffleOp  # STASHED
# from .mutagenize_op import mutagenize_op, MutagenizeOp  # STASHED
from .mutation_scan_op import mutation_scan, MutationScanOp
from .flip_flop_op import flip_flop, FlipFlopOp

# Scan operations
from .subseq_scan_op import subseq_scan, SubseqScanOp
from .window_scan_op import window_scan, WindowScanOp, SlotOp, shuffle_scan, deletion_scan, insertion_scan
from .breakpoint_scan_op import breakpoint_scan, BreakpointScanOp
# from .deletion_scan_op import deletion_scan_op, DeletionScanOp  # STASHED
# from .insertion_scan_op import insertion_scan_op, InsertionScanOp  # STASHED
# from .shuffle_scan_op import shuffle_scan_op, ShuffleScanOp  # STASHED
# from .multiinsertion_scan_op import multiinsertion_scan_op, MultiinsertionScanOp  # STASHED

# ORF operations
# from ..orf_operation import ORFOp  # STASHED
# from ..orf_operations.orf_mutation_scan_op import orf_mutation_scan_op, ORFMutationScanOp  # STASHED

# Composition operations
from .concatenate_op import concatenate, ConcatenateOp
from .repeat_op import repeat, RepeatOp
from .slice_op import subseq, SliceOp

__all__ = [
    # Core generators
    'from_seqs', 'FromSeqsOp',
    'get_kmers', 'GetKmersOp',
    'from_motif', 'FromMotifOp',
    'mix', 'MixOp',
    'from_iupac', 'FromIupacOp',
    'filter', 'FilterOp',
    # 'get_barcodes_op', 'GetBarcodesOp',  # STASHED
    
    # Simple transformers
    # 'subseq_op', 'SubseqOp',  # STASHED
    # 'shuffle_op', 'ShuffleOp',  # STASHED
    # 'mutagenize_op', 'MutagenizeOp',  # STASHED
    'mutation_scan', 'MutationScanOp',
    'flip_flop', 'FlipFlopOp',
    
    # Scan operations
    'subseq_scan', 'SubseqScanOp',
    'window_scan', 'WindowScanOp', 'SlotOp', 'shuffle_scan', 'deletion_scan', 'insertion_scan',
    'breakpoint_scan', 'BreakpointScanOp',
    # 'deletion_scan_op', 'DeletionScanOp',  # STASHED
    # 'insertion_scan_op', 'InsertionScanOp',  # STASHED
    # 'shuffle_scan_op', 'ShuffleScanOp',  # STASHED
    # 'multiinsertion_scan_op', 'MultiinsertionScanOp',  # STASHED
    
    # ORF operations
    # 'ORFOp',  # STASHED
    # 'orf_mutation_scan_op', 'ORFMutationScanOp',  # STASHED
    
    # Composition operations
    'concatenate', 'ConcatenateOp',
    'repeat', 'RepeatOp',
    'subseq', 'SliceOp',
]
