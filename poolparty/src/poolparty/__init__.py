"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.3.0"

import statecounter as sc

from .party import Party, get_active_party, init, _init_default_party
from .pool import Pool
from .operation import Operation
from .marker import Marker
from .generate_library import generate_library
from .alphabet import get_alphabet, NAMED_ALPHABETS
# Import fixed operations from fixed_ops module
from .fixed_ops import (
    fixed_operation, FixedOp,
    from_seq,
    join,
    seq_slice,
    reverse_complement,
    swap_case,
    upper,
    lower,
    clear_nonmolecular_chars,
    clear_ignore_chars,
)
# Import other operations from base_ops module
from .base_ops import (
    from_seqs, FromSeqsOp,
    from_iupac_motif, FromIupacMotifOp,
    from_prob_motif, FromProbMotifOp,
    get_kmers, GetKmersOp,
    mutagenize, MutagenizeOp,
    breakpoint_scan, BreakpointScanOp,
    seq_shuffle, SeqShuffleOp,
)
# Import ORF operations from orf_ops module
from .orf_ops import (
    mutagenize_orf, MutagenizeOrfOp
)
# Import state operations from state_ops module
from .state_ops import (
    stack, StackOp,
    sync,
    state_slice, StateSliceOp,
    state_sample, StateSampleOp,
    state_shuffle, StateShuffleOp,
    repeat, RepeatOp,
)
# Import scan functions from scan_ops module
from .scan_ops import (
    insertion_scan,
    replacement_scan,
    deletion_scan,
    shuffle_scan,
    mutagenize_scan,
    subseq_scan,
)
# Import from marker_ops module
from .marker_ops import (
    insert_marker,
    marker_scan,
    marker_multiscan,
    extract_marker_content,
    replace_marker_content,
    apply_at_marker,
    remove_marker,
)
# Import from multiscan_ops module
from .multiscan_ops import (
    deletion_multiscan,
    insertion_multiscan,
    replacement_multiscan,
)

__all__ = [
    '__version__',
    'Party', 'get_active_party', 'init',
    'set_default', 'load_defaults',
    'Pool', 'Operation', 'Marker', 'Counter', 'CounterManager', 'generate_library',
    'get_alphabet', 'NAMED_ALPHABETS',
    'fixed_operation', 'FixedOp',
    'from_seq',
    'from_seqs', 'FromSeqsOp',
    'from_iupac_motif', 'FromIupacMotifOp',
    'from_prob_motif', 'FromProbMotifOp',
    'get_kmers', 'GetKmersOp',
    'join',
    'seq_slice',
    'mutagenize', 'MutagenizeOp',
    'mutagenize_orf', 'MutagenizeOrfOp',
    'breakpoint_scan', 'BreakpointScanOp',
    'insertion_scan',
    'replacement_scan',
    'deletion_scan',
    'shuffle_scan',
    'mutagenize_scan',
    'subseq_scan',
    'seq_shuffle', 'SeqShuffleOp',
    'reverse_complement',
    'swap_case',
    'upper',
    'lower',
    'clear_nonmolecular_chars',
    'clear_ignore_chars',
    'stack', 'StackOp',
    'repeat', 'RepeatOp',
    'state_slice', 'StateSliceOp',
    'state_shuffle', 'StateShuffleOp',
    'state_sample', 'StateSampleOp',
    'sync',
    # Marker operations
    'insert_marker',
    'marker_scan',
    'marker_multiscan',
    'extract_marker_content',
    'replace_marker_content',
    'apply_at_marker',
    'remove_marker',
    # Multiscan operations
    'deletion_multiscan',
    'insertion_multiscan',
    'replacement_multiscan',
]

# Re-export statecounter primitives for backward compatibility
Counter = sc.Counter
CounterManager = sc.Manager

# Initialize default Party context on import
_init_default_party()


def set_default(key: str, value) -> None:
    """Set a default parameter on the active Party."""
    get_active_party().set_default(key, value)


def load_defaults(filepath: str) -> None:
    """Load default parameters from a TOML file into the active Party."""
    get_active_party().load_defaults(filepath)
