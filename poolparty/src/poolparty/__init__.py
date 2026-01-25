"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.3.0"

import statetracker as st

from .party import Party, get_active_party, init, clear_pools, _init_default_party
from .pool import Pool
from .operation import Operation
from .marker import Marker
from .generate_library import generate_library
from .utils.dna_utils import BASES, COMPLEMENT, IUPAC_TO_DNA, IGNORE_CHARS, VALID_CHARS
# Import fixed operations from fixed_ops module
from .fixed_ops import (
    fixed_operation, FixedOp,
    from_seq,
    from_fasta,
    join,
    slice_seq,
    rc,
    swapcase,
    upper,
    lower,
    clear_gaps,
    stylize, StylizeOp,
)
# Import other operations from base_ops module
from .base_ops import (
    from_seqs, FromSeqsOp,
    from_iupac, FromIupacOp,
    from_motif, FromMotifOp,
    get_kmers, GetKmersOp,
    mutagenize, MutagenizeOp,
    shuffle_seq, SeqShuffleOp,
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
# Import styling utilities
from .utils.style_utils import print_named_colors
# Import from multiscan_ops module
from .multiscan_ops import (
    deletion_multiscan,
    insertion_multiscan,
    replacement_multiscan,
)

__all__ = [
    '__version__',
    'Party', 'get_active_party', 'init', 'clear_pools',
    'set_default', 'load_defaults',
    'Pool', 'Operation', 'Marker', 'State', 'StateManager', 'generate_library',
    'BASES', 'COMPLEMENT', 'IUPAC_TO_DNA', 'IGNORE_CHARS', 'VALID_CHARS',
    'fixed_operation', 'FixedOp',
    'from_seq',
    'from_fasta',
    'from_seqs', 'FromSeqsOp',
    'from_iupac', 'FromIupacOp',
    'from_motif', 'FromMotifOp',
    'get_kmers', 'GetKmersOp',
    'join',
    'slice_seq',
    'mutagenize', 'MutagenizeOp',
    'mutagenize_orf', 'MutagenizeOrfOp',
    'insertion_scan',
    'replacement_scan',
    'deletion_scan',
    'shuffle_scan',
    'mutagenize_scan',
    'subseq_scan',
    'shuffle_seq', 'SeqShuffleOp',
    'rc',
    'swapcase',
    'upper',
    'lower',
    'clear_gaps',
    'stylize', 'StylizeOp',
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
    # Styling utilities
    'print_named_colors',
]

# Re-export statetracker primitives for backward compatibility
State = st.State
StateManager = st.Manager

# Initialize default Party context on import
_init_default_party()


def set_default(key: str, value) -> None:
    """Set a default parameter on the active Party."""
    if key == 'iter_order':
        st.set_product_order_mode(value)
    get_active_party().set_default(key, value)


def load_defaults(filepath: str) -> None:
    """Load default parameters from a TOML file into the active Party."""
    get_active_party().load_defaults(filepath)
