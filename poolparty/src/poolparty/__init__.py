"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.3.0"

import statecounter as sc

from .party import Party, get_active_party, reset_default_party, _init_default_party
from .pool import Pool
from .operation import Operation
from .marker import Marker
from .generate_seqs import generate_seqs
from .alphabet import get_alphabet, NAMED_ALPHABETS
from .operations import (
    from_seq, FromSeqOp,
    from_seqs, FromSeqsOp,
    from_iupac_motif, FromIupacMotifOp,
    from_prob_motif, FromProbMotifOp,
    get_kmers, GetKmersOp,
    join, JoinOp,
    seq_slice, SeqSliceOp,
    mutagenize, MutagenizeOp,
    mutagenize_orf, MutagenizeOrfOp,
    breakpoint_scan, BreakpointScanOp,
    marker_scan, MarkerScanOp,
    marker_multiscan, MarkerMultiScanOp,
    replace_marker, ReplaceMarkerOp,
    insertion_scan,
    replacement_scan,
    deletion_scan,
    seq_shuffle, SeqShuffleOp,
    reverse_complement, ReverseComplementOp,
    stack, StackOp,
    repeat, RepeatOp,
    state_slice, StateSliceOp,
    state_shuffle, StateShuffleOp,
    state_sample, StateSampleOp,
    sync,
)

__all__ = [
    '__version__',
    'Party', 'get_active_party', 'reset_default_party',
    'set_default', 'load_defaults',
    'Pool', 'Operation', 'Marker', 'Counter', 'CounterManager', 'generate_seqs',
    'get_alphabet', 'NAMED_ALPHABETS',
    'from_seq', 'FromSeqOp',
    'from_seqs', 'FromSeqsOp',
    'from_iupac_motif', 'FromIupacMotifOp',
    'from_prob_motif', 'FromProbMotifOp',
    'get_kmers', 'GetKmersOp',
    'join', 'JoinOp',
    'seq_slice', 'SeqSliceOp',
    'mutagenize', 'MutagenizeOp',
    'mutagenize_orf', 'MutagenizeOrfOp',
    'breakpoint_scan', 'BreakpointScanOp',
    'marker_scan', 'MarkerScanOp',
    'marker_multiscan', 'MarkerMultiScanOp',
    'replace_marker', 'ReplaceMarkerOp',
    'insertion_scan',
    'replacement_scan',
    'deletion_scan',
    'seq_shuffle', 'SeqShuffleOp',
    'reverse_complement', 'ReverseComplementOp',
    'stack', 'StackOp',
    'repeat', 'RepeatOp',
    'state_slice', 'StateSliceOp',
    'state_shuffle', 'StateShuffleOp',
    'state_sample', 'StateSampleOp',
    'sync',
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
