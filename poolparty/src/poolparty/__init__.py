"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.3.0"

import statetracker as st

from .party import Party, get_active_party, init, clear_pools, _init_default_party
from .pool import Pool
from .operation import Operation
from .region import Region
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
    clear_annotation,
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
    recombine, RecombineOp,
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
# Import from region_ops module
from .region_ops import (
    insert_tags,
    region_scan,
    region_multiscan,
    extract_region,
    replace_region,
    apply_at_region,
    remove_tags,
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
    'set_default', 'load_defaults', 'toggle_styles', 'toggle_cards',
    'Pool', 'Operation', 'Region', 'State', 'StateManager', 'generate_library',
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
    'recombine', 'RecombineOp',
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
    # Region operations
    'insert_tags',
    'region_scan',
    'region_multiscan',
    'extract_region',
    'replace_region',
    'apply_at_region',
    'remove_tags',
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


def toggle_styles(on: bool = True) -> None:
    """Toggle inline styling on/off for the active Party.
    
    When off (on=False), Seq.style will be None to avoid style overhead.
    When on (on=True), normal style tracking is restored.
    """
    get_active_party().set_default('suppress_styles', not on)


def toggle_cards(on: bool = True) -> None:
    """Toggle design card computation on/off for the active Party.
    
    When off (on=False), operations skip building design card data.
    Inline styles are unaffected (controlled by toggle_styles).
    """
    get_active_party().set_default('suppress_cards', not on)


# === Copy factory docstrings to Pool methods ===
import re

def _remove_pool_param_from_docstring(docstring: str) -> str:
    """Remove the 'pool' parameter section from a numpy-style docstring."""
    if not docstring:
        return docstring
    # Pattern matches parameter block: "pool : Type\n    description..."
    # Continuation lines must be more indented than the parameter name line (8+ spaces)
    # This prevents matching the next parameter which starts with 4 spaces
    pattern = r'^\s*pool\s*:\s*[^\n]+\n(?:\s{8,}[^\n]*\n)*'
    return re.sub(pattern, '', docstring, flags=re.MULTILINE)

# Map Pool method names to their factory functions
_POOL_FACTORY_MAP = {
    # Base ops
    'mutagenize': mutagenize,
    'shuffle_seq': shuffle_seq,
    'insert_from_iupac': from_iupac,
    'insert_from_motif': from_motif,
    'insert_kmers': get_kmers,
    'recombine': recombine,
    # Scan ops
    'mutagenize_scan': mutagenize_scan,
    'deletion_scan': deletion_scan,
    'insertion_scan': insertion_scan,
    'replacement_scan': replacement_scan,
    'shuffle_scan': shuffle_scan,
    # Fixed ops
    'rc': rc,
    'swapcase': swapcase,
    'upper': upper,
    'lower': lower,
    'clear_gaps': clear_gaps,
    'clear_annotation': clear_annotation,
    'stylize': stylize,
    # State ops
    'repeat_states': repeat,
    'sample_states': state_sample,
    'shuffle_states': state_shuffle,
    'slice_states': state_slice,
    # Region ops
    'apply_at_region': apply_at_region,
    'insert_tags': insert_tags,
    'remove_tags': remove_tags,
    'replace_region': replace_region,
    # Generation
    'generate_library': generate_library,
}

# Copy filtered docstrings from factory functions to Pool methods
for _method_name, _factory_fn in _POOL_FACTORY_MAP.items():
    if hasattr(Pool, _method_name) and _factory_fn.__doc__:
        getattr(Pool, _method_name).__doc__ = _remove_pool_param_from_docstring(_factory_fn.__doc__)
