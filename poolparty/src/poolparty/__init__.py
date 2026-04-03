"""poolparty - A Python package for designing oligonucleotide sequence libraries."""

__version__ = "0.1.0"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import statetracker as st

# Import other operations from base_ops module
from .base_ops import (
    FilterOp,
    FromIupacOp,
    FromMotifOp,
    FromSeqsOp,
    GetBarcodesOp,
    GetKmersOp,
    MaterializeOp,
    MutagenizeOp,
    FlipOp,
    RecombineOp,
    SeqShuffleOp,
    filter,
    from_iupac,
    from_motif,
    from_seqs,
    get_barcodes,
    get_kmers,
    materialize,
    mutagenize,
    flip,
    recombine,
    shuffle_seq,
)

# Import restriction enzyme data
from .data.restriction_enzymes import (
    ENZYME_PRESETS,
    ENZYME_SITES,
    get_enzyme_site,
    get_preset_enzymes,
)
from .dna_pool import DnaPool

# Import fixed operations from fixed_ops module
from .fixed_ops import (
    AddPrefixOp,
    FixedOp,
    ScoreOp,
    StylizeOp,
    add_prefix,
    clear_annotation,
    clear_gaps,
    fixed_operation,
    from_fasta,
    from_seq,
    join,
    lower,
    rc,
    score,
    slice_seq,
    stylize,
    swapcase,
    upper,
)
from .generate_library import generate_library

# Import from multiscan_ops module
from .multiscan_ops import (
    deletion_multiscan,
    insertion_multiscan,
    replacement_multiscan,
)
from .operation import Operation

# Import ORF operations from orf_ops module
from .orf_ops import MutagenizeOrfOp, StylizeOrfOp, annotate_orf, mutagenize_orf, stylize_orf
from .orf_ops.reverse_translate import ReverseTranslateOp, reverse_translate
from .orf_ops.translate import TranslateOp, translate
from .party import (
    Party,
    _init_default_party,
    clear_pools,
    configure_logging,
    get_active_party,
    init,
    set_genetic_code,
)
from .pool import Pool
from .protein_pool import ProteinPool
from .region import OrfRegion, Region

# Import from region_ops module
from .region_ops import (
    annotate_region,
    apply_at_region,
    extract_region,
    insert_tags,
    region_multiscan,
    region_scan,
    remove_tags,
    replace_region,
)

# Import scan functions from scan_ops module
from .scan_ops import (
    deletion_scan,
    insertion_scan,
    mutagenize_scan,
    replacement_scan,
    shuffle_scan,
    subseq_scan,
)

# Import state operations from state_ops module
from .state_ops import (
    RepeatOp,
    SampleOp,
    StackOp,
    StateShuffleOp,
    StateSliceOp,
    repeat,
    sample,
    stack,
    state_shuffle,
    state_slice,
    sync,
)
from .types import NullSeq, is_null_seq
from .utils.dna_utils import BASES, COMPLEMENT, IGNORE_CHARS, IUPAC_TO_DNA, VALID_CHARS

# Import sequence property functions
from .utils.seq_properties import (
    calc_complexity,
    calc_dust,
    calc_gc,
    has_homopolymer,
    has_restriction_site,
)

# Import styling utilities
from .utils.style_utils import print_named_colors

__all__ = [
    "__version__",
    "Party",
    "get_active_party",
    "init",
    "clear_pools",
    "set_genetic_code",
    "configure_logging",
    "toggle_styles",
    "toggle_cards",
    "set_progress_mode",
    "Pool",
    "DnaPool",
    "ProteinPool",
    "Operation",
    "Region",
    "OrfRegion",
    "State",
    "StateManager",
    "generate_library",
    "BASES",
    "COMPLEMENT",
    "IUPAC_TO_DNA",
    "IGNORE_CHARS",
    "VALID_CHARS",
    "add_prefix",
    "AddPrefixOp",
    "fixed_operation",
    "FixedOp",
    "from_seq",
    "from_fasta",
    "from_seqs",
    "FromSeqsOp",
    "from_iupac",
    "FromIupacOp",
    "from_motif",
    "FromMotifOp",
    "get_barcodes",
    "GetBarcodesOp",
    "get_kmers",
    "GetKmersOp",
    "join",
    "slice_seq",
    "materialize",
    "MaterializeOp",
    "mutagenize",
    "MutagenizeOp",
    "flip",
    "FlipOp",
    "annotate_orf",
    "mutagenize_orf",
    "MutagenizeOrfOp",
    "stylize_orf",
    "StylizeOrfOp",
    "reverse_translate",
    "ReverseTranslateOp",
    "translate",
    "TranslateOp",
    "recombine",
    "RecombineOp",
    "insertion_scan",
    "replacement_scan",
    "deletion_scan",
    "shuffle_scan",
    "mutagenize_scan",
    "subseq_scan",
    "shuffle_seq",
    "SeqShuffleOp",
    "rc",
    "score",
    "ScoreOp",
    "swapcase",
    "upper",
    "lower",
    "clear_gaps",
    "filter",
    "FilterOp",
    "stylize",
    "StylizeOp",
    "stack",
    "StackOp",
    "repeat",
    "RepeatOp",
    "state_slice",
    "StateSliceOp",
    "state_shuffle",
    "StateShuffleOp",
    "sample",
    "SampleOp",
    "sync",
    # Region operations
    "annotate_region",
    "insert_tags",
    "region_scan",
    "region_multiscan",
    "extract_region",
    "replace_region",
    "apply_at_region",
    "remove_tags",
    # Multiscan operations
    "deletion_multiscan",
    "insertion_multiscan",
    "replacement_multiscan",
    # Styling utilities
    "print_named_colors",
    # Sequence property functions
    "calc_gc",
    "calc_complexity",
    "calc_dust",
    "has_homopolymer",
    "has_restriction_site",
    # Restriction enzyme data
    "ENZYME_SITES",
    "ENZYME_PRESETS",
    "get_enzyme_site",
    "get_preset_enzymes",
]

# Re-export statetracker primitives for backward compatibility
State = st.State
StateManager = st.Manager

# Initialize default Party context on import
_init_default_party()


def toggle_styles(on: bool = True) -> None:
    """Toggle inline styling on/off for the active Party.

    When off (on=False), Seq.style will be None to avoid style overhead.
    When on (on=True), normal style tracking is restored.
    """
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party._config.suppress_styles = not on


def toggle_cards(on: bool = True) -> None:
    """Toggle design card computation on/off for the active Party.

    When off (on=False), operations skip building design card data and columns.
    Inline styles are unaffected (controlled by toggle_styles).
    """
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party._config.suppress_cards = not on


def set_progress_mode(mode: str = "auto") -> None:
    """Set progress bar style for the active Party.

    Args:
        mode: 'text' for plain text tqdm bars (terminal-friendly),
              'auto' for tqdm.auto (notebook widget when available).
    """
    from .config import VALID_PROGRESS_MODES

    if mode not in VALID_PROGRESS_MODES:
        raise ValueError(
            f"progress_mode must be one of {VALID_PROGRESS_MODES}, got {mode!r}"
        )
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party._config.progress_mode = mode


# === Copy factory docstrings to Pool methods ===
import re


def _remove_pool_param_from_docstring(docstring: str) -> str:
    """Remove the 'pool' parameter section from a numpy-style docstring."""
    if not docstring:
        return docstring
    # Pattern matches parameter block: "pool : Type\n    description..."
    # Continuation lines must be more indented than the parameter name line (8+ spaces)
    # This prevents matching the next parameter which starts with 4 spaces
    pattern = r"^\s*pool\s*:\s*[^\n]+\n(?:\s{8,}[^\n]*\n)*"
    return re.sub(pattern, "", docstring, flags=re.MULTILINE)


# Map method names to their factory functions - generic methods on Pool
_POOL_FACTORY_MAP = {
    # Common ops (generic)
    "mutagenize": mutagenize,
    "shuffle_seq": shuffle_seq,
    "recombine": recombine,
    "materialize": materialize,
    "filter": filter,
    # Scan ops (generic)
    "mutagenize_scan": mutagenize_scan,
    "deletion_scan": deletion_scan,
    "insertion_scan": insertion_scan,
    "replacement_scan": replacement_scan,
    "shuffle_scan": shuffle_scan,
    # Multiscan ops (generic)
    "deletion_multiscan": deletion_multiscan,
    "insertion_multiscan": insertion_multiscan,
    "replacement_multiscan": replacement_multiscan,
    # Fixed ops (generic)
    "score": score,
    "slice_seq": slice_seq,
    "swapcase": swapcase,
    "upper": upper,
    "lower": lower,
    "clear_gaps": clear_gaps,
    "clear_annotation": clear_annotation,
    "stylize": stylize,
    # State ops (generic)
    "repeat": repeat,
    "sample": sample,
    "shuffle_states": state_shuffle,
    "slice_states": state_slice,
    # Region ops (generic)
    "annotate_region": annotate_region,
    "apply_at_region": apply_at_region,
    "insert_tags": insert_tags,
    "remove_tags": remove_tags,
    "replace_region": replace_region,
    # Generation
    "generate_library": generate_library,
}

# Map method names to their factory functions - DNA-specific methods on DnaPool
_DNAPOOL_FACTORY_MAP = {
    # DNA-specific base ops
    "insert_from_iupac": from_iupac,
    "insert_from_motif": from_motif,
    "insert_kmers": get_kmers,
    # DNA-specific base ops (stateful)
    "flip": flip,
    # DNA-specific fixed ops
    "rc": rc,
    # ORF ops (DNA-specific)
    "annotate_orf": annotate_orf,
    "mutagenize_orf": mutagenize_orf,
    "stylize_orf": stylize_orf,
    "translate": translate,
}

# Copy filtered docstrings from factory functions to Pool methods
for _method_name, _factory_fn in _POOL_FACTORY_MAP.items():
    if hasattr(Pool, _method_name) and _factory_fn.__doc__:
        getattr(Pool, _method_name).__doc__ = _remove_pool_param_from_docstring(_factory_fn.__doc__)

# Copy filtered docstrings from factory functions to DnaPool methods
for _method_name, _factory_fn in _DNAPOOL_FACTORY_MAP.items():
    if hasattr(DnaPool, _method_name) and _factory_fn.__doc__:
        getattr(DnaPool, _method_name).__doc__ = _remove_pool_param_from_docstring(
            _factory_fn.__doc__
        )
