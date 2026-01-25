"""Utility modules for poolparty."""

# Re-export from submodules for convenient access
from .utils import validate_iteration_order, clean_df_int_columns
from .style_utils import (
    STYLE_CODES, CSS_COLORS, ANSI_ESCAPE_PATTERN, DEFAULT_GAP_CHARS,
    reset, validate_style_positions, split_styles_by_region,
    shift_style_positions, reassemble_styles, apply_inline_styles,
    print_named_colors,
)
from .orf_utils import validate_orf_extent
from .dna_utils import (
    BASES, COMPLEMENT, IUPAC_TO_DNA, IGNORE_CHARS, VALID_CHARS, IUPAC_CHARS,
    complement, reverse_complement, get_mutations,
    get_nonmarker_positions, get_length_without_markers,
    get_molecular_positions, get_seq_length,
)
from .df_utils import (
    counter_col_name, get_pools_reverse_topo,
    organize_columns, finalize_generate_df,
)

__all__ = [
    # utils
    'validate_iteration_order', 'clean_df_int_columns',
    # style_utils
    'STYLE_CODES', 'CSS_COLORS', 'ANSI_ESCAPE_PATTERN', 'DEFAULT_GAP_CHARS',
    'reset', 'validate_style_positions', 'split_styles_by_region',
    'shift_style_positions', 'reassemble_styles', 'apply_inline_styles',
    'print_named_colors',
    # orf_utils
    'validate_orf_extent',
    # dna_utils
    'BASES', 'COMPLEMENT', 'IUPAC_TO_DNA', 'IGNORE_CHARS', 'VALID_CHARS', 'IUPAC_CHARS',
    'complement', 'reverse_complement', 'get_mutations',
    'get_nonmarker_positions', 'get_length_without_markers',
    'get_molecular_positions', 'get_seq_length',
    # df_utils
    'counter_col_name', 'get_pools_reverse_topo',
    'organize_columns', 'finalize_generate_df',
]
