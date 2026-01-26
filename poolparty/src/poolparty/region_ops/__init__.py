"""Region module for poolparty - XML-style region tags for sequence annotation."""

# Import parsing utilities (no dependencies on Operation)
from .parsing import (
    TAG_PATTERN,
    ParsedRegion,
    parse_region,
    find_all_regions,
    has_region,
    validate_single_region,
    strip_all_tags,
    get_length_without_tags,
    get_nontag_positions,
    get_literal_positions,
    nontag_pos_to_literal_pos,
    build_region_tags,
)

# Import operation functions (Operation is now available)
from .insert_tags import insert_tags as insert_tags
from .extract_region import extract_region as extract_region
from .replace_region import replace_region as replace_region
from .apply_at_region import apply_at_region as apply_at_region
from .remove_tags import remove_tags as remove_tags
from .region_scan import region_scan as region_scan, RegionScanOp
from .region_multiscan import region_multiscan as region_multiscan, RegionMultiScanOp

__all__ = [
    # Parsing utilities
    'TAG_PATTERN',
    'ParsedRegion',
    'parse_region',
    'find_all_regions',
    'has_region',
    'validate_single_region',
    'strip_all_tags',
    'get_length_without_tags',
    'get_nontag_positions',
    'get_literal_positions',
    'nontag_pos_to_literal_pos',
    'build_region_tags',
    # Operations
    'insert_tags',
    'extract_region',
    'replace_region',
    'apply_at_region',
    'remove_tags',
    'region_scan',
    'region_multiscan',
    'RegionScanOp',
    'RegionMultiScanOp',
]
