"""Marker module for poolparty - XML-style region markers for sequence annotation."""

# Import parsing utilities (no dependencies on Operation)
from .parsing import (
    TAG_PATTERN,
    ParsedMarker,
    parse_marker,
    find_all_markers,
    has_marker,
    validate_single_marker,
    strip_all_markers,
    get_length_without_markers,
    get_nonmarker_positions,
    get_literal_positions,
    nonmarker_pos_to_literal_pos,
    build_marker_tag,
)

# Import operation functions (Operation is now available)
from .insert_marker import insert_marker as insert_marker
from .extract_marker_content import extract_marker_content as extract_marker_content
from .replace_marker_content import replace_marker_content as replace_marker_content
from .apply_at_marker import apply_at_marker as apply_at_marker
from .remove_marker import remove_marker as remove_marker
from .marker_scan import marker_scan as marker_scan, MarkerScanOp
from .marker_multiscan import marker_multiscan as marker_multiscan, MarkerMultiScanOp

__all__ = [
    # Parsing utilities
    'TAG_PATTERN',
    'ParsedMarker',
    'parse_marker',
    'find_all_markers',
    'has_marker',
    'validate_single_marker',
    'strip_all_markers',
    'get_length_without_markers',
    'get_nonmarker_positions',
    'get_literal_positions',
    'nonmarker_pos_to_literal_pos',
    'build_marker_tag',
    # Operations
    'insert_marker',
    'extract_marker_content',
    'replace_marker_content',
    'apply_at_marker',
    'remove_marker',
    'marker_scan',
    'marker_multiscan',
    'MarkerScanOp',
    'MarkerMultiScanOp',
]
