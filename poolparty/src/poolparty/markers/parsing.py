"""XML-style marker parsing utilities for poolparty."""
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from poolparty.types import Optional, Literal


@dataclass
class RegionMarker:
    """Represents a parsed XML-style region marker in a sequence."""
    name: str
    start: int          # Position of opening tag '<'
    end: int            # Position after closing tag '>' or '/>'
    content_start: int  # Position of first content character
    content_end: int    # Position after last content character
    strand: str         # '+' or '-'
    content: str        # The sequence content inside the marker
    declared_seq_length: Optional[int]  # None if not declared, int value if declared (including 'None' -> None)
    
    @property
    def is_zero_length(self) -> bool:
        """True if this is a zero-length (self-closing) marker."""
        return self.content_start == self.content_end
    
    @property
    def inferred_seq_length(self) -> int:
        """The actual length of the content (inferred from content)."""
        return len(self.content)
    
    @property
    def is_variable_length(self) -> bool:
        """True if declared_seq_length is explicitly 'None' (variable length marker)."""
        # We use a sentinel: declared_seq_length = -1 means variable length was explicitly declared
        return self.declared_seq_length == -1


def _parse_attributes(attrs_str: str) -> tuple[str, Optional[int]]:
    """Parse strand and seq_length from an attributes string.
    
    Returns:
        (strand, declared_seq_length) where:
        - strand is '+' or '-' (defaults to '+')
        - declared_seq_length is None (not declared), -1 (declared as 'None'), 
          or an int >= 0
    """
    strand = '+'
    declared_seq_length: Optional[int] = None
    
    if not attrs_str:
        return strand, declared_seq_length
    
    # Parse strand attribute
    strand_match = re.search(r"strand=['\"]([+-])['\"]", attrs_str)
    if strand_match:
        strand = strand_match.group(1)
    
    # Parse seq_length attribute
    seq_len_match = re.search(r"seq_length=['\"](\w+)['\"]", attrs_str)
    if seq_len_match:
        value = seq_len_match.group(1)
        if value == 'None':
            declared_seq_length = -1  # Sentinel for variable length
        else:
            try:
                declared_seq_length = int(value)
                if declared_seq_length < 0:
                    raise ValueError(f"seq_length must be non-negative, got {declared_seq_length}")
            except ValueError:
                raise ValueError(f"Invalid seq_length value: '{value}'. Must be an integer or 'None'.")
    
    return strand, declared_seq_length


# Unified regex pattern for all XML-style marker tags
# Captures: (1) closing slash, (2) name, (3) attributes, (4) self-closing slash
# Matches: <name>, <name/>, </name>, <name attr='val'>, <name attr='val'/>, etc.
TAG_PATTERN = re.compile(r'<(/?)(\w+)((?:\s+\w+=[\'"][^\'"]*[\'"])*)\s*(/?)>')

# Alias for backward compatibility (used by strip_all_markers, get_nonmarker_positions)
MARKER_PATTERN = TAG_PATTERN


def find_all_markers(seq: str) -> list[RegionMarker]:
    """Find all markers in a sequence.
    
    Returns a list of RegionMarker objects for each marker found.
    Raises ValueError if markers are malformed (unmatched open/close tags).
    Supports nested markers.
    """
    # Validate structure with stdlib XML parser
    try:
        ET.fromstring(f"<_root_>{seq}</_root_>")
    except ET.ParseError as e:
        raise ValueError(f"Invalid marker syntax: {e}")
    
    markers = []
    open_stack = []  # [(name, strand, declared_seq_length, tag_start, content_start)]
    
    for match in TAG_PATTERN.finditer(seq):
        is_close = match.group(1) == '/'
        name = match.group(2)
        attrs = match.group(3) or ''
        is_self_close = match.group(4) == '/'
        
        if is_self_close:
            # Self-closing tag: <name/>
            strand, decl_len = _parse_attributes(attrs)
            # Validate seq_length for self-closing
            if decl_len is not None and decl_len not in (0, -1):
                raise ValueError(
                    f"Self-closing marker '<{name}/>' has seq_length='{decl_len}' "
                    f"but contains no content. Use seq_length='0' or omit the attribute."
                )
            markers.append(RegionMarker(
                name=name,
                start=match.start(),
                end=match.end(),
                content_start=match.end(),
                content_end=match.end(),
                strand=strand,
                content='',
                declared_seq_length=decl_len,
            ))
        elif is_close:
            # Closing tag: </name> - pop innermost matching open tag
            for i in range(len(open_stack) - 1, -1, -1):
                if open_stack[i][0] == name:
                    oname, strand, decl_len, ostart, cstart = open_stack.pop(i)
                    content = seq[cstart:match.start()]
                    # Validate seq_length if declared
                    if decl_len is not None and decl_len >= 0:
                        if len(content) != decl_len:
                            raise ValueError(
                                f"Marker '<{oname}>' has seq_length='{decl_len}' "
                                f"but content has length {len(content)}: '{content}'"
                            )
                    markers.append(RegionMarker(
                        name=oname,
                        start=ostart,
                        end=match.end(),
                        content_start=cstart,
                        content_end=match.start(),
                        strand=strand,
                        content=content,
                        declared_seq_length=decl_len,
                    ))
                    break
        else:
            # Opening tag: <name>
            strand, decl_len = _parse_attributes(attrs)
            open_stack.append((name, strand, decl_len, match.start(), match.end()))
    
    return sorted(markers, key=lambda m: m.start)


def has_marker(seq: str, name: str) -> bool:
    """Check if a marker with the given name exists in the sequence."""
    markers = find_all_markers(seq)
    return any(m.name == name for m in markers)


def validate_single_marker(seq: str, name: str) -> RegionMarker:
    """Validate that exactly one marker with the given name exists.
    
    Returns the RegionMarker if found.
    Raises ValueError if marker not found or appears multiple times.
    """
    markers = find_all_markers(seq)
    matching = [m for m in markers if m.name == name]
    
    if len(matching) == 0:
        available = [m.name for m in markers]
        if available:
            raise ValueError(
                f"Marker '{name}' not found in sequence. "
                f"Available markers: {available}"
            )
        else:
            raise ValueError(f"Marker '{name}' not found in sequence (no markers present)")
    
    if len(matching) > 1:
        positions = [m.start for m in matching]
        raise ValueError(
            f"Marker '{name}' appears {len(matching)} times in sequence "
            f"(at positions {positions}). Each marker must appear exactly once."
        )
    
    return matching[0]


def parse_marker(seq: str, name: str) -> tuple[str, str, str, str]:
    """Parse a named marker from a sequence.
    
    Returns (prefix, content, suffix, strand) where:
    - prefix: sequence before the marker (excluding the opening tag)
    - content: sequence inside the marker
    - suffix: sequence after the marker (excluding the closing tag)
    - strand: '+' or '-'
    
    Raises ValueError if marker not found or appears multiple times.
    """
    marker = validate_single_marker(seq, name)
    prefix = seq[:marker.start]
    suffix = seq[marker.end:]
    return prefix, marker.content, suffix, marker.strand


def strip_all_markers(seq: str) -> str:
    """Remove all marker tags from sequence, keeping content.
    
    Example:
        'ACG<region>TT</region>AA' -> 'ACGTTAA'
        'ACG<ins/>TT' -> 'ACGTT'
    """
    return TAG_PATTERN.sub('', seq)


def get_length_without_markers(seq: str) -> int:
    """Get sequence length excluding marker tags (but including marker content)."""
    return len(strip_all_markers(seq))


def get_nonmarker_positions(seq: str) -> list[int]:
    """Get raw string positions of all characters excluding marker tag interiors.
    
    Returns positions of characters that are NOT part of marker tags.
    This includes marker content but excludes the <...> tag syntax itself.
    """
    # Find all marker tag spans (the tags themselves, not content)
    tag_spans: set[int] = set()
    for match in TAG_PATTERN.finditer(seq):
        for i in range(match.start(), match.end()):
            tag_spans.add(i)
    
    return [i for i in range(len(seq)) if i not in tag_spans]


def get_literal_positions(seq: str) -> list[int]:
    """Get all raw string positions in a sequence.
    
    Returns list(range(len(seq))). Provided for API completeness
    alongside get_nonmarker_positions and get_biological_positions.
    """
    return list(range(len(seq)))


def nonmarker_pos_to_literal_pos(seq: str, nonmarker_pos: int) -> int:
    """Convert a non-marker position to a literal string position.
    
    Args:
        seq: Sequence string possibly containing markers.
        nonmarker_pos: Position in non-marker coordinate space (0-indexed).
            Can be 0 to len(nonmarker_positions) inclusive, where the
            maximum value represents "one past the end" for slicing.
    
    Returns:
        The corresponding literal string position.
    
    Raises:
        ValueError: If nonmarker_pos is out of range.
    """
    nonmarker_positions = get_nonmarker_positions(seq)
    seq_len = len(nonmarker_positions)
    
    if nonmarker_pos < 0 or nonmarker_pos > seq_len:
        raise ValueError(
            f"nonmarker_pos ({nonmarker_pos}) out of range [0, {seq_len}]"
        )
    
    if nonmarker_pos == seq_len:
        return len(seq)  # One past the end
    return nonmarker_positions[nonmarker_pos]


def build_marker_tag(
    name: str, 
    content: str = '', 
    strand: str = '+',
    seq_length: Optional[int] = None,
) -> str:
    """Build an XML marker tag string.
    
    Args:
        name: Marker name
        content: Content to wrap (empty string for zero-length marker)
        strand: '+' or '-' ('+' is omitted from output for brevity)
        seq_length: Optional declared seq_length (None to omit, -1 for 'None')
    
    Returns:
        XML marker string like '<name>content</name>' or '<name/>'
    """
    attrs = []
    if strand == '-':
        attrs.append("strand='-'")
    if seq_length is not None:
        if seq_length == -1:
            attrs.append("seq_length='None'")
        else:
            attrs.append(f"seq_length='{seq_length}'")
    
    attrs_str = ' ' + ' '.join(attrs) if attrs else ''
    
    if content == '':
        return f"<{name}{attrs_str}/>"
    else:
        return f"<{name}{attrs_str}>{content}</{name}>"


def _validate_markers(seq: str) -> set:
    """
    Parse markers from a sequence string and register/validate with Party.
    
    For each marker found:
    - If seq_length attribute is declared and not 'None': validate content matches
    - If seq_length='None': register as variable-length (seq_length=None)
    - If no seq_length attribute: infer seq_length from content length
    
    This function should be called by any function that takes a user sequence
    string and creates a Pool (like from_seq).
    
    Parameters
    ----------
    seq : str
        The sequence string potentially containing XML markers.
    
    Returns
    -------
    set[Marker]
        Set of Marker objects found in the sequence.
    
    Raises
    ------
    ValueError
        If marker length conflicts with previously registered marker.
    """
    from ..party import get_active_party
    from ..marker import Marker
    
    party = get_active_party()
    if party is None:
        # No active party, return empty set
        return set()
    
    markers_found: set[Marker] = set()
    region_markers = find_all_markers(seq)
    
    for rm in region_markers:
        # Determine the seq_length to register
        if rm.declared_seq_length == -1:
            # Explicitly declared as variable length
            seq_length = None
        elif rm.declared_seq_length is not None:
            # Explicitly declared with a specific length (already validated in parsing)
            seq_length = rm.declared_seq_length
        else:
            # Not declared, infer from content
            seq_length = len(rm.content)
        
        # Register with party (will raise if conflict)
        marker = party.register_marker(rm.name, seq_length)
        markers_found.add(marker)
    
    return markers_found
