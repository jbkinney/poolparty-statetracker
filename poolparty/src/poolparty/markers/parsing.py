"""XML-style marker parsing utilities for poolparty."""
import re
from dataclasses import dataclass
from typing import Optional, Literal


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


# Regex patterns for XML-style markers
# Matches: <name>, <name strand='+'>, <name seq_length='6'>, <name strand='-' seq_length='6'>, etc.
_OPENING_TAG = r"<(\w+)((?:\s+(?:strand|seq_length)=['\"][^'\"]+['\"])*)\s*>"
# Matches: </name>
_CLOSING_TAG = r"</(\w+)>"
# Matches: <name/>, <name strand='+'/>, <name seq_length='0'/>, etc.
_SELF_CLOSING_TAG = r"<(\w+)((?:\s+(?:strand|seq_length)=['\"][^'\"]+['\"])*)\s*/>"

# Combined pattern to find any marker tag (for stripping/length calculations)
MARKER_PATTERN = re.compile(
    rf"(?:{_SELF_CLOSING_TAG})|(?:{_OPENING_TAG})|(?:{_CLOSING_TAG})"
)

# Compiled patterns for parsing
_OPENING_PATTERN = re.compile(_OPENING_TAG)
_CLOSING_PATTERN = re.compile(_CLOSING_TAG)
_SELF_CLOSING_PATTERN = re.compile(_SELF_CLOSING_TAG)


def find_all_markers(seq: str) -> list[RegionMarker]:
    """Find all markers in a sequence.
    
    Returns a list of RegionMarker objects for each marker found.
    Raises ValueError if markers are malformed (unmatched open/close tags).
    """
    markers = []
    
    # First, find all self-closing markers
    for match in _SELF_CLOSING_PATTERN.finditer(seq):
        name = match.group(1)
        attrs_str = match.group(2) or ''
        strand, declared_seq_length = _parse_attributes(attrs_str)
        
        # For self-closing markers, validate seq_length if declared
        if declared_seq_length is not None and declared_seq_length != 0 and declared_seq_length != -1:
            raise ValueError(
                f"Self-closing marker '<{name}/>' has seq_length='{declared_seq_length}' "
                f"but contains no content. Use seq_length='0' or omit the attribute."
            )
        
        markers.append(RegionMarker(
            name=name,
            start=match.start(),
            end=match.end(),
            content_start=match.end(),  # No content
            content_end=match.end(),
            strand=strand,
            content='',
            declared_seq_length=declared_seq_length,
        ))
    
    # Find all opening tags that aren't self-closing
    open_tags = []
    for match in _OPENING_PATTERN.finditer(seq):
        # Skip if this position is inside a self-closing marker we already found
        if any(m.start <= match.start() < m.end for m in markers):
            continue
        attrs_str = match.group(2) or ''
        strand, declared_seq_length = _parse_attributes(attrs_str)
        open_tags.append((match.group(1), strand, declared_seq_length, match.start(), match.end()))
    
    # Find all closing tags
    close_tags = []
    for match in _CLOSING_PATTERN.finditer(seq):
        # Skip if inside a self-closing marker
        if any(m.start <= match.start() < m.end for m in markers):
            continue
        close_tags.append((match.group(1), match.start(), match.end()))
    
    # Match opening tags with closing tags
    used_closes = set()
    for open_name, strand, declared_seq_length, open_start, open_end in open_tags:
        # Find the first unused closing tag with matching name after this opening tag
        found = False
        for i, (close_name, close_start, close_end) in enumerate(close_tags):
            if i in used_closes:
                continue
            if close_name == open_name and close_start > open_end:
                used_closes.add(i)
                content = seq[open_end:close_start]
                
                # Validate content length against declared seq_length
                if declared_seq_length is not None and declared_seq_length >= 0:
                    if len(content) != declared_seq_length:
                        raise ValueError(
                            f"Marker '<{open_name}>' has seq_length='{declared_seq_length}' "
                            f"but content has length {len(content)}: '{content}'"
                        )
                
                markers.append(RegionMarker(
                    name=open_name,
                    start=open_start,
                    end=close_end,
                    content_start=open_end,
                    content_end=close_start,
                    strand=strand,
                    content=content,
                    declared_seq_length=declared_seq_length,
                ))
                found = True
                break
        if not found:
            raise ValueError(f"Unmatched opening tag '<{open_name}>' at position {open_start}")
    
    # Check for unmatched closing tags
    for i, (close_name, close_start, close_end) in enumerate(close_tags):
        if i not in used_closes:
            raise ValueError(f"Unmatched closing tag '</{close_name}>' at position {close_start}")
    
    # Sort by start position
    markers.sort(key=lambda m: m.start)
    return markers


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
    # Remove self-closing tags first
    result = _SELF_CLOSING_PATTERN.sub('', seq)
    # Remove opening and closing tags
    result = _OPENING_PATTERN.sub('', result)
    result = _CLOSING_PATTERN.sub('', result)
    return result


def get_length_without_markers(seq: str) -> int:
    """Get sequence length excluding marker tags (but including marker content)."""
    return len(strip_all_markers(seq))


def get_positions_without_markers(seq: str) -> list[int]:
    """Get raw string positions of all characters excluding marker tag interiors.
    
    Returns positions of characters that are NOT part of marker tags.
    This includes marker content but excludes the <...> tag syntax itself.
    """
    # Find all marker tag spans (the tags themselves, not content)
    tag_spans: set[int] = set()
    
    # Self-closing tags: entire match is a tag
    for match in _SELF_CLOSING_PATTERN.finditer(seq):
        for i in range(match.start(), match.end()):
            tag_spans.add(i)
    
    # Opening tags
    for match in _OPENING_PATTERN.finditer(seq):
        # Skip if inside self-closing marker
        if match.start() in tag_spans:
            continue
        for i in range(match.start(), match.end()):
            tag_spans.add(i)
    
    # Closing tags
    for match in _CLOSING_PATTERN.finditer(seq):
        if match.start() in tag_spans:
            continue
        for i in range(match.start(), match.end()):
            tag_spans.add(i)
    
    return [i for i in range(len(seq)) if i not in tag_spans]


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
