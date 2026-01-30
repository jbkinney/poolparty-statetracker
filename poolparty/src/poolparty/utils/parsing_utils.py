"""XML-style region parsing utilities for poolparty."""
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from poolparty.types import Optional, Literal, Callable, Sequence

# Regex pattern for XML-style region tags (simplified, no attributes)
# 
# Examples:
#   <div>       -> ("", "div", "")
#   </div>      -> ("/", "div", "")
#   <br/>       -> ("", "br", "/")
TAG_PATTERN = re.compile(r'<(/?)(\w+)\s*(/?)>')


@dataclass
class ParsedRegion:
    """Represents a parsed XML-style region in a sequence."""
    name: str
    start: int          # Position of opening tag '<'
    end: int            # Position after closing tag '>' or '/>'
    content_start: int  # Position of first content character
    content_end: int    # Position after last content character
    content: str        # The sequence content inside the region
    
    @property
    def is_zero_length(self) -> bool:
        """True if this is a zero-length (self-closing) region."""
        return self.content_start == self.content_end
    
    @property
    def inferred_seq_length(self) -> int:
        """The actual length of the content (inferred from content)."""
        return len(self.content)


def find_all_regions(seq: str) -> list[ParsedRegion]:
    """Find all regions in a sequence.
    
    Returns a list of ParsedRegion objects for each region found.
    Raises ValueError if regions are malformed (unmatched open/close tags).
    Supports nested regions.
    """
    # Validate structure with stdlib XML parser
    try:
        ET.fromstring(f"<_root_>{seq}</_root_>")
    except ET.ParseError as e:
        raise ValueError(f"Invalid region syntax: {e}")
    
    regions = []
    open_stack = []  # [(name, tag_start, content_start)]
    
    for match in TAG_PATTERN.finditer(seq):
        is_close = match.group(1) == '/'
        name = match.group(2)
        is_self_close = match.group(3) == '/'
        
        if is_self_close:
            # Self-closing tag: <name/>
            regions.append(ParsedRegion(
                name=name,
                start=match.start(),
                end=match.end(),
                content_start=match.end(),
                content_end=match.end(),
                content='',
            ))
        elif is_close:
            # Closing tag: </name> - pop innermost matching open tag
            for i in reversed(range(len(open_stack))):
                if open_stack[i][0] == name:
                    oname, ostart, cstart = open_stack.pop(i)
                    content = seq[cstart:match.start()]
                    regions.append(ParsedRegion(
                        name=oname,
                        start=ostart,
                        end=match.end(),
                        content_start=cstart,
                        content_end=match.start(),
                        content=content,
                    ))
                    break
        else:
            # Opening tag: <name>
            open_stack.append((name, match.start(), match.end()))
    
    return sorted(regions, key=lambda r: r.start)


def has_region(seq: str, name: str) -> bool:
    """Check if a region with the given name exists in the sequence."""
    regions = find_all_regions(seq)
    return any(r.name == name for r in regions)


def validate_single_region_from_list(
    regions: Sequence[ParsedRegion], 
    name: str, 
    seq: str = ""
) -> ParsedRegion:
    """Validate that exactly one region with the given name exists.
    
    Uses pre-parsed regions to avoid redundant parsing.
    
    Parameters
    ----------
    regions : Sequence[ParsedRegion]
        Pre-parsed regions list.
    name : str
        Region name to validate.
    seq : str, default=""
        Optional sequence string for error messages.
    
    Returns
    -------
    ParsedRegion
        The ParsedRegion if found.
    
    Raises
    ------
    ValueError
        If region not found or appears multiple times.
    """
    matching = [r for r in regions if r.name == name]
    
    if len(matching) == 0:
        available = [r.name for r in regions]
        if available:
            raise ValueError(
                f"Region '{name}' not found in sequence. "
                f"Available regions: {available}"
            )
        else:
            raise ValueError(f"Region '{name}' not found in sequence (no regions present)")
    
    if len(matching) > 1:
        positions = [r.start for r in matching]
        raise ValueError(
            f"Region '{name}' appears {len(matching)} times in sequence "
            f"(at positions {positions}). Each region must appear exactly once."
        )
    
    return matching[0]


def validate_single_region(seq: str, name: str) -> ParsedRegion:
    """Validate that exactly one region with the given name exists.
    
    Parameters
    ----------
    seq : str
        Sequence string to parse and validate.
    name : str
        Region name to validate.
    
    Returns
    -------
    ParsedRegion
        The ParsedRegion if found.
    
    Raises
    ------
    ValueError
        If region not found or appears multiple times.
    """
    regions = find_all_regions(seq)
    return validate_single_region_from_list(regions, name, seq)


def parse_region(seq: str, name: str) -> tuple[str, str, str]:
    """Parse a named region from a sequence.
    
    Returns (prefix, content, suffix) where:
    - prefix: sequence before the region (excluding the opening tag)
    - content: sequence inside the region
    - suffix: sequence after the region (excluding the closing tag)
    
    Raises ValueError if region not found or appears multiple times.
    """
    region = validate_single_region(seq, name)
    prefix = seq[:region.start]
    suffix = seq[region.end:]
    return prefix, region.content, suffix


def strip_all_tags(seq: str) -> str:
    """Remove all region tags from sequence, keeping content.
    
    Example:
        'ACG<region>TT</region>AA' -> 'ACGTTAA'
        'ACG<ins/>TT' -> 'ACGTT'
    """
    return TAG_PATTERN.sub('', seq)


def transform_nontag_chars(seq: str, transform_fn: Callable[[str], str]) -> str:
    """Apply a transformation to only non-tag characters in a sequence.
    
    Preserves XML region tags exactly as they appear while transforming
    all other characters using the provided function.
    
    Example:
        transform_nontag_chars('ACgt<region>TT</region>', str.lower)
        -> 'acgt<region>tt</region>'
    """
    result = []
    last_end = 0
    for match in TAG_PATTERN.finditer(seq):
        # Transform text before this tag
        result.append(transform_fn(seq[last_end:match.start()]))
        # Keep the tag unchanged
        result.append(match.group(0))
        last_end = match.end()
    # Transform remaining text after last tag
    result.append(transform_fn(seq[last_end:]))
    return ''.join(result)


def get_length_without_tags(seq: str) -> int:
    """Get sequence length excluding region tags (but including region content)."""
    return len(strip_all_tags(seq))


def get_nontag_positions(seq: str) -> list[int]:
    """Get raw string positions of all characters excluding region tag interiors.
    
    Returns positions of characters that are NOT part of region tags.
    This includes region content but excludes the <...> tag syntax itself.
    """
    # Fast path: no tags possible if no '<' character
    if '<' not in seq:
        return list(range(len(seq)))
    
    # Find all region tag spans (the tags themselves, not content)
    tag_spans: set[int] = set()
    for match in TAG_PATTERN.finditer(seq):
        for i in range(match.start(), match.end()):
            tag_spans.add(i)
    
    return [i for i in range(len(seq)) if i not in tag_spans]


def get_literal_positions(seq: str) -> list[int]:
    """Get all raw string positions in a sequence.
    
    Returns list(range(len(seq))). Provided for API completeness
    alongside get_nontag_positions and get_molecular_positions.
    """
    return list(range(len(seq)))


def nontag_pos_to_literal_pos(seq: str, nontag_pos: int) -> int:
    """Convert a non-tag position to a literal string position.
    
    Args:
        seq: Sequence string possibly containing regions.
        nontag_pos: Position in non-tag coordinate space (0-indexed).
            Can be 0 to len(nontag_positions) inclusive, where the
            maximum value represents "one past the end" for slicing.
    
    Returns:
        The corresponding literal string position.
    
    Raises:
        ValueError: If nontag_pos is out of range.
    """
    nontag_positions = get_nontag_positions(seq)
    seq_len = len(nontag_positions)
    
    if nontag_pos < 0 or nontag_pos > seq_len:
        raise ValueError(
            f"nontag_pos ({nontag_pos}) out of range [0, {seq_len}]"
        )
    
    if nontag_pos == seq_len:
        return len(seq)  # One past the end
    return nontag_positions[nontag_pos]


def build_region_tags(name: str, content: str = '') -> str:
    """Build an XML region tag string.
    
    Args:
        name: Region name
        content: Content to wrap (empty string for zero-length region)
    
    Returns:
        XML region string like '<name>content</name>' or '<name/>'
    """
    if content == '':
        return f"<{name}/>"
    else:
        return f"<{name}>{content}</{name}>"


def _validate_regions(seq: str) -> set:
    """
    Parse regions from a sequence string and register/validate with Party.
    
    For each region found, infer seq_length from content length (excluding nested region tags).
    
    This function should be called by any function that takes a user sequence
    string and creates a Pool (like from_seq).
    
    Parameters
    ----------
    seq : str
        The sequence string potentially containing XML regions.
    
    Returns
    -------
    set[Region]
        Set of Region objects found in the sequence.
    
    Raises
    ------
    ValueError
        If region length conflicts with previously registered region.
    """
    from ..party import get_active_party
    from ..region import Region
    
    party = get_active_party()
    if party is None:
        # No active party, return empty set
        return set()
    
    regions_found: set[Region] = set()
    parsed_regions = find_all_regions(seq)
    
    for pr in parsed_regions:
        # Infer seq_length from content (excluding nested region tags)
        seq_length = get_length_without_tags(pr.content)
        
        # Register with party (will raise if conflict)
        region = party.register_region(pr.name, seq_length)
        regions_found.add(region)
    
    return regions_found
