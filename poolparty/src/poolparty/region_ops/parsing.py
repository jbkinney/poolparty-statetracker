"""XML-style region parsing utilities for poolparty."""
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from poolparty.types import Optional, Literal, Callable

# Regex pattern for XML-style region tags
# 
# Example: <item id="5" type='main'/>
#          │ │   │                  │
#          │ │   │                  └─ Group 4: "/" (self-closing slash, or empty)
#          │ │   └─ Group 3: " id=\"5\" type='main'" (all attributes, or empty)
#          │ └─ Group 2: "item" (tag name)
#          └─ Group 1: "" (closing slash, or empty if opening tag)
#
# Other examples:
#   <div>       -> ("", "div", "", "")
#   </div>      -> ("/", "div", "", "")
#   <br/>       -> ("", "br", "", "/")
#   <a href="x"> -> ("", "a", " href=\"x\"", "")
TAG_PATTERN = re.compile(r'<(/?)(\w+)((?:\s+\w+=[\'"][^\'"]*[\'"])*)\s*(/?)>')


@dataclass
class ParsedRegion:
    """Represents a parsed XML-style region in a sequence."""
    name: str
    start: int          # Position of opening tag '<'
    end: int            # Position after closing tag '>' or '/>'
    content_start: int  # Position of first content character
    content_end: int    # Position after last content character
    strand: str         # '+' or '-'
    content: str        # The sequence content inside the region
    declared_seq_length_str: Optional[str]  # None if not declared, 'None' if variable, '4' if declared as 4
    
    @property
    def is_zero_length(self) -> bool:
        """True if this is a zero-length (self-closing) region."""
        return self.content_start == self.content_end
    
    @property
    def inferred_seq_length(self) -> int:
        """The actual length of the content (inferred from content)."""
        return len(self.content)
    
    @property
    def is_variable_length(self) -> bool:
        """True if seq_length was explicitly declared as 'None' (variable length region)."""
        return self.declared_seq_length_str == 'None'


def _parse_attributes(attrs_str: str) -> tuple[str, Optional[str]]:
    """Parse strand and seq_length from an attributes string.
    
    Returns:
        (strand, declared_seq_length_str) where:
        - strand is '+' or '-' (defaults to '+')
        - declared_seq_length_str is None (not declared), 'None' (variable length),
          or the string representation of the declared int (e.g., '4')
    """
    strand = '+'
    declared_seq_length_str: Optional[str] = None
    
    if not attrs_str:
        return strand, declared_seq_length_str
    
    # Parse strand attribute
    strand_match = re.search(r"strand=['\"]([+-])['\"]", attrs_str)
    if strand_match:
        strand = strand_match.group(1)
    
    # Parse seq_length attribute
    seq_len_match = re.search(r"seq_length=['\"](\w+)['\"]", attrs_str)
    if seq_len_match:
        value = seq_len_match.group(1)
        if value == 'None':
            declared_seq_length_str = 'None'
        else:
            try:
                parsed_int = int(value)
                if parsed_int < 0:
                    raise ValueError(f"seq_length must be non-negative, got {parsed_int}")
                declared_seq_length_str = value  # Store the string representation
            except ValueError:
                raise ValueError(f"Invalid seq_length value: '{value}'. Must be an integer in quotes or 'None'.")
    
    return strand, declared_seq_length_str


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
    open_stack = []  # [(name, strand, declared_seq_length_str, tag_start, content_start)]
    
    for match in TAG_PATTERN.finditer(seq):
        is_close = match.group(1) == '/'
        name = match.group(2)
        attrs = match.group(3) or ''
        is_self_close = match.group(4) == '/'
        
        if is_self_close:
            # Self-closing tag: <name/>
            strand, declared_seq_length_str = _parse_attributes(attrs)
            # Validate seq_length for self-closing
            if declared_seq_length_str is not None and declared_seq_length_str not in ('0', 'None'):
                raise ValueError(
                    f"Self-closing region '<{name}/>' has seq_length='{declared_seq_length_str}' "
                    f"but contains no content. Use seq_length='0' or omit the attribute."
                )
            regions.append(ParsedRegion(
                name=name,
                start=match.start(),
                end=match.end(),
                content_start=match.end(),
                content_end=match.end(),
                strand=strand,
                content='',
                declared_seq_length_str=declared_seq_length_str,
            ))
        elif is_close:
            # Closing tag: </name> - pop innermost matching open tag
            for i in reversed(range(len(open_stack))):
                if open_stack[i][0] == name:
                    oname, strand, declared_seq_length_str, ostart, cstart = open_stack.pop(i)
                    content = seq[cstart:match.start()]
                    # Validate seq_length if declared as an integer
                    if declared_seq_length_str is not None and declared_seq_length_str != 'None':
                        if len(content) != int(declared_seq_length_str):
                            raise ValueError(
                                f"Region '<{oname}>' has seq_length='{declared_seq_length_str}' "
                                f"but content has length {len(content)}: '{content}'"
                            )
                    regions.append(ParsedRegion(
                        name=oname,
                        start=ostart,
                        end=match.end(),
                        content_start=cstart,
                        content_end=match.start(),
                        strand=strand,
                        content=content,
                        declared_seq_length_str=declared_seq_length_str,
                    ))
                    break
        else:
            # Opening tag: <name>
            strand, declared_seq_length_str = _parse_attributes(attrs)
            open_stack.append((name, strand, declared_seq_length_str, match.start(), match.end()))
    
    return sorted(regions, key=lambda r: r.start)


def has_region(seq: str, name: str) -> bool:
    """Check if a region with the given name exists in the sequence."""
    regions = find_all_regions(seq)
    return any(r.name == name for r in regions)


def validate_single_region(seq: str, name: str) -> ParsedRegion:
    """Validate that exactly one region with the given name exists.
    
    Returns the ParsedRegion if found.
    Raises ValueError if region not found or appears multiple times.
    """
    regions = find_all_regions(seq)
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


def parse_region(seq: str, name: str) -> tuple[str, str, str, str]:
    """Parse a named region from a sequence.
    
    Returns (prefix, content, suffix, strand) where:
    - prefix: sequence before the region (excluding the opening tag)
    - content: sequence inside the region
    - suffix: sequence after the region (excluding the closing tag)
    - strand: '+' or '-'
    
    Raises ValueError if region not found or appears multiple times.
    """
    region = validate_single_region(seq, name)
    prefix = seq[:region.start]
    suffix = seq[region.end:]
    return prefix, region.content, suffix, region.strand


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


def reverse_complement_with_tags(
    seq: str, 
    complement_fn: Callable[[str], str]
) -> str:
    """Reverse complement a sequence while preserving XML region tag structure.
    
    Regions are repositioned based on their content coordinates:
    - Region tags [start, end) map to [n-end, n-start) in reversed sequence
    - Self-closing tags at position i map to position n-i
    
    Example:
        # complement_fn maps A<->T, C<->G
        reverse_complement_with_tags('ACG<region>TT</region>AA', complement_fn)
        -> 'TT<region>AA</region>CGT'
    """
    # Parse regions
    regions = find_all_regions(seq)
    
    # If no regions, just reverse complement the content
    if not regions:
        return ''.join(complement_fn(c) for c in reversed(seq))
    
    # Get content without tags
    content = strip_all_tags(seq)
    n = len(content)
    
    # Build mapping from literal positions to content positions
    nontag_positions = get_nontag_positions(seq)
    literal_to_content = {lit: i for i, lit in enumerate(nontag_positions)}
    
    # Calculate content ranges for each region and their new positions
    region_info = []
    for r in regions:
        if r.is_zero_length:
            # Self-closing tag: find its content position
            # It's positioned "before" the character at the next content position
            # Find the content position after this tag
            content_pos = None
            for lit_pos in range(r.end, len(seq) + 1):
                if lit_pos in literal_to_content:
                    content_pos = literal_to_content[lit_pos]
                    break
            if content_pos is None:
                content_pos = n  # At the end
            
            # New position: n - content_pos
            new_pos = n - content_pos
            
            # Determine seq_length for build_region_tags
            if r.declared_seq_length_str == 'None':
                seq_length_arg = -1  # Will produce seq_length='None'
            elif r.declared_seq_length_str is not None:
                seq_length_arg = int(r.declared_seq_length_str)
            else:
                seq_length_arg = None
            
            region_info.append({
                'name': r.name,
                'strand': r.strand,
                'seq_length_arg': seq_length_arg,
                'is_zero_length': True,
                'new_start': new_pos,
                'new_end': new_pos,
            })
        else:
            # Region tag: find content start and end
            content_start = literal_to_content[r.content_start]
            # content_end is one past the last character inside the region
            # r.content_end is the literal position of the closing tag '<'
            # We need to find the content index after the last content char
            if r.content_end == r.content_start:
                # Empty region content
                content_end = content_start
            else:
                # Find the last content character before r.content_end
                last_content_literal = r.content_end - 1
                while last_content_literal >= r.content_start and last_content_literal not in literal_to_content:
                    last_content_literal -= 1
                if last_content_literal >= r.content_start:
                    content_end = literal_to_content[last_content_literal] + 1
                else:
                    content_end = content_start
            
            # New positions: [n - end, n - start)
            new_start = n - content_end
            new_end = n - content_start
            
            # Determine seq_length for build_region_tags
            if r.declared_seq_length_str == 'None':
                seq_length_arg = -1
            elif r.declared_seq_length_str is not None:
                seq_length_arg = int(r.declared_seq_length_str)
            else:
                seq_length_arg = None
            
            region_info.append({
                'name': r.name,
                'strand': r.strand,
                'seq_length_arg': seq_length_arg,
                'is_zero_length': False,
                'new_start': new_start,
                'new_end': new_end,
            })
    
    # Reverse complement the content
    rc_content = ''.join(complement_fn(c) for c in reversed(content))
    
    # Build result by inserting tags at their new positions
    # We need to handle overlapping/nested regions properly
    # Strategy: collect all "events" (tag starts and ends) and process in order
    
    events = []  # (position, priority, event_type, region_idx)
    # Priority: lower = processed first at same position
    # For opening tags: use region length as priority (longer regions open first)
    # For closing tags: use negative region length (shorter regions close first)
    # For self-closing: use 0
    
    for idx, ri in enumerate(region_info):
        if ri['is_zero_length']:
            events.append((ri['new_start'], 0, 'self_close', idx))
        else:
            region_len = ri['new_end'] - ri['new_start']
            events.append((ri['new_start'], -region_len, 'open', idx))
            events.append((ri['new_end'], region_len, 'close', idx))
    
    # Sort events by position, then by priority
    events.sort(key=lambda e: (e[0], e[1]))
    
    # Build result
    result = []
    last_pos = 0
    
    for pos, priority, event_type, idx in events:
        # Add content up to this position
        if pos > last_pos:
            result.append(rc_content[last_pos:pos])
            last_pos = pos
        
        ri = region_info[idx]
        if event_type == 'self_close':
            result.append(build_region_tags(
                ri['name'],
                content='',
                strand=ri['strand'],
                seq_length=ri['seq_length_arg'],
            ))
        elif event_type == 'open':
            # Build opening tag
            attrs = []
            if ri['strand'] == '-':
                attrs.append("strand='-'")
            if ri['seq_length_arg'] is not None:
                if ri['seq_length_arg'] == -1:
                    attrs.append("seq_length='None'")
                else:
                    attrs.append(f"seq_length='{ri['seq_length_arg']}'")
            attrs_str = ' ' + ' '.join(attrs) if attrs else ''
            result.append(f"<{ri['name']}{attrs_str}>")
        elif event_type == 'close':
            result.append(f"</{ri['name']}>")
    
    # Add remaining content
    if last_pos < n:
        result.append(rc_content[last_pos:])
    
    return ''.join(result)


def get_length_without_tags(seq: str) -> int:
    """Get sequence length excluding region tags (but including region content)."""
    return len(strip_all_tags(seq))


def get_nontag_positions(seq: str) -> list[int]:
    """Get raw string positions of all characters excluding region tag interiors.
    
    Returns positions of characters that are NOT part of region tags.
    This includes region content but excludes the <...> tag syntax itself.
    """
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


def build_region_tags(
    name: str, 
    content: str = '', 
    strand: str = '+',
    seq_length: Optional[int] = None,
    explicit_strand: bool = False,
) -> str:
    """Build an XML region tag string.
    
    Args:
        name: Region name
        content: Content to wrap (empty string for zero-length region)
        strand: '+' or '-' ('+' is omitted from output by default)
        seq_length: Optional declared seq_length (None to omit, -1 for 'None')
        explicit_strand: If True, include strand='+' explicitly (useful for strand='both')
    
    Returns:
        XML region string like '<name>content</name>' or '<name/>'
    """
    attrs = []
    if strand == '-':
        attrs.append("strand='-'")
    elif explicit_strand:
        attrs.append("strand='+'")
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


def _validate_regions(seq: str) -> set:
    """
    Parse regions from a sequence string and register/validate with Party.
    
    For each region found:
    - If seq_length attribute is declared and not 'None': validate content matches
    - If seq_length='None': register as variable-length (seq_length=None)
    - If no seq_length attribute: infer seq_length from content length
    
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
        # Determine the seq_length to register
        if pr.declared_seq_length_str == 'None':
            # Explicitly declared as variable length
            seq_length = None
        elif pr.declared_seq_length_str is not None:
            # Explicitly declared with a specific length (already validated in parsing)
            seq_length = int(pr.declared_seq_length_str)
        else:
            # Not declared, infer from content (excluding nested region tags)
            seq_length = get_length_without_tags(pr.content)
        
        # Register with party (will raise if conflict)
        region = party.register_region(pr.name, seq_length)
        regions_found.add(region)
    
    return regions_found
