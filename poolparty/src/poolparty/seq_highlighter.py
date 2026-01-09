"""Sequence highlighting with regex-based ANSI styling."""
import re
from .types import Literal, Sequence, Optional, beartype

# ANSI escape codes for styling
STYLE_CODES = {
    'red':      '91',
    'green':    '92',
    'yellow':   '93',
    'blue':     '94',
    'magenta':  '95',
    'cyan':     '96',
    'bold':     '1',
    'underline':'4',
}

StyleType = Literal['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'bold', 'underline']
WhichType = Literal['all', 'upper', 'lower', 'gap', 'tags', 'contents']

# Regex to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\033\[[0-9;]*m')

# Default gap characters (from alphabet.py DEFAULT_IGNORE_CHARS)
DEFAULT_GAP_CHARS = '-. '


@beartype
class Highlighter:
    """Applies ANSI styling to sequences based on regex or predefined patterns."""
    
    def __init__(
        self,
        style: StyleType,
        region: Optional[str] = None,
        which: WhichType = 'contents',
        regex: Optional[str] = None,
    ) -> None:
        """Create a highlighter.
        
        Args:
            style: ANSI style ('red', 'green', 'yellow', 'blue', 'magenta', 
                   'cyan', 'bold', 'underline').
            region: Optional marker name to restrict highlighting to that region.
            which: Predefined pattern (default 'contents'). Options:
                   'all', 'upper', 'lower', 'gap', 'tags', 'contents'.
                   'upper'/'lower'/'gap'/'contents' only match molecular chars (outside XML tags).
                   'tags' highlights XML tags (all tags if no region, or specific region's tags).
                   'contents' highlights all non-tag chars (in region if specified, else whole seq).
            regex: Custom regex pattern. If specified, overrides `which`.
        """
        
        self.style = style
        self.region = region
        self.which = which if regex is None else None  # regex overrides which
        self.regex = regex
        self._code = STYLE_CODES[style]
        
        # These patterns only apply to molecular characters (outside tags)
        # 'contents' also excludes tags when no region is specified
        self._excludes_tags = self.which in ('upper', 'lower', 'gap', 'contents')
        
        # Build the internal regex pattern
        self._pattern = self._build_pattern()
    
    def _build_pattern(self) -> re.Pattern:
        """Build the regex pattern based on which/regex."""
        if self.regex is not None:
            return re.compile(self.regex)
        
        # Build pattern based on 'which'
        match self.which:
            case 'all' | 'contents':
                return re.compile(r'.')
            case 'upper':
                return re.compile(r'[A-Z]')
            case 'lower':
                return re.compile(r'[a-z]')
            case 'gap':
                # Escape special chars for character class
                escaped = re.escape(DEFAULT_GAP_CHARS)
                return re.compile(f'[{escaped}]')
            case 'tags':
                if self.region is None:
                    # Match all XML tags
                    from .marker_ops.parsing import TAG_PATTERN
                    return TAG_PATTERN
                else:
                    # Match the opening and closing tags for the specific region
                    name = re.escape(self.region)
                    return re.compile(rf'</?{name}(?:\s[^>]*)?>|<{name}(?:\s[^>]*)?/>')
            case _:
                raise ValueError(f"Unknown 'which' value: {self.which}")
    
    @staticmethod
    def _get_tag_positions(text: str) -> set[int]:
        """Get positions of all characters inside XML tags (the <...> syntax)."""
        from .marker_ops.parsing import TAG_PATTERN
        tag_positions: set[int] = set()
        for match in TAG_PATTERN.finditer(text):
            for i in range(match.start(), match.end()):
                tag_positions.add(i)
        return tag_positions
    
    def _get_region_bounds(self, text: str) -> Optional[tuple[int, int]]:
        """Get the start/end positions of the region in text.
        
        For which='contents', returns content bounds (excluding tags).
        Otherwise, returns full region bounds (including tags).
        
        Returns (start, end) or None if region not found.
        """
        if self.region is None:
            return None
        
        from .marker_ops.parsing import find_all_markers
        
        try:
            markers = find_all_markers(text)
        except ValueError:
            return None
        
        for m in markers:
            if m.name == self.region:
                if self.which == 'contents':
                    # Return content bounds (excluding tags)
                    return (m.content_start, m.content_end)
                else:
                    # Return full region bounds (including tags)
                    return (m.start, m.end)
        
        return None
    
    def apply(self, text: str) -> str:
        """Apply this highlighter to text, returning styled string.
        
        If region is specified, only highlights within that region.
        For 'upper', 'lower', 'gap': only highlights molecular chars (outside tags).
        Returns the full sequence with highlighting applied.
        
        Note: For multiple overlapping highlights, use apply_highlights() instead.
        """
        clean_text = self.reset(text)
        n = len(clean_text)
        if n == 0:
            return clean_text
        
        # Determine the bounds to highlight within
        bounds = self._get_region_bounds(clean_text)
        
        # Get tag positions if we need to exclude them
        tag_positions = self._get_tag_positions(clean_text) if self._excludes_tags else set()
        
        # Determine which positions are eligible for highlighting
        if bounds is None:
            eligible_start, eligible_end = 0, n
        else:
            eligible_start, eligible_end = bounds
        
        # Build character-by-character with highlighting
        start_code = f'\033[{self._code}m'
        reset_code = '\033[0m'
        
        # Track which positions should be highlighted
        highlight_positions: set[int] = set()
        search_text = clean_text[eligible_start:eligible_end]
        for match in self._pattern.finditer(search_text):
            for i in range(match.start(), match.end()):
                pos = eligible_start + i
                if pos not in tag_positions:
                    highlight_positions.add(pos)
        
        # Build output
        result = []
        in_highlight = False
        for i, char in enumerate(clean_text):
            should_highlight = i in highlight_positions
            if should_highlight and not in_highlight:
                result.append(start_code)
                in_highlight = True
            elif not should_highlight and in_highlight:
                result.append(reset_code)
                in_highlight = False
            result.append(char)
        
        if in_highlight:
            result.append(reset_code)
        
        return ''.join(result)
    
    @staticmethod
    def reset(text: str) -> str:
        """Strip all ANSI escape codes from text."""
        return ANSI_ESCAPE_PATTERN.sub('', text)
    
    def __repr__(self) -> str:
        parts = [f"style={self.style!r}"]
        if self.region:
            parts.append(f"region={self.region!r}")
        if self.which:
            parts.append(f"which={self.which!r}")
        if self.regex:
            parts.append(f"regex={self.regex!r}")
        return f"Highlighter({', '.join(parts)})"


@beartype
def apply_highlights(text: str, highlighters: Sequence[Highlighter]) -> str:
    """Apply multiple highlighters with proper overlap handling.
    
    Uses character-by-character style tracking so overlapping regions
    get all applicable styles combined (e.g., red + underline).
    """
    if not highlighters:
        return text
    
    # Strip any existing ANSI codes
    clean_text = Highlighter.reset(text)
    n = len(clean_text)
    if n == 0:
        return clean_text
    
    # Get tag positions once (shared across highlighters that need it)
    tag_positions = Highlighter._get_tag_positions(clean_text)
    
    # Track styles for each character position
    char_styles: list[set[str]] = [set() for _ in range(n)]
    
    # Apply each highlighter's matches to the style tracking
    for hl in highlighters:
        # Determine bounds for this highlighter
        bounds = hl._get_region_bounds(clean_text)
        
        if bounds is None:
            eligible_start, eligible_end = 0, n
        else:
            eligible_start, eligible_end = bounds
        
        # Match within eligible region
        search_text = clean_text[eligible_start:eligible_end]
        for match in hl._pattern.finditer(search_text):
            for i in range(match.start(), match.end()):
                pos = eligible_start + i
                # Skip tag positions if this highlighter excludes them
                if hl._excludes_tags and pos in tag_positions:
                    continue
                char_styles[pos].add(hl._code)
    
    # Build output with combined ANSI codes
    result = []
    current_styles: set[str] = set()
    
    for i, char in enumerate(clean_text):
        new_styles = char_styles[i]
        if new_styles != current_styles:
            if current_styles:
                result.append('\033[0m')  # Reset previous styles
            if new_styles:
                codes = ';'.join(sorted(new_styles))
                result.append(f'\033[{codes}m')
            current_styles = new_styles
        result.append(char)
    
    # Reset at end if we have active styles
    if current_styles:
        result.append('\033[0m')
    
    return ''.join(result)


@beartype
def add_highlight(
    style: StyleType,
    region: Optional[str] = None,
    which: WhichType = 'contents',
    regex: Optional[str] = None,
) -> Highlighter:
    """Create a Highlighter and add it to the active party's highlight list.
    
    Args same as Highlighter.__init__().
    Returns the created Highlighter.
    """
    from .party import get_active_party
    hl = Highlighter(style, region, which, regex)
    get_active_party()._highlights.append(hl)
    return hl


@beartype
def clear_highlights() -> None:
    """Clear all highlights from the active party."""
    from .party import get_active_party
    get_active_party()._highlights.clear()
