"""Sequence highlighting with regex-based ANSI styling."""
import re
from .types import Literal, Sequence, Optional, beartype
from . import dna

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

# CSS/HTML named colors (hex values)
CSS_COLORS = {
    'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7', 'aqua': '#00ffff',
    'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc',
    'bisque': '#ffe4c4', 'black': '#000000', 'blanchedalmond': '#ffebcd',
    'blueviolet': '#8a2be2', 'brown': '#a52a2a', 'burlywood': '#deb887',
    'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00', 'chocolate': '#d2691e',
    'coral': '#ff7f50', 'cornflowerblue': '#6495ed', 'cornsilk': '#fff8dc',
    'crimson': '#dc143c', 'darkblue': '#00008b', 'darkcyan': '#008b8b',
    'darkgoldenrod': '#b8860b', 'darkgray': '#a9a9a9', 'darkgreen': '#006400',
    'darkgrey': '#a9a9a9', 'darkkhaki': '#bdb76b', 'darkmagenta': '#8b008b',
    'darkolivegreen': '#556b2f', 'darkorange': '#ff8c00', 'darkorchid': '#9932cc',
    'darkred': '#8b0000', 'darksalmon': '#e9967a', 'darkseagreen': '#8fbc8f',
    'darkslateblue': '#483d8b', 'darkslategray': '#2f4f4f', 'darkslategrey': '#2f4f4f',
    'darkturquoise': '#00ced1', 'darkviolet': '#9400d3', 'deeppink': '#ff1493',
    'deepskyblue': '#00bfff', 'dimgray': '#696969', 'dimgrey': '#696969',
    'dodgerblue': '#1e90ff', 'firebrick': '#b22222', 'floralwhite': '#fffaf0',
    'forestgreen': '#228b22', 'fuchsia': '#ff00ff', 'gainsboro': '#dcdcdc',
    'ghostwhite': '#f8f8ff', 'gold': '#ffd700', 'goldenrod': '#daa520',
    'gray': '#808080', 'grey': '#808080', 'greenyellow': '#adff2f',
    'honeydew': '#f0fff0', 'hotpink': '#ff69b4', 'indianred': '#cd5c5c',
    'indigo': '#4b0082', 'ivory': '#fffff0', 'khaki': '#f0e68c',
    'lavender': '#e6e6fa', 'lavenderblush': '#fff0f5', 'lawngreen': '#7cfc00',
    'lemonchiffon': '#fffacd', 'lightblue': '#add8e6', 'lightcoral': '#f08080',
    'lightcyan': '#e0ffff', 'lightgoldenrodyellow': '#fafad2', 'lightgray': '#d3d3d3',
    'lightgreen': '#90ee90', 'lightgrey': '#d3d3d3', 'lightpink': '#ffb6c1',
    'lightsalmon': '#ffa07a', 'lightseagreen': '#20b2aa', 'lightskyblue': '#87cefa',
    'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#b0c4de',
    'lightyellow': '#ffffe0', 'lime': '#00ff00', 'limegreen': '#32cd32',
    'linen': '#faf0e6', 'maroon': '#800000', 'mediumaquamarine': '#66cdaa',
    'mediumblue': '#0000cd', 'mediumorchid': '#ba55d3', 'mediumpurple': '#9370db',
    'mediumseagreen': '#3cb371', 'mediumslateblue': '#7b68ee', 'mediumspringgreen': '#00fa9a',
    'mediumturquoise': '#48d1cc', 'mediumvioletred': '#c71585', 'midnightblue': '#191970',
    'mintcream': '#f5fffa', 'mistyrose': '#ffe4e1', 'moccasin': '#ffe4b5',
    'navajowhite': '#ffdead', 'navy': '#000080', 'oldlace': '#fdf5e6',
    'olive': '#808000', 'olivedrab': '#6b8e23', 'orange': '#ffa500',
    'orangered': '#ff4500', 'orchid': '#da70d6', 'palegoldenrod': '#eee8aa',
    'palegreen': '#98fb98', 'paleturquoise': '#afeeee', 'palevioletred': '#db7093',
    'papayawhip': '#ffefd5', 'peachpuff': '#ffdab9', 'peru': '#cd853f',
    'pink': '#ffc0cb', 'plum': '#dda0dd', 'powderblue': '#b0e0e6',
    'purple': '#800080', 'rebeccapurple': '#663399', 'rosybrown': '#bc8f8f',
    'royalblue': '#4169e1', 'saddlebrown': '#8b4513', 'salmon': '#fa8072',
    'sandybrown': '#f4a460', 'seagreen': '#2e8b57', 'seashell': '#fff5ee',
    'sienna': '#a0522d', 'silver': '#c0c0c0', 'skyblue': '#87ceeb',
    'slateblue': '#6a5acd', 'slategray': '#708090', 'slategrey': '#708090',
    'snow': '#fffafa', 'springgreen': '#00ff7f', 'steelblue': '#4682b4',
    'tan': '#d2b48c', 'teal': '#008080', 'thistle': '#d8bfd8',
    'tomato': '#ff6347', 'turquoise': '#40e0d0', 'violet': '#ee82ee',
    'wheat': '#f5deb3', 'white': '#ffffff', 'whitesmoke': '#f5f5f5',
    'yellowgreen': '#9acd32',
}

WhichType = Literal['all', 'upper', 'lower', 'gap', 'tags', 'contents']

# Regex to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\033\[[0-9;]*m')

# Default gap characters - subset of dna.IGNORE_CHARS commonly used as gaps
DEFAULT_GAP_CHARS = '-. '

# Basic ANSI foreground color codes (mutually exclusive)
_BASIC_FG_CODES = {'91', '92', '93', '94', '95', '96'}


def _hex_to_ansi(hex_color: str) -> str:
    """Convert hex color (#RRGGBB) to true-color ANSI code (38;2;R;G;B)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'38;2;{r};{g};{b}'


def _parse_style(style: str) -> str:
    """Convert style name to ANSI code. Supports STYLE_CODES, CSS colors, hex, or raw codes."""
    if style in STYLE_CODES:
        return STYLE_CODES[style]
    if style in CSS_COLORS:
        return _hex_to_ansi(CSS_COLORS[style])
    if style.startswith('#') and len(style) == 7:
        return _hex_to_ansi(style)
    # Assume raw ANSI code
    return style


def _is_foreground_code(code: str) -> bool:
    """Check if an ANSI code is a foreground color (basic or true-color)."""
    return code in _BASIC_FG_CODES or code.startswith('38;2;') or code.startswith('38;5;')


@beartype
class Highlighter:
    """Applies ANSI styling to sequences based on regex or predefined patterns."""
    
    def __init__(
        self,
        style: str,
        region: Optional[str] = None,
        which: WhichType = 'contents',
        regex: Optional[str] = None,
    ) -> None:
        """Create a highlighter.
        
        Args:
            style: Color/style name. Supports basic ANSI ('red', 'bold', etc.),
                   CSS named colors ('coral', 'tomato', etc.), hex ('#ff7f50'),
                   or raw ANSI codes.
            region: Optional marker name to restrict highlighting to that region.
            which: Predefined pattern (default 'contents'). Options:
                   'all', 'upper', 'lower', 'gap', 'tags', 'contents'.
            regex: Custom regex pattern. If specified, overrides `which`.
        """
        
        self.style = style
        self.region = region
        self.which = which if regex is None else None  # regex overrides which
        self.regex = regex
        self._code = _parse_style(style)
        
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
        
        # If region specified but not found, return unchanged
        if bounds is None and self.region is not None:
            return clean_text
        
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


def _resolve_styles(styles_with_priority: dict[str, int]) -> list[str]:
    """Resolve conflicting styles using priority (higher wins for foreground colors)."""
    # Separate foreground colors from other styles
    fg_codes = {c: p for c, p in styles_with_priority.items() if _is_foreground_code(c)}
    other_codes = sorted(c for c in styles_with_priority if not _is_foreground_code(c))
    
    # Build result with COLOR first, then modifiers (bold/underline)
    # Some terminals (VS Code) need this order to render correctly
    result = []
    if fg_codes:
        # Pick the foreground color with highest priority (later highlighter wins)
        winner = max(fg_codes.items(), key=lambda x: x[1])[0]
        result.append(winner)
    result.extend(other_codes)  # Add modifiers after color
    
    return result


@beartype
def apply_highlights(text: str, highlighters: Sequence[Highlighter]) -> str:
    """Apply multiple highlighters with proper overlap handling.
    
    Uses character-by-character style tracking. For overlapping regions:
    - Foreground colors: later-added highlight takes precedence
    - Other styles (bold, underline): all are combined
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
    
    # Track styles for each character position: code -> priority (highlighter index)
    char_styles: list[dict[str, int]] = [{} for _ in range(n)]
    
    # Apply each highlighter's matches to the style tracking
    for priority, hl in enumerate(highlighters):
        # Determine bounds for this highlighter
        bounds = hl._get_region_bounds(clean_text)
        
        if bounds is None:
            if hl.region is not None:
                # Region specified but not found in text - skip this highlighter
                continue
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
                char_styles[pos][hl._code] = priority
    
    # Build output with combined ANSI codes
    result = []
    current_codes: list[str] = []
    
    for i, char in enumerate(clean_text):
        new_codes = _resolve_styles(char_styles[i]) if char_styles[i] else []
        if new_codes != current_codes:
            if current_codes:
                result.append('\033[0m')  # Reset previous styles
            if new_codes:
                codes = ';'.join(new_codes)
                result.append(f'\033[{codes}m')
            current_codes = new_codes
        result.append(char)
    
    # Reset at end if we have active styles
    if current_codes:
        result.append('\033[0m')
    
    return ''.join(result)


@beartype
def add_highlight(
    style: str,
    region: Optional[str] = None,
    which: str = 'contents',
    regex: Optional[str] = None,
) -> None:
    """Create Highlighter(s) and add to the active party's highlight list.
    
    Each of style, region, and which can contain whitespace-separated values.
    Creates one Highlighter for every combination in the Cartesian product.
    
    Examples:
        add_highlight(style='bold cyan')  # 2 Highlighters
        add_highlight(style='bold', region='cre bc')  # 2 Highlighters
        add_highlight(style='bold cyan', region='cre bc')  # 4 Highlighters
    
    Args same as Highlighter.__init__(), except style/region/which can be space-separated.
    Returns the created Highlighter, or list of Highlighters if multiple combinations.
    """
    from itertools import product
    from .party import get_active_party
    
    styles = style.split()
    regions = region.split() if region is not None else [None]
    whiches = which.split()
    
    for s, r, w in product(styles, regions, whiches):
        hl = Highlighter(s, r, w, regex)  # type: ignore
        get_active_party()._highlights.append(hl)


@beartype
def clear_highlights() -> None:
    """Clear all highlights from the active party."""
    from .party import get_active_party
    get_active_party()._highlights.clear()


@beartype
def set_highlights(highlighters: Sequence[Highlighter]) -> None:
    """Replace the active party's highlights with the given list."""
    from .party import get_active_party
    party = get_active_party()
    party._highlights.clear()
    party._highlights.extend(highlighters)


def print_named_colors() -> None:
    """Print all named colors (CSS + basic ANSI) each styled in that color."""
    # Basic ANSI colors
    print("Basic ANSI colors:")
    for name in ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']:
        code = STYLE_CODES[name]
        print(f"  \033[{code}m{name}\033[0m")
    print()
    
    # CSS colors (sorted alphabetically)
    print("CSS named colors:")
    names = sorted(CSS_COLORS.keys())
    # Print in columns
    cols = 4
    col_width = max(len(n) for n in names) + 2
    for i, name in enumerate(names):
        code = _hex_to_ansi(CSS_COLORS[name])
        styled = f"\033[{code}m{name}\033[0m"
        # Pad to column width (accounting for ANSI codes)
        padding = col_width - len(name)
        print(styled + ' ' * padding, end='')
        if (i + 1) % cols == 0:
            print()
    if len(names) % cols != 0:
        print()
