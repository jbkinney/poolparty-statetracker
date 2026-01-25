"""Inline sequence styling with ANSI colors."""
import re
from ..types import Literal, Optional, beartype, StyleList
import numpy as np

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

# Regex to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\033\[[0-9;]*m')

# Default gap characters - subset of dna_utils.IGNORE_CHARS commonly used as gaps
DEFAULT_GAP_CHARS = '-. '

# Basic ANSI foreground color codes (mutually exclusive)
_BASIC_FG_CODES = {'91', '92', '93', '94', '95', '96'}

# Case transformation tokens for inline styles (not ANSI codes)
CASE_TRANSFORMS = {'upper', 'lower', 'uppercase', 'lowercase', 'swapcase'}


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
    # Allow raw numeric ANSI codes (e.g., '91', '38;2;R;G;B')
    if re.match(r'^[\d;]+$', style):
        return style
    # Unknown style - raise helpful error
    raise ValueError(
        f"Unknown style: {style!r}. Valid options: "
        f"ANSI names ({', '.join(sorted(STYLE_CODES.keys()))}), "
        f"CSS color names, hex codes (#RRGGBB), or raw ANSI codes (numeric)."
    )


def _is_foreground_code(code: str) -> bool:
    """Check if an ANSI code is a foreground color (basic or true-color)."""
    return code in _BASIC_FG_CODES or code.startswith('38;2;') or code.startswith('38;5;')


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


def reset(text: str) -> str:
    """Strip all ANSI escape codes from text."""
    return ANSI_ESCAPE_PATTERN.sub('', text)


@beartype
def validate_style_positions(seq_len: int, styles: StyleList) -> None:
    """Validate that all style positions are within bounds.
    
    Parameters
    ----------
    seq_len : int
        Length of the sequence being styled.
    styles : StyleList
        List of (spec, positions) tuples to validate.
    
    Raises
    ------
    ValueError
        If any position is negative or >= seq_len, with detailed context.
    """
    for spec, positions in styles:
        if len(positions) == 0:
            continue
        min_pos, max_pos = int(positions.min()), int(positions.max())
        if min_pos < 0:
            raise ValueError(
                f"Style '{spec}' has negative position(s): min={min_pos}"
            )
        if max_pos >= seq_len:
            raise ValueError(
                f"Style '{spec}' has position(s) >= seq_len={seq_len}: max={max_pos}"
            )


@beartype
def split_styles_by_region(
    styles: StyleList,
    region_start: int,
    region_end: int,
) -> tuple[StyleList, StyleList, StyleList]:
    """Split styles into (prefix, region, suffix) based on position bounds.
    
    Positions in [0, region_start) go to prefix.
    Positions in [region_start, region_end) go to region (shifted to 0-indexed).
    Positions in [region_end, ...) go to suffix.
    
    Parameters
    ----------
    styles : StyleList
        List of (spec, positions) tuples to split.
    region_start : int
        Start position of region (inclusive).
    region_end : int
        End position of region (exclusive).
    
    Returns
    -------
    tuple[StyleList, StyleList, StyleList]
        (prefix_styles, region_styles, suffix_styles) where:
        - prefix_styles: positions < region_start (unchanged)
        - region_styles: positions in [region_start, region_end) shifted to 0-indexed
        - suffix_styles: positions >= region_end (unchanged)
    """
    prefix_styles: StyleList = []
    region_styles: StyleList = []
    suffix_styles: StyleList = []
    
    for spec, positions in styles:
        if len(positions) == 0:
            continue
        
        # Split positions into three groups
        prefix_mask = positions < region_start
        region_mask = (positions >= region_start) & (positions < region_end)
        suffix_mask = positions >= region_end
        
        if np.any(prefix_mask):
            prefix_styles.append((spec, positions[prefix_mask]))
        
        if np.any(region_mask):
            # Shift to region-relative coordinates (0-indexed)
            adjusted_positions = positions[region_mask] - region_start
            region_styles.append((spec, adjusted_positions))
        
        if np.any(suffix_mask):
            suffix_styles.append((spec, positions[suffix_mask]))
    
    return prefix_styles, region_styles, suffix_styles


@beartype
def shift_style_positions(styles: StyleList, offset: int) -> StyleList:
    """Shift all positions in styles by offset.
    
    Parameters
    ----------
    styles : StyleList
        List of (spec, positions) tuples.
    offset : int
        Amount to shift positions (can be negative).
    
    Returns
    -------
    StyleList
        New StyleList with all positions shifted by offset.
    """
    if not styles:
        return []
    
    shifted: StyleList = []
    for spec, positions in styles:
        if len(positions) > 0:
            shifted.append((spec, positions + offset))
        else:
            shifted.append((spec, positions))
    
    return shifted


@beartype
def reassemble_styles(
    prefix_styles: StyleList,
    region_styles: StyleList,
    suffix_styles: StyleList,
    prefix_len: int,
    old_region_len: int,
    new_region_len: int,
) -> StyleList:
    """Reassemble styles after region operation.
    
    - prefix_styles: unchanged
    - region_styles: shifted by prefix_len
    - suffix_styles: shifted by (prefix_len + new_region_len - old_region_len)
    
    Parameters
    ----------
    prefix_styles : StyleList
        Styles for positions before the region (unchanged).
    region_styles : StyleList
        Styles from region operation (will be shifted by prefix_len).
    suffix_styles : StyleList
        Styles for positions after the region (will be shifted by length delta).
    prefix_len : int
        Length of prefix before region.
    old_region_len : int
        Original length of region content.
    new_region_len : int
        New length of region content after operation.
    
    Returns
    -------
    StyleList
        Reassembled styles with correct positions.
    """
    result: StyleList = []
    
    # Add prefix styles unchanged
    result.extend(prefix_styles)
    
    # Shift region styles by prefix length
    shifted_region = shift_style_positions(region_styles, prefix_len)
    result.extend(shifted_region)
    
    # Shift suffix styles by prefix_len + length_delta
    length_delta = new_region_len - old_region_len
    suffix_offset = prefix_len + length_delta
    shifted_suffix = shift_style_positions(suffix_styles, suffix_offset)
    result.extend(shifted_suffix)
    
    return result


@beartype
def apply_inline_styles(seq: str, styles: StyleList, validate: bool = True) -> str:
    """Apply per-sequence inline styles to a sequence.
    
    Each style tuple in `styles` is (spec, positions) where:
    - spec: Style specification string (e.g., 'bold blue', '#ff7f50', 'coral')
      Can include 'upper' or 'lower' to transform character case at positions.
    - positions: np.ndarray of character positions to style
    
    Styles are applied in order. Later styles override foreground colors
    but combine with modifiers (bold, underline). Later case transforms
    override earlier ones.
    
    Parameters
    ----------
    seq : str
        The sequence to style.
    styles : StyleList
        List of (spec, positions) tuples.
    validate : bool, default=True
        If True, validate positions are within bounds before applying.
    """
    if not styles:
        return seq
    
    # Strip any existing ANSI codes
    clean_seq = reset(seq)
    n = len(clean_seq)
    if n == 0:
        return clean_seq
    
    # Validate positions if requested
    if validate:
        validate_style_positions(n, styles)
    
    # Track styles for each character: code -> priority (style index)
    char_styles: list[dict[str, int]] = [{} for _ in range(n)]
    # Track case transforms for each character: transform -> priority
    char_transforms: list[dict[str, int]] = [{} for _ in range(n)]
    
    # Apply each style tuple
    for priority, (spec, positions) in enumerate(styles):
        # Parse the style spec, separating case transforms from ANSI codes
        tokens = spec.split()
        case_transform = None
        style_tokens = []
        for t in tokens:
            if t in CASE_TRANSFORMS:
                case_transform = t
            else:
                style_tokens.append(t)
        
        codes = [_parse_style(s) for s in style_tokens]
        
        # Apply codes and transforms to specified positions
        for pos in positions:
            if 0 <= pos < n:
                for code in codes:
                    char_styles[pos][code] = priority
                if case_transform is not None:
                    char_transforms[pos][case_transform] = priority
    
    # Build output with combined ANSI codes and case transforms
    result = []
    current_codes: list[str] = []
    
    for i, char in enumerate(clean_seq):
        # Apply case transform if present (highest priority wins)
        if char_transforms[i]:
            transform = max(char_transforms[i].items(), key=lambda x: x[1])[0]
            if transform in ('upper', 'uppercase'):
                char = char.upper()
            elif transform in ('lower', 'lowercase'):
                char = char.lower()
            elif transform == 'swapcase':
                char = char.swapcase()
        
        new_codes = _resolve_styles(char_styles[i]) if char_styles[i] else []
        if new_codes != current_codes:
            if current_codes:
                result.append('\033[0m')  # Reset previous styles
            if new_codes:
                codes_str = ';'.join(new_codes)
                result.append(f'\033[{codes_str}m')
            current_codes = new_codes
        result.append(char)
    
    # Reset at end if we have active styles
    if current_codes:
        result.append('\033[0m')
    
    return ''.join(result)


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
