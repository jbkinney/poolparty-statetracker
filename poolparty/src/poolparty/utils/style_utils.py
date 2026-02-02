"""Inline sequence styling with ANSI colors."""

import re
from dataclasses import dataclass

import numpy as np

from ..types import Sequence, StyleList, beartype


def styles_suppressed() -> bool:
    """Return True if inline styles are suppressed in the active party."""
    # Import locally to avoid circular import (party.py imports types.py which imports style_utils.py)
    # Access _active_party directly to bypass beartype overhead in get_active_party()
    from ..party import _active_party

    return _active_party.suppress_styles if _active_party else False


# ANSI escape codes for styling
STYLE_CODES = {
    # Foreground colors
    "red": "91",
    "green": "92",
    "yellow": "93",
    "blue": "94",
    "magenta": "95",
    "cyan": "96",
    "white": "97",
    "black": "30",
    # Modifiers
    "bold": "1",
    "underline": "4",
    "blink": "5",
    "blinking": "5",
    "invert": "7",
    "reverse": "7",
    # Background colors
    "on_red": "101",
    "on_green": "102",
    "on_yellow": "103",
    "on_blue": "104",
    "on_magenta": "105",
    "on_cyan": "106",
    "on_white": "107",
    "on_black": "40",
}

# CSS/HTML named colors (hex values)
CSS_COLORS = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blanchedalmond": "#ffebcd",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgreen": "#006400",
    "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "floralwhite": "#fffaf0",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "grey": "#808080",
    "greenyellow": "#adff2f",
    "honeydew": "#f0fff0",
    "hotpink": "#ff69b4",
    "indianred": "#cd5c5c",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead",
    "navy": "#000080",
    "oldlace": "#fdf5e6",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "seashell": "#fff5ee",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "whitesmoke": "#f5f5f5",
    "yellowgreen": "#9acd32",
}

# Regex to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r"\033\[[0-9;]*m")

# Default gap characters - subset of dna_utils.IGNORE_CHARS commonly used as gaps
DEFAULT_GAP_CHARS = "-. "

# Basic ANSI foreground color codes (mutually exclusive)
_BASIC_FG_CODES = {"91", "92", "93", "94", "95", "96", "97", "30"}

# Basic ANSI background color codes (mutually exclusive)
_BASIC_BG_CODES = {"101", "102", "103", "104", "105", "106", "107", "40"}

# Case transformation tokens for inline styles (not ANSI codes)
CASE_TRANSFORMS = {"upper", "lower", "uppercase", "lowercase", "swapcase"}


def _hex_to_ansi(hex_color: str) -> str:
    """Convert hex color (#RRGGBB) to true-color ANSI foreground code (38;2;R;G;B)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"38;2;{r};{g};{b}"


def _hex_to_ansi_bg(hex_color: str) -> str:
    """Convert hex color (#RRGGBB) to true-color ANSI background code (48;2;R;G;B)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"48;2;{r};{g};{b}"


def _parse_style(style: str) -> str:
    """Convert style name to ANSI code. Supports STYLE_CODES, CSS colors, hex, or raw codes."""
    # Check for on_ prefix (background colors)
    if style.startswith("on_"):
        color_part = style[3:]  # Strip "on_"
        # Check if it's a basic background color in STYLE_CODES
        if style in STYLE_CODES:
            return STYLE_CODES[style]
        # Check if it's a CSS color name
        if color_part in CSS_COLORS:
            return _hex_to_ansi_bg(CSS_COLORS[color_part])
        # Check if it's a hex code
        if color_part.startswith("#") and len(color_part) == 7:
            return _hex_to_ansi_bg(color_part)
        # Unknown background color
        raise ValueError(
            f"Unknown background color: {style!r}. Valid options: "
            f"on_<color> where color is a CSS color name or hex code (#RRGGBB)."
        )
    # Foreground colors and modifiers
    if style in STYLE_CODES:
        return STYLE_CODES[style]
    if style in CSS_COLORS:
        return _hex_to_ansi(CSS_COLORS[style])
    if style.startswith("#") and len(style) == 7:
        return _hex_to_ansi(style)
    # Allow raw numeric ANSI codes (e.g., '91', '38;2;R;G;B')
    if re.match(r"^[\d;]+$", style):
        return style
    # Unknown style - raise helpful error
    raise ValueError(
        f"Unknown style: {style!r}. Valid options: "
        f"ANSI names ({', '.join(sorted(STYLE_CODES.keys()))}), "
        f"CSS color names, hex codes (#RRGGBB), or raw ANSI codes (numeric)."
    )


def _is_foreground_code(code: str) -> bool:
    """Check if an ANSI code is a foreground color (basic or true-color)."""
    return code in _BASIC_FG_CODES or code.startswith("38;2;") or code.startswith("38;5;")


def _is_background_code(code: str) -> bool:
    """Check if an ANSI code is a background color (basic or true-color)."""
    return code in _BASIC_BG_CODES or code.startswith("48;2;") or code.startswith("48;5;")


def _resolve_styles(styles_with_priority: dict[str, int]) -> list[str]:
    """Resolve conflicting styles using priority (higher wins for fg/bg colors)."""
    # Separate foreground colors, background colors, and other styles
    fg_codes = {c: p for c, p in styles_with_priority.items() if _is_foreground_code(c)}
    bg_codes = {c: p for c, p in styles_with_priority.items() if _is_background_code(c)}
    other_codes = sorted(
        c for c in styles_with_priority if not _is_foreground_code(c) and not _is_background_code(c)
    )

    # Build result with FG COLOR first, then BG COLOR, then modifiers
    # Some terminals (VS Code) need this order to render correctly
    result = []
    if fg_codes:
        # Pick the foreground color with highest priority (later highlighter wins)
        winner = max(fg_codes.items(), key=lambda x: x[1])[0]
        result.append(winner)
    if bg_codes:
        # Pick the background color with highest priority (later highlighter wins)
        winner = max(bg_codes.items(), key=lambda x: x[1])[0]
        result.append(winner)
    result.extend(other_codes)  # Add modifiers after colors

    return result


@dataclass(frozen=True)
class SeqStyle:
    """Immutable container for inline styles with associated sequence length.

    Provides Pythonic operations for extracting, reversing, and concatenating
    styles while maintaining position correctness.
    """

    _style_list: StyleList
    length: int

    # --- Access ---
    @property
    def style_list(self) -> StyleList:
        """Get underlying StyleList."""
        return self._style_list

    def __len__(self) -> int:
        return self.length

    def __bool__(self) -> bool:
        """Return True if has any styles."""
        return len(self._style_list) > 0

    def __repr__(self) -> str:
        n = len(self._style_list)
        return f"SeqStyle({n} style{'s' if n != 1 else ''}, length={self.length})"

    # --- Construction ---
    @classmethod
    def empty(cls, length: int) -> "SeqStyle":
        """Create a SeqStyle with no styles (spacer)."""
        return cls([], length)

    @classmethod
    def from_style_list(cls, style_list: StyleList, length: int) -> "SeqStyle":
        """Create from existing StyleList."""
        return cls(style_list, length)

    @classmethod
    def from_parent(
        cls, parent_styles: list["SeqStyle"] | None, index: int, length: int
    ) -> "SeqStyle":
        """Get style from parent_styles or create empty if not available.

        Parameters
        ----------
        parent_styles : list[SeqStyle] | None
            List of parent styles, or None.
        index : int
            Index into parent_styles to retrieve.
        length : int
            Length for empty SeqStyle if parent not available.

        Returns
        -------
        SeqStyle
            Parent style at index, or empty SeqStyle of given length.
        """
        if parent_styles and len(parent_styles) > index:
            return parent_styles[index]
        return cls.empty(length)

    @classmethod
    def full(cls, length: int, style: str | None = None) -> "SeqStyle":
        """Create SeqStyle covering all positions with optional style.

        Parameters
        ----------
        length : int
            Length of the sequence.
        style : str | None, default=None
            Style spec to apply to all positions. If None, returns empty SeqStyle.

        Returns
        -------
        SeqStyle
            SeqStyle with style applied to all positions, or empty if style is None.
        """
        if style is None or length == 0:
            return cls.empty(length)
        positions = np.arange(length, dtype=np.int64)
        return cls([(style, positions)], length)

    def copy(self) -> "SeqStyle":
        """Return self (SeqStyle is immutable, so no actual copy needed)."""
        return self

    # --- Adding styles (returns new SeqStyle) ---
    def add_style(self, spec: str, positions: np.ndarray) -> "SeqStyle":
        """Return new SeqStyle with additional style appended."""
        new_style_list = self._style_list + [(spec, positions)]
        return SeqStyle(new_style_list, self.length)

    # --- Slicing ---
    def __getitem__(self, key: slice) -> "SeqStyle":
        """Extract region: seq_style[start:end] returns 0-indexed SeqStyle.

        Examples:
            seq_style[10:50]  # positions 10-49 -> 0-indexed, length=40
            seq_style[:50]    # first 50 positions
            seq_style[50:]    # from position 50 to end
        """
        if not isinstance(key, slice):
            raise TypeError("SeqStyle only supports slice indexing")
        if key.step is not None:
            raise ValueError("SeqStyle slicing does not support step")

        start = key.start if key.start is not None else 0
        end = key.stop if key.stop is not None else self.length
        new_length = end - start

        if new_length <= 0:
            return SeqStyle.empty(0)

        result: StyleList = []
        for spec, positions in self._style_list:
            mask = (positions >= start) & (positions < end)
            filtered = positions[mask]
            if len(filtered) > 0:
                # Shift to 0-indexed
                result.append((spec, filtered - start))

        return SeqStyle(result, new_length)

    # --- Reversal ---
    def reversed(self, do_reverse: bool = True) -> "SeqStyle":
        """Return SeqStyle with positions mirrored within length.

        Useful for reverse complement operations. If do_reverse=False,
        returns self unchanged (convenient for conditional reversal).
        """
        if not do_reverse:
            return self

        result: StyleList = []
        for spec, positions in self._style_list:
            # Mirror: new_pos = length - 1 - old_pos
            mirrored = self.length - 1 - positions
            result.append((spec, mirrored))

        return SeqStyle(result, self.length)

    # --- Concatenation ---
    @classmethod
    def join(cls, seq_styles: Sequence["SeqStyle"]) -> "SeqStyle":
        """Concatenate multiple SeqStyles with automatic position offsets."""
        if not seq_styles:
            return cls.empty(0)

        result: StyleList = []
        offset = 0

        for seq_style in seq_styles:
            for spec, positions in seq_style._style_list:
                if len(positions) > 0:
                    result.append((spec, positions + offset))
            offset += seq_style.length

        return cls(result, offset)

    def __add__(self, other: "SeqStyle") -> "SeqStyle":
        """Enable: seq_style_a + seq_style_b"""
        return SeqStyle.join([self, other])

    # --- Splitting ---
    def split(self, breakpoints: list[int]) -> tuple["SeqStyle", ...]:
        """Split at breakpoints, returning N+1 SeqStyle objects.

        Example: seq_style.split([10, 30]) on length=50 returns:
            (seq_style[0:10], seq_style[10:30], seq_style[30:50])
        """
        points = [0] + sorted(breakpoints) + [self.length]
        return tuple(self[points[i] : points[i + 1]] for i in range(len(points) - 1))

    # --- Validation and application ---
    def validate(self) -> None:
        """Validate all positions are within [0, length)."""
        validate_style_positions(self.length, self._style_list)

    def apply(self, seq: str) -> str:
        """Apply styles to sequence, returning ANSI-styled string."""
        return apply_inline_styles(seq, self._style_list)


def reset(text: str) -> str:
    """Strip all ANSI escape codes from text."""
    return ANSI_ESCAPE_PATTERN.sub("", text)


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
            raise ValueError(f"Style '{spec}' has negative position(s): min={min_pos}")
        if max_pos >= seq_len:
            raise ValueError(f"Style '{spec}' has position(s) >= seq_len={seq_len}: max={max_pos}")


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
            if transform in ("upper", "uppercase"):
                char = char.upper()
            elif transform in ("lower", "lowercase"):
                char = char.lower()
            elif transform == "swapcase":
                char = char.swapcase()

        new_codes = _resolve_styles(char_styles[i]) if char_styles[i] else []
        if new_codes != current_codes:
            if current_codes:
                result.append("\033[0m")  # Reset previous styles
            if new_codes:
                codes_str = ";".join(new_codes)
                result.append(f"\033[{codes_str}m")
            current_codes = new_codes
        result.append(char)

    # Reset at end if we have active styles
    if current_codes:
        result.append("\033[0m")

    return "".join(result)


def print_named_colors() -> None:
    """Print all named colors (CSS + basic ANSI) each styled in that color."""
    # Basic ANSI colors
    print("Basic ANSI colors:")
    for name in ["red", "green", "yellow", "blue", "magenta", "cyan"]:
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
        print(styled + " " * padding, end="")
        if (i + 1) % cols == 0:
            print()
    if len(names) % cols != 0:
        print()
