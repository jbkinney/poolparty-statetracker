"""Seq class for bundling sequence string and style."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from .style_utils import SeqStyle

CoordSystem = Literal["literal", "nontag", "molecular"]

# Sentinel for lazy coordinate map computation
_NOT_COMPUTED = object()


@dataclass(frozen=True)
class Seq:
    """Immutable container bundling sequence string, style, and region metadata.

    This is a generic base class for sequences. Subclasses (DnaSeq, ProteinSeq)
    define their own VALID_CHARS for molecular coordinate computation.

    Coordinate Systems
    ------------------
    Seq provides three coordinate systems for indexing positions:

    **literal** - Raw string indices (0 to len(string)-1)
        Includes all characters: sequence chars, gaps, AND region tag characters.
        Use for direct string indexing: `seq.string[literal_pos]`

    **nontag** - Indices excluding region tag characters `<...>`
        Includes sequence chars AND gap/annotation characters, but NOT tag markup.
        Length: `seq.nontag_length`
        Use when working with sequence content ignoring tag syntax.

    **molecular** - Only valid sequence characters (defined by VALID_CHARS)
        Excludes tags, gaps, punctuation, and non-alphabet characters.
        Length: `seq.molecular_length`
        Use for biological position references (e.g., "mutate position 5").

    Example
    -------
    For sequence: 'AC<region>T-G</region>A'

    - literal positions:   0  1  2-11      12 13 14-22       23
                           A  C  <region>  T  -  G  </region> A
    - nontag positions:    0  1            2  3  4            5
                           A  C            T  -  G            A
    - molecular positions: 0  1            2     3            4
                           A  C            T     G            A

    Conversion Methods
    ------------------
    Use `convert_pos()` to convert between any two coordinate systems:
        literal_pos = seq.convert_pos(2, 'molecular', 'literal')  # -> 12
        mol_pos = seq.convert_pos(12, 'literal', 'molecular')     # -> 2

    Returns None if the position doesn't exist in the target system
    (e.g., a gap character has no molecular position).
    """

    # Subclasses override this to define their alphabet for molecular coordinates.
    # Empty frozenset means all non-tag characters are considered molecular.
    VALID_CHARS: ClassVar[frozenset] = frozenset()

    string: str  # Literal string WITH tags
    style: SeqStyle | None  # Per-position styling (None if suppressed)
    # Computed on construction (not lazy, for immutability)
    _clean: str = field(default="", repr=False)
    _regions: tuple = field(default=(), repr=False)
    _nontag_to_literal: tuple = field(default=(), repr=False)
    _molecular_to_literal: tuple = field(default=(), repr=False)
    _literal_to_nontag: tuple = field(default=(), repr=False)
    _literal_to_molecular: tuple = field(default=(), repr=False)

    def __len__(self) -> int:
        """Return length of sequence string."""
        return len(self.string)

    def __repr__(self) -> str:
        """String representation."""
        num_styles = len(self.style.style_list) if self.style else 0
        return f"Seq(len={len(self.string)}, styles={num_styles})"

    # Coordinate system properties
    @property
    def literal_length(self) -> int:
        """Length in literal coordinates (same as len(self.string))."""
        return len(self.string)

    @property
    def nontag_length(self) -> int:
        """Length in nontag coordinates (excludes tag markup)."""
        self._ensure_coord_maps()
        return len(self._nontag_to_literal)

    @property
    def molecular_length(self) -> int:
        """Length in molecular coordinates (DNA characters only)."""
        self._ensure_coord_maps()
        return len(self._molecular_to_literal)

    @property
    def clean(self) -> str:
        """Sequence with all tags removed (nontag content as string)."""
        return self._clean

    @property
    def regions(self) -> tuple:
        """Parsed region tags in this sequence."""
        return self._regions

    def _ensure_coord_maps(self) -> None:
        """Compute coordinate maps if not already computed (lazy initialization).

        Note: We check for both _NOT_COMPUTED sentinel and empty tuple () because
        Seq objects created via direct constructor Seq(string, style) have empty
        tuple defaults that need to trigger computation.

        Subclasses define VALID_CHARS for their alphabet. If VALID_CHARS is empty,
        all non-tag characters are considered molecular positions.
        """
        if self._nontag_to_literal is _NOT_COMPUTED or (
            self._nontag_to_literal == () and len(self.string) > 0
        ):
            from . import parsing_utils

            string = self.string
            n = len(string)
            valid_chars = self.VALID_CHARS

            # Fast path: no tags possible if no '<' character
            if "<" not in string:
                # For tag-free strings: nontag coords == literal coords
                identity_map = tuple(range(n))

                # molecular_to_literal: only valid alphabet characters
                molecular_positions = []
                if valid_chars:
                    # Use subclass-defined alphabet
                    for i, c in enumerate(string):
                        if c in valid_chars:
                            molecular_positions.append(i)
                else:
                    # Empty VALID_CHARS: all characters are molecular
                    molecular_positions = list(range(n))
                molecular_to_literal = tuple(molecular_positions)

                # literal_to_molecular: reverse mapping (None for non-alphabet chars)
                literal_to_molecular_list = [None] * n
                for mol_idx, lit_pos in enumerate(molecular_to_literal):
                    literal_to_molecular_list[lit_pos] = mol_idx
                literal_to_molecular = tuple(literal_to_molecular_list)

                object.__setattr__(self, "_nontag_to_literal", identity_map)
                object.__setattr__(self, "_molecular_to_literal", molecular_to_literal)
                object.__setattr__(self, "_literal_to_nontag", identity_map)
                object.__setattr__(self, "_literal_to_molecular", literal_to_molecular)
            else:
                # Has tags - need full parsing
                # 1. nontag_to_literal: map nontag positions to literal positions
                nontag_positions = parsing_utils.get_nontag_positions(string)
                nontag_to_literal = tuple(nontag_positions)

                # 2. molecular_to_literal: map molecular positions to literal
                molecular_positions = []
                if valid_chars:
                    # Use subclass-defined alphabet
                    for lit_pos in nontag_positions:
                        char = string[lit_pos]
                        if char in valid_chars:
                            molecular_positions.append(lit_pos)
                else:
                    # Empty VALID_CHARS: all nontag characters are molecular
                    molecular_positions = list(nontag_positions)
                molecular_to_literal = tuple(molecular_positions)

                # 3. literal_to_nontag: map literal positions to nontag (None for tag chars)
                literal_to_nontag_list = [None] * n
                for nontag_idx, lit_pos in enumerate(nontag_to_literal):
                    literal_to_nontag_list[lit_pos] = nontag_idx
                literal_to_nontag = tuple(literal_to_nontag_list)

                # 4. literal_to_molecular: map literal positions to molecular
                literal_to_molecular_list = [None] * n
                for mol_idx, lit_pos in enumerate(molecular_to_literal):
                    literal_to_molecular_list[lit_pos] = mol_idx
                literal_to_molecular = tuple(literal_to_molecular_list)

                object.__setattr__(self, "_nontag_to_literal", nontag_to_literal)
                object.__setattr__(self, "_molecular_to_literal", molecular_to_literal)
                object.__setattr__(self, "_literal_to_nontag", literal_to_nontag)
                object.__setattr__(self, "_literal_to_molecular", literal_to_molecular)

    # Coordinate conversion methods
    def convert_pos(self, pos: int, from_coord: CoordSystem, to_coord: CoordSystem) -> int | None:
        """Convert a position between coordinate systems.

        Parameters
        ----------
        pos : int
            Position in the source coordinate system.
        from_coord : {'literal', 'nontag', 'molecular'}
            Source coordinate system.
        to_coord : {'literal', 'nontag', 'molecular'}
            Target coordinate system.

        Returns
        -------
        int | None
            Position in target coordinate system, or None if the position
            doesn't exist in the target system (e.g., a gap has no molecular pos).

        Raises
        ------
        IndexError
            If pos is out of range for the source coordinate system.

        Examples
        --------
        >>> seq = Seq.from_string('AC<region>T-G</region>A')
        >>> seq.convert_pos(2, 'molecular', 'literal')  # T is at literal 12
        12
        >>> seq.convert_pos(13, 'literal', 'molecular')  # '-' has no molecular pos
        None
        >>> seq.convert_pos(3, 'nontag', 'molecular')   # '-' at nontag 3
        None
        """
        # Same coordinate system
        if from_coord == to_coord:
            return pos

        # Ensure coordinate maps are computed
        self._ensure_coord_maps()

        # Convert to literal first if needed
        if from_coord == "nontag":
            if pos < 0 or pos >= len(self._nontag_to_literal):
                raise IndexError(
                    f"nontag position {pos} out of range [0, {len(self._nontag_to_literal)})"
                )
            literal_pos = self._nontag_to_literal[pos]
        elif from_coord == "molecular":
            if pos < 0 or pos >= len(self._molecular_to_literal):
                raise IndexError(
                    f"molecular position {pos} out of range [0, {len(self._molecular_to_literal)})"
                )
            literal_pos = self._molecular_to_literal[pos]
        else:  # from_coord == 'literal'
            if pos < 0 or pos >= len(self.string):
                raise IndexError(f"literal position {pos} out of range [0, {len(self.string)})")
            literal_pos = pos

        # Convert from literal to target
        if to_coord == "literal":
            return literal_pos
        elif to_coord == "nontag":
            return self._literal_to_nontag[literal_pos]
        else:  # to_coord == 'molecular'
            return self._literal_to_molecular[literal_pos]

    def molecular_to_literal(self, pos: int) -> int:
        """Convert molecular position to literal position."""
        self._ensure_coord_maps()
        if pos < 0 or pos >= len(self._molecular_to_literal):
            raise IndexError(
                f"molecular position {pos} out of range [0, {len(self._molecular_to_literal)})"
            )
        return self._molecular_to_literal[pos]

    def nontag_to_literal(self, pos: int) -> int:
        """Convert nontag position to literal position."""
        self._ensure_coord_maps()
        if pos < 0 or pos >= len(self._nontag_to_literal):
            raise IndexError(
                f"nontag position {pos} out of range [0, {len(self._nontag_to_literal)})"
            )
        return self._nontag_to_literal[pos]

    def literal_to_molecular(self, pos: int) -> int | None:
        """Convert literal position to molecular position (None if not DNA)."""
        self._ensure_coord_maps()
        if pos < 0 or pos >= len(self._literal_to_molecular):
            raise IndexError(
                f"literal position {pos} out of range [0, {len(self._literal_to_molecular)})"
            )
        return self._literal_to_molecular[pos]

    def literal_to_nontag(self, pos: int) -> int | None:
        """Convert literal position to nontag position (None if inside tag)."""
        self._ensure_coord_maps()
        if pos < 0 or pos >= len(self._literal_to_nontag):
            raise IndexError(
                f"literal position {pos} out of range [0, {len(self._literal_to_nontag)})"
            )
        return self._literal_to_nontag[pos]

    def convert_positions(
        self,
        positions: Sequence[int],
        from_coord: CoordSystem,
        to_coord: CoordSystem,
    ) -> np.ndarray:
        """Convert multiple positions, returning numpy array.

        Parameters
        ----------
        positions : Sequence[int]
            Positions to convert.
        from_coord : {'literal', 'nontag', 'molecular'}
            Source coordinate system.
        to_coord : {'literal', 'nontag', 'molecular'}
            Target coordinate system.

        Returns
        -------
        np.ndarray
            Array of converted positions (None values remain as None).
        """
        return np.array([self.convert_pos(p, from_coord, to_coord) for p in positions])

    # Region operations
    def has_region(self, name: str) -> bool:
        """Check if a region with the given name exists."""
        return any(r.name == name for r in self._regions)

    def get_region(self, name: str):
        """Get a region by name.

        Returns
        -------
        ParsedRegion
            The region with the given name.

        Raises
        ------
        ValueError
            If region not found.
        """
        for r in self._regions:
            if r.name == name:
                return r
        raise ValueError(f"Region '{name}' not found")

    def split_at_region(self, name: str) -> tuple["Seq", "Seq", "Seq"]:
        """Split sequence at a named region.

        Returns (prefix, content, suffix) where:
        - prefix: Seq before the region (tags removed)
        - content: Seq of region content (tags removed)
        - suffix: Seq after the region (tags removed)

        Raises
        ------
        ValueError
            If region not found or appears multiple times.
        """
        from . import parsing_utils

        region = parsing_utils.validate_single_region(self.string, name)

        prefix_str = self.string[: region.start]
        content_str = region.content
        suffix_str = self.string[region.end :]

        # Create Seq objects with empty coordinate maps (will be computed on construction)
        prefix = Seq.from_string(prefix_str, style=None)
        content = Seq.from_string(content_str, style=None)
        suffix = Seq.from_string(suffix_str, style=None)

        return prefix, content, suffix

    def extract_region(self, name: str) -> "Seq":
        """Extract region content as a new Seq.

        Parameters
        ----------
        name : str
            Region name to extract.

        Returns
        -------
        Seq
            New Seq containing just the region content (tags removed).

        Raises
        ------
        ValueError
            If region not found.
        """
        _, content, _ = self.split_at_region(name)
        return content

    def __getitem__(self, key: slice) -> "Seq":
        """Atomic slicing of string + style.

        Parameters
        ----------
        key : slice
            Slice to extract from sequence.

        Returns
        -------
        Seq
            New Seq with sliced string and style (coordinate maps cleared).

        Examples
        --------
        >>> seq = Seq.from_string('ACGTACGT')
        >>> seq[2:6]
        Seq(len=4, styles=0)

        Notes
        -----
        Slicing may create partial/invalid tags. Coordinate maps are cleared
        and will need to be rebuilt if needed via from_string().
        """
        if not isinstance(key, slice):
            raise TypeError("Seq only supports slice indexing")
        # Don't parse tags on sliced strings (may create partial tags)
        # Return Seq with empty coordinate maps
        seq = Seq.__new__(Seq)
        object.__setattr__(seq, "string", self.string[key])
        object.__setattr__(seq, "style", self.style[key] if self.style is not None else None)
        object.__setattr__(seq, "_clean", "")
        object.__setattr__(seq, "_regions", ())
        object.__setattr__(seq, "_nontag_to_literal", ())
        object.__setattr__(seq, "_molecular_to_literal", ())
        object.__setattr__(seq, "_literal_to_nontag", ())
        object.__setattr__(seq, "_literal_to_molecular", ())
        return seq

    @classmethod
    def join(cls, input_seqs: Sequence["Seq"], sep: str = "") -> "Seq":
        """Join multiple Seq with optional separator.

        Handles string join and SeqStyle.join with automatic offset handling.
        Coordinate maps are rebuilt.

        Parameters
        ----------
        input_seqs : Sequence[Seq]
            Sequence of Seq objects to join.
        sep : str, default=''
            Separator string to insert between sequences.

        Returns
        -------
        Seq
            New Seq with joined strings and styles.

        Examples
        --------
        >>> a = Seq.from_string('ACG')
        >>> b = Seq.from_string('TGC')
        >>> Seq.join([a, b], sep='N')
        Seq(len=7, styles=0)
        """
        if not input_seqs:
            return cls.empty()

        strings = []
        styles = []

        for i, s in enumerate(input_seqs):
            if i > 0 and sep:
                strings.append(sep)
                if s.style is not None:
                    styles.append(SeqStyle.empty(len(sep)))
            strings.append(s.string)
            if s.style is not None:
                styles.append(s.style)

        # Use from_string to rebuild coordinate maps
        # If any input has None style, result has None style
        if any(s.style is None for s in input_seqs):
            result_style = None
        else:
            result_style = SeqStyle.join(styles)

        return cls.from_string(
            string="".join(strings),
            style=result_style,
        )

    @classmethod
    def _join_fast(cls, input_seqs: Sequence["Seq"]) -> "Seq":
        """Fast join for tag-free segments - skips coordinate map computation.

        Use when joining segments that are known to be tag-free (e.g., slices
        from recombine operations). Does not support separators.
        """
        if not input_seqs:
            return cls.empty()

        # Join strings directly
        string = "".join(s.string for s in input_seqs)

        # Join styles if present
        if any(s.style is None for s in input_seqs):
            result_style = None
        else:
            styles = [s.style for s in input_seqs]
            result_style = SeqStyle.join(styles)

        # Construct Seq without reparsing (coordinate maps computed lazily)
        seq = cls.__new__(cls)
        object.__setattr__(seq, "string", string)
        object.__setattr__(seq, "style", result_style)
        object.__setattr__(seq, "_clean", string)  # No tags
        object.__setattr__(seq, "_regions", ())
        object.__setattr__(seq, "_nontag_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_molecular_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_nontag", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_molecular", _NOT_COMPUTED)
        return seq

    @classmethod
    def empty(cls) -> "Seq":
        """Create empty Seq.

        Returns
        -------
        Seq
            Empty Seq with no string or style.
        """
        return cls.from_string("", SeqStyle.empty(0))

    @classmethod
    def from_string(cls, string: str, style: SeqStyle | None = None) -> "Seq":
        """Create Seq from string, parsing tags and building coordinate maps.

        Parameters
        ----------
        string : str
            DNA sequence string (may contain region tags).
        style : SeqStyle | None, default=None
            Optional style. If None, creates empty style (or None if suppressed).

        Returns
        -------
        Seq
            New Seq with parsed regions and coordinate maps.
        """
        from . import parsing_utils
        from .style_utils import styles_suppressed

        if style is None:
            style = None if styles_suppressed() else SeqStyle.empty(len(string))

        # Fast path: no tags possible if no '<' character
        if "<" not in string:
            seq = cls.__new__(cls)
            object.__setattr__(seq, "string", string)
            object.__setattr__(seq, "style", style)
            object.__setattr__(seq, "_clean", string)  # No tags to strip
            object.__setattr__(seq, "_regions", ())  # No regions
            # Coordinate maps computed lazily via _ensure_coord_maps()
            object.__setattr__(seq, "_nontag_to_literal", _NOT_COMPUTED)
            object.__setattr__(seq, "_molecular_to_literal", _NOT_COMPUTED)
            object.__setattr__(seq, "_literal_to_nontag", _NOT_COMPUTED)
            object.__setattr__(seq, "_literal_to_molecular", _NOT_COMPUTED)
            return seq

        # Parse regions (skip validation since fast path handles tag-free case)
        regions = tuple(parsing_utils.find_all_regions(string, _skip_validation=True))

        # Strip tags to get clean content
        clean = parsing_utils.strip_all_tags(string)

        # Use object.__setattr__ for frozen dataclass
        seq = cls.__new__(cls)
        object.__setattr__(seq, "string", string)
        object.__setattr__(seq, "style", style)
        object.__setattr__(seq, "_clean", clean)
        object.__setattr__(seq, "_regions", regions)
        # Coordinate maps computed lazily via _ensure_coord_maps()
        object.__setattr__(seq, "_nontag_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_molecular_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_nontag", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_molecular", _NOT_COMPUTED)

        return seq

    def insert(self, pos: int, content: "Seq") -> "Seq":
        """Insert content at position.

        Parameters
        ----------
        pos : int
            Position to insert at (0-indexed).
        content : Seq
            Seq to insert.

        Returns
        -------
        Seq
            New Seq with content inserted.
        """
        return Seq.join([self[:pos], content, self[pos:]])

    def with_style(self, style: SeqStyle | None) -> "Seq":
        """Return copy with updated style (preserves coordinate maps).

        Parameters
        ----------
        style : SeqStyle | None
            New style for the sequence.

        Returns
        -------
        Seq
            New Seq with updated style.
        """
        # Preserve coordinate maps without rebuilding
        seq = Seq.__new__(Seq)
        object.__setattr__(seq, "string", self.string)
        object.__setattr__(seq, "style", style)
        object.__setattr__(seq, "_clean", self._clean)
        object.__setattr__(seq, "_regions", self._regions)
        object.__setattr__(seq, "_nontag_to_literal", self._nontag_to_literal)
        object.__setattr__(seq, "_molecular_to_literal", self._molecular_to_literal)
        object.__setattr__(seq, "_literal_to_nontag", self._literal_to_nontag)
        object.__setattr__(seq, "_literal_to_molecular", self._literal_to_molecular)
        return seq

    def add_style(self, style_spec: str, positions) -> "Seq":
        """Return copy with additional style added (preserves coordinate maps).

        Parameters
        ----------
        style_spec : str
            Style specification (e.g., 'red', 'bold', '#ff0000').
        positions : np.ndarray
            Positions to apply style.

        Returns
        -------
        Seq
            New Seq with added style.
        """
        if self.style is None:
            return self  # If styles suppressed, don't add any
        new_style = self.style.add_style(style_spec, positions)
        return self.with_style(new_style)


class NullSeq(Seq):
    """Singleton sentinel indicating a filtered/rejected sequence.

    NullSeq propagates through the DAG - any operation receiving a NullSeq
    parent will produce NullSeq output. Used by filter operations to mark
    sequences that should be excluded from the final library.

    NullSeq is falsy: `if seq:` returns False for NullSeq instances.
    """

    _instance = None

    def __new__(cls):
        """Return singleton instance."""
        if cls._instance is None:
            inst = object.__new__(cls)
            # Set frozen dataclass fields via object.__setattr__
            object.__setattr__(inst, "string", "")
            object.__setattr__(inst, "style", None)
            object.__setattr__(inst, "_clean", "")
            object.__setattr__(inst, "_regions", ())
            object.__setattr__(inst, "_nontag_to_literal", ())
            object.__setattr__(inst, "_molecular_to_literal", ())
            object.__setattr__(inst, "_literal_to_nontag", ())
            object.__setattr__(inst, "_literal_to_molecular", ())
            cls._instance = inst
        return cls._instance

    def __init__(self, string=None, style=None, **kwargs):
        """Override to prevent dataclass __init__ from running."""
        # Do nothing - fields are set in __new__
        pass

    def __bool__(self) -> bool:
        """NullSeq is falsy."""
        return False

    def __repr__(self) -> str:
        return "NullSeq()"

    def __len__(self) -> int:
        return 0


def is_null_seq(seq) -> bool:
    """Check if a Seq is NullSeq."""
    return isinstance(seq, NullSeq)
