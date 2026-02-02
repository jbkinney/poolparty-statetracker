"""DnaSeq class for DNA sequences with reverse complement support."""

from dataclasses import dataclass
from typing import ClassVar

from . import dna_utils
from .seq import _NOT_COMPUTED, Seq
from .style_utils import SeqStyle, styles_suppressed


@dataclass(frozen=True)
class DnaSeq(Seq):
    """DNA sequence with reverse complement support."""

    VALID_CHARS: ClassVar[frozenset] = dna_utils.IUPAC_CHARS

    def __repr__(self) -> str:
        """String representation."""
        num_styles = len(self.style.style_list) if self.style else 0
        return f"DnaSeq(len={len(self.string)}, styles={num_styles})"

    def reversed(self, do_reverse: bool = True) -> "DnaSeq":
        """Reverse complement the sequence and mirror style positions.

        Parameters
        ----------
        do_reverse : bool, default=True
            If False, returns self unchanged (convenient for conditional reversal).

        Returns
        -------
        DnaSeq
            New DnaSeq with reverse complemented string and mirrored style positions.
        """
        if not do_reverse:
            return self

        reversed_string = dna_utils.reverse_complement(self.string)
        reversed_style = self.style.reversed(do_reverse=True)

        return DnaSeq.from_string(reversed_string, reversed_style)

    def rc(self, do_rc: bool = True) -> "DnaSeq":
        """Alias for reversed() - reverse complement the sequence.

        Parameters
        ----------
        do_rc : bool, default=True
            If False, returns self unchanged (convenient for conditional reversal).

        Returns
        -------
        DnaSeq
            New DnaSeq with reverse complemented string and mirrored style positions.
        """
        return self.reversed(do_rc)

    @classmethod
    def from_string(cls, string: str, style: SeqStyle | None = None) -> "DnaSeq":
        """Create DnaSeq from string, parsing tags and building coordinate maps.

        Parameters
        ----------
        string : str
            DNA sequence string (may contain region tags).
        style : SeqStyle | None, default=None
            Optional style. If None, creates empty style (or None if suppressed).

        Returns
        -------
        DnaSeq
            New DnaSeq with parsed regions and coordinate maps.
        """
        from . import parsing_utils

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

    @classmethod
    def empty(cls) -> "DnaSeq":
        """Create empty DnaSeq.

        Returns
        -------
        DnaSeq
            Empty DnaSeq with no string or style.
        """
        return cls.from_string("", SeqStyle.empty(0))

    def __getitem__(self, key: slice) -> "DnaSeq":
        """Atomic slicing of string + style.

        Parameters
        ----------
        key : slice
            Slice to extract from sequence.

        Returns
        -------
        DnaSeq
            New DnaSeq with sliced string and style (coordinate maps cleared).
        """
        if not isinstance(key, slice):
            raise TypeError("DnaSeq only supports slice indexing")
        # Don't parse tags on sliced strings (may create partial tags)
        # Return DnaSeq with empty coordinate maps
        seq = DnaSeq.__new__(DnaSeq)
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
    def join(cls, input_seqs, sep: str = "") -> "DnaSeq":
        """Join multiple DnaSeq with optional separator.

        Parameters
        ----------
        input_seqs : Sequence[DnaSeq]
            Sequence of DnaSeq objects to join.
        sep : str, default=''
            Separator string to insert between sequences.

        Returns
        -------
        DnaSeq
            New DnaSeq with joined strings and styles.
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
    def _join_fast(cls, input_seqs) -> "DnaSeq":
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

        # Construct DnaSeq without reparsing (coordinate maps computed lazily)
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

    def with_style(self, style: SeqStyle | None) -> "DnaSeq":
        """Return copy with updated style (preserves coordinate maps).

        Parameters
        ----------
        style : SeqStyle | None
            New style for the sequence.

        Returns
        -------
        DnaSeq
            New DnaSeq with updated style.
        """
        # Preserve coordinate maps without rebuilding
        seq = DnaSeq.__new__(DnaSeq)
        object.__setattr__(seq, "string", self.string)
        object.__setattr__(seq, "style", style)
        object.__setattr__(seq, "_clean", self._clean)
        object.__setattr__(seq, "_regions", self._regions)
        object.__setattr__(seq, "_nontag_to_literal", self._nontag_to_literal)
        object.__setattr__(seq, "_molecular_to_literal", self._molecular_to_literal)
        object.__setattr__(seq, "_literal_to_nontag", self._literal_to_nontag)
        object.__setattr__(seq, "_literal_to_molecular", self._literal_to_molecular)
        return seq

    def add_style(self, style_spec: str, positions) -> "DnaSeq":
        """Return copy with additional style added (preserves coordinate maps).

        Parameters
        ----------
        style_spec : str
            Style specification (e.g., 'red', 'bold', '#ff0000').
        positions : np.ndarray
            Positions to apply style.

        Returns
        -------
        DnaSeq
            New DnaSeq with added style.
        """
        if self.style is None:
            return self  # If styles suppressed, don't add any
        new_style = self.style.add_style(style_spec, positions)
        return self.with_style(new_style)

    def insert(self, pos: int, content: "DnaSeq") -> "DnaSeq":
        """Insert content at position.

        Parameters
        ----------
        pos : int
            Position to insert at (0-indexed).
        content : DnaSeq
            DnaSeq to insert.

        Returns
        -------
        DnaSeq
            New DnaSeq with content inserted.
        """
        return DnaSeq.join([self[:pos], content, self[pos:]])
