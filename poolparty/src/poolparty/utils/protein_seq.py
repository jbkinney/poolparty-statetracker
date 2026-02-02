"""ProteinSeq class for protein sequences with styling support."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .seq import _NOT_COMPUTED, Seq
from .style_utils import SeqStyle, styles_suppressed

# Valid amino acid characters (single-letter codes + stop codon)
VALID_PROTEIN_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy*")

# 1-letter to 3-letter amino acid code mapping
AA_THREE_LETTER = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "*": "***",  # Stop codon
    # Lowercase versions
    "a": "ala",
    "c": "cys",
    "d": "asp",
    "e": "glu",
    "f": "phe",
    "g": "gly",
    "h": "his",
    "i": "ile",
    "k": "lys",
    "l": "leu",
    "m": "met",
    "n": "asn",
    "p": "pro",
    "q": "gln",
    "r": "arg",
    "s": "ser",
    "t": "thr",
    "v": "val",
    "w": "trp",
    "y": "tyr",
}


def to_three_letter(seq: str, separator: str = "-") -> str:
    """Convert 1-letter amino acid sequence to 3-letter format.

    Parameters
    ----------
    seq : str
        Protein sequence in 1-letter codes.
    separator : str, default="-"
        Separator between 3-letter codes.

    Returns
    -------
    str
        Sequence in 3-letter format (e.g., "Met-Ala-Lys").
    """
    codes = [AA_THREE_LETTER.get(aa, "???") for aa in seq]
    return separator.join(codes)


def map_style_positions_to_three_letter(
    positions: np.ndarray,
    seq_len: int,
    separator: str = "-",
) -> np.ndarray:
    """Map 1-letter style positions to 3-letter positions.

    Each amino acid at position i in the 1-letter sequence maps to positions
    covering the 3-letter code in the expanded sequence.

    Parameters
    ----------
    positions : np.ndarray
        Style positions in 1-letter sequence coordinates.
    seq_len : int
        Length of the 1-letter sequence.
    separator : str, default="-"
        Separator used between 3-letter codes.

    Returns
    -------
    np.ndarray
        Mapped positions for the 3-letter sequence.
    """
    new_positions = []
    sep_len = len(separator)
    for pos in positions:
        base = pos * (3 + sep_len)
        new_positions.extend([base, base + 1, base + 2])
    return np.array(new_positions, dtype=np.int64)


@dataclass(frozen=True)
class ProteinSeq(Seq):
    """Protein sequence with styling support.

    Subclass of Seq specialized for amino acid sequences. Uses protein alphabet
    for molecular coordinate computation instead of DNA alphabet.
    """

    VALID_CHARS: ClassVar[frozenset] = VALID_PROTEIN_CHARS

    def __repr__(self) -> str:
        """String representation."""
        num_styles = len(self.style.style_list) if self.style else 0
        return f"ProteinSeq(len={len(self.string)}, styles={num_styles})"

    def _ensure_coord_maps(self) -> None:
        """Compute coordinate maps using protein alphabet."""
        if self._nontag_to_literal is _NOT_COMPUTED or (
            self._nontag_to_literal == () and len(self.string) > 0
        ):
            from . import parsing_utils

            string = self.string
            n = len(string)

            # Fast path: no tags possible if no '<' character
            if "<" not in string:
                # For tag-free strings: nontag coords == literal coords
                identity_map = tuple(range(n))

                # molecular_to_literal: only valid amino acid characters
                molecular_positions = []
                for i, c in enumerate(string):
                    if c in VALID_PROTEIN_CHARS:
                        molecular_positions.append(i)
                molecular_to_literal = tuple(molecular_positions)

                # literal_to_molecular: reverse mapping (None for non-AA chars)
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
                nontag_positions = parsing_utils.get_nontag_positions(string)
                nontag_to_literal = tuple(nontag_positions)

                # molecular_to_literal: only valid amino acid characters
                molecular_positions = []
                for lit_pos in nontag_positions:
                    char = string[lit_pos]
                    if char in VALID_PROTEIN_CHARS:
                        molecular_positions.append(lit_pos)
                molecular_to_literal = tuple(molecular_positions)

                # literal_to_nontag: map literal positions to nontag
                literal_to_nontag_list = [None] * n
                for nontag_idx, lit_pos in enumerate(nontag_to_literal):
                    literal_to_nontag_list[lit_pos] = nontag_idx
                literal_to_nontag = tuple(literal_to_nontag_list)

                # literal_to_molecular: map literal positions to molecular
                literal_to_molecular_list = [None] * n
                for mol_idx, lit_pos in enumerate(molecular_to_literal):
                    literal_to_molecular_list[lit_pos] = mol_idx
                literal_to_molecular = tuple(literal_to_molecular_list)

                object.__setattr__(self, "_nontag_to_literal", nontag_to_literal)
                object.__setattr__(self, "_molecular_to_literal", molecular_to_literal)
                object.__setattr__(self, "_literal_to_nontag", literal_to_nontag)
                object.__setattr__(self, "_literal_to_molecular", literal_to_molecular)

    @classmethod
    def from_string(cls, string: str, style: SeqStyle | None = None) -> "ProteinSeq":
        """Create ProteinSeq from string.

        Parameters
        ----------
        string : str
            Protein sequence string (single-letter amino acid codes).
        style : SeqStyle | None, default=None
            Optional style. If None, creates empty style (or None if suppressed).

        Returns
        -------
        ProteinSeq
            New ProteinSeq with coordinate maps.
        """
        from . import parsing_utils

        if style is None:
            style = None if styles_suppressed() else SeqStyle.empty(len(string))

        # Fast path: no tags
        if "<" not in string:
            seq = cls.__new__(cls)
            object.__setattr__(seq, "string", string)
            object.__setattr__(seq, "style", style)
            object.__setattr__(seq, "_clean", string)
            object.__setattr__(seq, "_regions", ())
            object.__setattr__(seq, "_nontag_to_literal", _NOT_COMPUTED)
            object.__setattr__(seq, "_molecular_to_literal", _NOT_COMPUTED)
            object.__setattr__(seq, "_literal_to_nontag", _NOT_COMPUTED)
            object.__setattr__(seq, "_literal_to_molecular", _NOT_COMPUTED)
            return seq

        # Parse regions
        regions = tuple(parsing_utils.find_all_regions(string, _skip_validation=True))
        clean = parsing_utils.strip_all_tags(string)

        seq = cls.__new__(cls)
        object.__setattr__(seq, "string", string)
        object.__setattr__(seq, "style", style)
        object.__setattr__(seq, "_clean", clean)
        object.__setattr__(seq, "_regions", regions)
        object.__setattr__(seq, "_nontag_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_molecular_to_literal", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_nontag", _NOT_COMPUTED)
        object.__setattr__(seq, "_literal_to_molecular", _NOT_COMPUTED)

        return seq

    @classmethod
    def empty(cls) -> "ProteinSeq":
        """Create empty ProteinSeq."""
        return cls.from_string("", SeqStyle.empty(0))

    def with_style(self, style: SeqStyle | None) -> "ProteinSeq":
        """Return copy with updated style (preserves coordinate maps)."""
        seq = ProteinSeq.__new__(ProteinSeq)
        object.__setattr__(seq, "string", self.string)
        object.__setattr__(seq, "style", style)
        object.__setattr__(seq, "_clean", self._clean)
        object.__setattr__(seq, "_regions", self._regions)
        object.__setattr__(seq, "_nontag_to_literal", self._nontag_to_literal)
        object.__setattr__(seq, "_molecular_to_literal", self._molecular_to_literal)
        object.__setattr__(seq, "_literal_to_nontag", self._literal_to_nontag)
        object.__setattr__(seq, "_literal_to_molecular", self._literal_to_molecular)
        return seq

    def add_style(self, style_spec: str, positions) -> "ProteinSeq":
        """Return copy with additional style added."""
        if self.style is None:
            return self
        new_style = self.style.add_style(style_spec, positions)
        return self.with_style(new_style)
