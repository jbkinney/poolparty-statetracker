"""DNA sequence constants and utilities for poolparty."""
from ..types import beartype

# Canonical DNA bases
BASES: list[str] = ['A', 'C', 'G', 'T']

# Watson-Crick complement (both cases, including IUPAC codes)
COMPLEMENT: dict[str, str] = {
    # Canonical bases
    'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
    'a': 't', 't': 'a', 'g': 'c', 'c': 'g',
    # IUPAC ambiguity codes
    'R': 'Y', 'Y': 'R',  # R=AG, Y=CT
    'S': 'S',            # S=GC (self-complementary)
    'W': 'W',            # W=AT (self-complementary)
    'K': 'M', 'M': 'K',  # K=GT, M=AC
    'B': 'V', 'V': 'B',  # B=CGT, V=ACG
    'D': 'H', 'H': 'D',  # D=AGT, H=ACT
    'N': 'N',            # N=ACGT (self-complementary)
    # Lowercase IUPAC
    'r': 'y', 'y': 'r',
    's': 's', 'w': 'w',
    'k': 'm', 'm': 'k',
    'b': 'v', 'v': 'b',
    'd': 'h', 'h': 'd',
    'n': 'n',
}

# IUPAC nucleotide codes to DNA bases mapping
IUPAC_TO_DNA: dict[str, list[str]] = {
    "A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"],
    "U": ["T"],  # U = T in DNA context
    "R": ["A", "G"], "Y": ["C", "T"], "S": ["G", "C"], "W": ["A", "T"],
    "K": ["G", "T"], "M": ["A", "C"],
    "B": ["C", "G", "T"], "D": ["A", "G", "T"],
    "H": ["A", "C", "T"], "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}
# Add lowercase versions (lowercase keys -> lowercase values)
IUPAC_TO_DNA.update({
    k.lower(): [v.lower() for v in vals]
    for k, vals in list(IUPAC_TO_DNA.items())
})

# Ignore chars: punctuation safe in sequence content
# Only < and > are prohibited (they delimit XML tags)
IGNORE_CHARS: frozenset[str] = frozenset('-._ *:;,~`!@#$%^&()[]{}\\|?+/\'"=')

# All valid DNA characters (bases + lowercase)
VALID_CHARS: frozenset[str] = frozenset('ACGTacgt')

# All valid IUPAC characters
IUPAC_CHARS: frozenset[str] = frozenset(IUPAC_TO_DNA.keys())

# Pre-computed mutation targets: wt_char -> list of mutation targets
MUTATIONS_DICT: dict[str, list[str]] = {
    'A': ['C', 'G', 'T'], 'C': ['A', 'G', 'T'],
    'G': ['A', 'C', 'T'], 'T': ['A', 'C', 'G'],
    'a': ['c', 'g', 't'], 'c': ['a', 'g', 't'],
    'g': ['a', 'c', 't'], 't': ['a', 'c', 'g'],
}


@beartype
def complement(char: str) -> str:
    """Get Watson-Crick complement of a character. Non-DNA chars pass through."""
    return COMPLEMENT.get(char, char)


@beartype
def reverse_complement(seq: str) -> str:
    """Get reverse complement of a DNA sequence."""
    return ''.join(complement(c) for c in reversed(seq))


@beartype
def get_mutations(char: str) -> list[str]:
    """Get list of bases a character can mutate to (case-preserving)."""
    c_upper = char.upper()
    if c_upper not in 'ACGT':
        raise ValueError(f"Cannot mutate non-DNA character: {char!r}")
    others = [b for b in BASES if b != c_upper]
    if char.islower():
        return [b.lower() for b in others]
    return others


@beartype
def get_nontag_positions(seq: str) -> list[int]:
    """Get raw string positions of all chars excluding region tag interiors."""
    from .parsing_utils import get_nontag_positions as _get_nontag_positions
    return _get_nontag_positions(seq)


@beartype
def get_length_without_tags(seq: str) -> int:
    """Get sequence length excluding region tags (includes all other chars)."""
    from .parsing_utils import get_length_without_tags as _get_length_without_tags
    return _get_length_without_tags(seq)


@beartype
def get_molecular_positions(seq: str) -> list[int]:
    """Get positions of valid DNA characters, excluding gaps and region tags."""
    nontag_positions = get_nontag_positions(seq)
    return [i for i in nontag_positions if seq[i] in VALID_CHARS]


@beartype
def get_seq_length(seq: str) -> int:
    """Get count of valid DNA characters (excludes gaps and region tags)."""
    return len(get_molecular_positions(seq))
