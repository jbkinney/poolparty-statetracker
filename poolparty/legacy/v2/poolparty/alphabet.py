"""Alphabet-related constants and functions for poolparty.

This module provides:
- Named alphabets (DNA, RNA, protein)
- Alphabet validation and normalization
- Complement mappings (basic and IUPAC)
- IUPAC nucleotide code utilities
"""
from .types import Sequence, Union
import pandas as pd


# Complement mapping for reverse complement (basic DNA only)
_COMPLEMENT = str.maketrans('ACGTacgt', 'TGCAtgca')


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.
    
    Args:
        seq: DNA sequence string (case-insensitive)
    
    Returns:
        Reverse complement of the input sequence
    
    Example:
        >>> reverse_complement("ACGT")
        'ACGT'
        >>> reverse_complement("AAAA")
        'TTTT'
    """
    return seq.translate(_COMPLEMENT)[::-1]


# IUPAC complement mapping (full alphabet)
# A↔T, C↔G, U→A, R↔Y, S↔S, W↔W, K↔M, B↔V, D↔H, N↔N
_IUPAC_COMPLEMENT = str.maketrans(
    'ACGTUacgtuRYSWKMBDHVNryswkmbdhvn',
    'TGCAAtgcaaYRSWMKVHDBNyrswmkvhdbn'
)


def reverse_complement_iupac(seq: str) -> str:
    """Return the reverse complement of a DNA sequence using IUPAC alphabet.
    
    Supports the full IUPAC DNA alphabet including ambiguity codes.
    Characters not in the IUPAC alphabet are passed through unchanged.
    
    Args:
        seq: DNA sequence string (case-insensitive)
    
    Returns:
        Reverse complement of the input sequence
    
    IUPAC Complement Pairs:
        A ↔ T, C ↔ G, U → A (RNA uracil)
        R (A,G) ↔ Y (C,T)
        S (G,C) ↔ S (self-complementary)
        W (A,T) ↔ W (self-complementary)
        K (G,T) ↔ M (A,C)
        B (C,G,T) ↔ V (A,C,G)
        D (A,G,T) ↔ H (A,C,T)
        N (any) ↔ N (self-complementary)
    
    Example:
        >>> reverse_complement_iupac("ACGT")
        'ACGT'
        >>> reverse_complement_iupac("AAAA")
        'TTTT'
        >>> reverse_complement_iupac("RY")
        'RY'
        >>> reverse_complement_iupac("ACGT-NNN")  # Non-IUPAC chars preserved
        'NNN-ACGT'
    """
    return seq.translate(_IUPAC_COMPLEMENT)[::-1]


# Define named alphabets
named_alphabets_dict = {
    "dna": list("ACGT"),
    "rna": list("ACGU"),
    "protein": list("ACDEFGHIKLMNPQRSTVWY"),
    "protein*": list("*ACDEFGHIKLMNPQRSTVWY"),
}


def get_alphabet(name: str) -> list[str]:
    """Get an alphabet by name.
    
    Args:
        name: Name of the alphabet ("dna", "rna", "protein", "protein*").
    
    Returns:
        List of alphabet characters.
    
    Raises:
        KeyError: If the provided name is not a valid alphabet name.
    """
    if name not in named_alphabets_dict:
        raise KeyError(
            f"Alphabet '{name}' is not a valid named alphabet. "
            f"Valid options are: {', '.join(named_alphabets_dict.keys())}"
        )
    return named_alphabets_dict[name]


def validate_alphabet(alphabet: Union[str, Sequence[str]] = 'dna') -> list[str]:
    """Validate and normalize an alphabet parameter.
    
    Args:
        alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
            or a list of single-character strings to use as the alphabet.
            Default: 'dna'
    
    Returns:
        List of alphabet characters.
    
    Raises:
        KeyError: If a string is provided that is not a valid alphabet name.
        ValueError: If a list is provided but contains non-string elements,
            strings that are not exactly length 1, or duplicate characters.
        TypeError: If alphabet is neither a string nor a list.
    """
    if isinstance(alphabet, str):
        # String input - look up in named alphabets
        return get_alphabet(alphabet)
    elif isinstance(alphabet, list):
        # List input - validate all elements
        if len(alphabet) == 0:
            raise ValueError("alphabet list must be non-empty")
        
        for i, char in enumerate(alphabet):
            if not isinstance(char, str):
                raise ValueError(
                    f"All elements of alphabet list must be strings, "
                    f"but element at index {i} is {type(char).__name__}"
                )
            if len(char) != 1:
                raise ValueError(
                    f"All elements of alphabet list must be single characters, "
                    f"but element at index {i} is '{char}' (length {len(char)})"
                )
        
        # Check for uniqueness
        if len(alphabet) != len(set(alphabet)):
            duplicates = [char for char in set(alphabet) if alphabet.count(char) > 1]
            raise ValueError(
                f"All elements of alphabet list must be unique, "
                f"but found duplicates: {duplicates}"
            )
        
        return list(alphabet)
    else:
        raise TypeError(
            f"alphabet must be either a str (alphabet name) or list[str] (list of characters), "
            f"got {type(alphabet).__name__}"
        )


# IUPAC nucleotide codes to DNA bases mapping
IUPAC_TO_DNA = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "U": ["T"],  # U = T in DNA context
    "R": ["A", "G"],           # purine
    "Y": ["C", "T"],           # pyrimidine
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],      # not A
    "D": ["A", "G", "T"],      # not C
    "H": ["A", "C", "T"],      # not G
    "V": ["A", "C", "G"],      # not T
    "N": ["A", "C", "G", "T"], # any base
}


def iupac_to_prob_df(iupac_seq: str) -> pd.DataFrame:
    """Convert an IUPAC string to a probability matrix DataFrame.
    
    Args:
        iupac_seq: A string of IUPAC nucleotide codes.
        
    Returns:
        A DataFrame with columns ['A', 'C', 'G', 'T'] where each row 
        represents a position and contains equal probabilities for 
        the bases defined by each IUPAC code.
        
    Raises:
        ValueError: If iupac_seq contains invalid IUPAC characters.
    
    Example:
        >>> iupac_to_prob_df("AN")
           A     C     G     T
        0  1.0   0.0   0.0   0.0
        1  0.25  0.25  0.25  0.25
    """
    alphabet = ['A', 'C', 'G', 'T']
    
    # Validate all characters are valid IUPAC codes
    invalid_chars = set()
    for char in iupac_seq.upper():
        if char not in IUPAC_TO_DNA:
            invalid_chars.add(char)
    
    if invalid_chars:
        raise ValueError(
            f"Invalid IUPAC character(s): {sorted(invalid_chars)}. "
            f"Valid characters are: {sorted(IUPAC_TO_DNA.keys())}"
        )
    
    # Build probability matrix
    data = {base: [] for base in alphabet}
    for char in iupac_seq.upper():
        possible_bases = IUPAC_TO_DNA[char]
        prob = 1.0 / len(possible_bases)
        for base in alphabet:
            data[base].append(prob if base in possible_bases else 0.0)
    
    return pd.DataFrame(data)

