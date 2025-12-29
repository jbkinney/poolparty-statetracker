"""Utility functions for poolparty."""
from .types import Sequence, Union
import pandas as pd

# Re-export alphabet-related functions for backwards compatibility
from .alphabet import (
    reverse_complement,
    reverse_complement_iupac,
    named_alphabets_dict,
    get_alphabet,
    validate_alphabet,
    IUPAC_TO_DNA,
    iupac_to_prob_df,
)


def reset_op_id_counter() -> None:
    """Reset the operation ID counter to 0."""
    from .operation import Operation  # Late import to avoid circular dependency
    Operation.id_counter = 0


def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two equal-length strings.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Number of positions at which the corresponding characters differ.
    
    Raises:
        ValueError: If strings have different lengths.
    """
    if len(s1) != len(s2):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings.
    
    The edit distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform
    one string into another.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Minimum number of edits required to transform s1 into s2.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)
    for j in range(1, n + 1):
        curr_row[0] = j
        for i in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                curr_row[i] = prev_row[i-1]
            else:
                curr_row[i] = 1 + min(
                    prev_row[i],      # deletion
                    curr_row[i-1],    # insertion
                    prev_row[i-1]     # substitution
                )
        prev_row, curr_row = curr_row, prev_row
    return prev_row[m]


def max_homopolymer_length(seq: str) -> int:
    """Calculate the maximum homopolymer run length in a sequence.
    
    A homopolymer is a consecutive run of identical characters.
    
    Args:
        seq: Input sequence string
    
    Returns:
        Length of the longest homopolymer run. Returns 0 for empty string,
        1 for strings with no consecutive identical characters.
    
    Example:
        >>> max_homopolymer_length("AAACCCGGG")
        3
        >>> max_homopolymer_length("ACGTACGT")
        1
        >>> max_homopolymer_length("")
        0
    """
    if not seq:
        return 0
    
    max_run = 1
    current_run = 1
    
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    return max_run


def gc_content(seq: str) -> float:
    """Calculate GC content of a sequence.
    
    GC content is the fraction of bases that are G or C (case-insensitive).
    
    Args:
        seq: Input sequence string
    
    Returns:
        GC content as a fraction between 0.0 and 1.0. Returns 0.0 for empty string.
    
    Example:
        >>> gc_content("ACGT")
        0.5
        >>> gc_content("AAAA")
        0.0
        >>> gc_content("GGCC")
        1.0
    """
    if not seq:
        return 0.0
    
    gc_count = sum(1 for base in seq.upper() if base in ('G', 'C'))
    return gc_count / len(seq)
