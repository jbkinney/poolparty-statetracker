"""Alphabet utilities for poolparty.

Provides named alphabets (DNA, RNA, protein) and validation functions.
"""

from .types import AlphabetType, beartype

# Named alphabet definitions
NAMED_ALPHABETS: dict[str, list[str]] = {
    'dna': list('ACGT'),
    'rna': list('ACGU'),
    'protein': list('ACDEFGHIKLMNPQRSTVWY'),
    'binary': list('01'),
}


@beartype
def get_alphabet(alphabet: AlphabetType) -> list[str]:
    """Get an alphabet as a list of characters.
    
    Args:
        alphabet: Either a named alphabet ('dna', 'rna', 'protein', 'binary')
                  or a sequence of characters
    
    Returns:
        List of single characters
    
    Raises:
        ValueError: If alphabet name is unknown or characters are invalid
    """
    if isinstance(alphabet, str):
        if alphabet.lower() in NAMED_ALPHABETS:
            return NAMED_ALPHABETS[alphabet.lower()].copy()
        # Treat as sequence of characters
        return list(alphabet)
    
    # Already a sequence
    chars = list(alphabet)
    
    # Validate all are single characters
    for c in chars:
        if not isinstance(c, str) or len(c) != 1:
            raise ValueError(f"Alphabet must contain single characters, got {c!r}")
    
    if len(chars) < 2:
        raise ValueError(f"Alphabet must have at least 2 characters, got {len(chars)}")
    
    if len(chars) != len(set(chars)):
        raise ValueError("Alphabet contains duplicate characters")
    
    return chars


@beartype
def validate_sequence(seq: str, alphabet: list[str]) -> None:
    """Validate that a sequence contains only alphabet characters.
    
    Args:
        seq: Sequence to validate
        alphabet: Valid characters
    
    Raises:
        ValueError: If sequence contains invalid characters
    """
    alphabet_set = set(alphabet)
    invalid = set(seq) - alphabet_set
    if invalid:
        raise ValueError(
            f"Sequence contains invalid characters: {invalid}. "
            f"Valid alphabet: {''.join(alphabet)}"
        )
