"""Alphabet utilities for poolparty."""
from .types import beartype, Optional, Sequence, Union

# Named alphabet character lists
NAMED_ALPHABETS: dict[str, list[str]] = {
    'dna': list('ACGT'),
    'rna': list('ACGU'),
    'protein': list('ACDEFGHIKLMNPQRSTVWY'),
    'binary': list('01'),
}

# Watson-Crick complement mappings
DNA_COMPLEMENT: dict[str, str] = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
RNA_COMPLEMENT: dict[str, str] = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
BINARY_COMPLEMENT: dict[str, str] = {'0': '1', '1': '0'}

# Default ignore characters for sequence validation
DEFAULT_IGNORE_CHARS: tuple[str, ...] = ('-', '.', ' ')


@beartype
class Alphabet:
    """Alphabet configuration for sequence operations.
    
    Attributes:
        chars: List of allowed characters
        size: Number of characters in alphabet
        mutation_map: Dict mapping each char to list of chars it can mutate to
        complement: Dict mapping each char to its complement (or None)
        ignore_chars: Set of characters to ignore during validation
    """
    
    def __init__(
        self,
        chars: Sequence[str],
        complement: Optional[dict[str, str]] = None,
        ignore_chars: Sequence[str] = DEFAULT_IGNORE_CHARS,
    ) -> None:
        """Initialize Alphabet with custom characters.
        
        For named alphabets ('dna', 'rna', etc.), use get_alphabet() instead.
        """
        # Validate and store characters
        char_list = list(chars)
        for c in char_list:
            if not isinstance(c, str) or len(c) != 1:
                raise ValueError(f"Alphabet must contain single characters, got {c!r}")
        if len(char_list) < 2:
            raise ValueError(f"Alphabet must have at least 2 characters, got {len(char_list)}")
        if len(char_list) != len(set(char_list)):
            raise ValueError("Alphabet contains duplicate characters")
        
        self.chars: list[str] = char_list
        self.size: int = len(char_list)
        
        # Build mutation map: each char -> all other chars
        self.mutation_map: dict[str, list[str]] = {
            c: [other for other in char_list if other != c]
            for c in char_list
        }
        
        # Store complement (validate if provided)
        if complement is not None:
            for c in char_list:
                if c not in complement:
                    raise ValueError(f"Complement mapping missing for character '{c}'")
                if complement[c] not in char_list:
                    raise ValueError(f"Complement '{complement[c]}' for '{c}' not in alphabet")
        self.complement: Optional[dict[str, str]] = complement
        
        # Store ignore chars as a set
        self.ignore_chars: set[str] = set(ignore_chars)
    
    def validate_sequence(self, seq: str) -> None:
        """Validate that a sequence contains only alphabet characters (plus ignore chars)."""
        valid_chars = set(self.chars) | self.ignore_chars
        invalid = set(seq) - valid_chars
        if invalid:
            raise ValueError(
                f"Sequence contains invalid characters: {invalid}. "
                f"Valid alphabet: {''.join(self.chars)}"
            )
    
    def get_mutations(self, char: str) -> list[str]:
        """Get list of characters that a given character can mutate to."""
        if char not in self.mutation_map:
            raise ValueError(f"Character '{char}' not in alphabet")
        return self.mutation_map[char]
    
    def get_complement(self, char: str) -> str:
        """Get complement of a character."""
        if self.complement is None:
            raise ValueError("This alphabet has no complement mapping defined")
        if char not in self.complement:
            raise ValueError(f"Character '{char}' not in alphabet")
        return self.complement[char]
    
    def get_seq_length(self, seq: str) -> int:
        """Get the number of valid alphabet characters in a sequence.
        
        Counts only characters in the alphabet, ignoring any ignore_chars.
        Useful for determining the effective length of a gapped alignment.
        """
        char_set = set(self.chars)
        return sum(1 for c in seq if c in char_set)
    
    def get_valid_seq_positions(self, seq: str) -> list[int]:
        """Get the indices of valid alphabet characters in a sequence.
        
        Returns positions of characters that are in the alphabet,
        skipping any ignore_chars (gaps, spaces, etc.).
        Useful for determining which positions are eligible for mutagenesis.
        """
        char_set = set(self.chars)
        return [i for i, c in enumerate(seq) if c in char_set]
    
    def __repr__(self) -> str:
        return f"Alphabet({''.join(self.chars)})"


@beartype
def get_alphabet(name: str) -> Alphabet:
    """Get a named alphabet.
    
    Available alphabets:
        - 'dna': ACGT with Watson-Crick complement (A<->T, G<->C)
        - 'rna': ACGU with Watson-Crick complement (A<->U, G<->C)
        - 'protein': 20 amino acids, no complement
        - 'binary': 01, complement 0<->1
    """
    name_lower = name.lower()
    if name_lower not in NAMED_ALPHABETS:
        valid = list(NAMED_ALPHABETS.keys())
        raise ValueError(f"Unknown alphabet '{name}'. Valid options: {valid}")
    
    chars = NAMED_ALPHABETS[name_lower]
    
    # Set complement based on alphabet type
    if name_lower == 'dna':
        complement = DNA_COMPLEMENT
    elif name_lower == 'rna':
        complement = RNA_COMPLEMENT
    elif name_lower == 'binary':
        complement = BINARY_COMPLEMENT
    else:
        complement = None
    
    return Alphabet(chars=chars, complement=complement)
