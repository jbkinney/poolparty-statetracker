"""Alphabet utilities for poolparty."""
from .types import beartype, Optional, Sequence, Union

# Import XML-style marker pattern from markers module
from .markers.parsing import (
    MARKER_PATTERN,
    get_length_without_markers as _get_length_without_markers,
    get_positions_without_markers as _get_positions_without_markers,
)

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

# Default ignore characters for sequence validation
DEFAULT_IGNORE_CHARS: tuple[str, ...] = ('-', '.', ' ')


@beartype
class Alphabet:
    """Alphabet configuration for sequence operations.
    
    Attributes:
        chars: List of canonical characters (used for random generation/drawing)
        all_chars: List of all valid characters (includes case variants if support_both_cases=True)
        size: Number of canonical characters in alphabet
        mutation_map: Dict mapping each char to list of chars it can mutate to (case-preserving)
        complement: Dict mapping each char to its complement (case-preserving, or None)
        ignore_chars: Set of characters to ignore during validation
        support_both_cases: Whether both uppercase and lowercase are supported
    """
    
    def __init__(
        self,
        chars: Sequence[str],
        complement: Optional[dict[str, str]] = None,
        ignore_chars: Sequence[str] = DEFAULT_IGNORE_CHARS,
        support_both_cases: bool = True,
    ) -> None:
        """Initialize Alphabet with custom characters.
        
        Args:
            chars: Sequence of canonical alphabet characters (typically uppercase).
            complement: Optional dict mapping each char to its complement.
            ignore_chars: Characters to ignore during sequence validation.
            support_both_cases: If True, automatically expand to support both
                uppercase and lowercase versions of alphabetic characters.
                Complement and mutation_map will preserve case.
        
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
        self.support_both_cases: bool = support_both_cases
        
        # Build all_chars: canonical chars plus case variants if support_both_cases
        if support_both_cases:
            all_chars_set = set(char_list)
            for c in char_list:
                if c.isalpha():
                    all_chars_set.add(c.lower())
                    all_chars_set.add(c.upper())
            self.all_chars: list[str] = char_list + [
                c for c in sorted(all_chars_set) if c not in char_list
            ]
        else:
            self.all_chars = char_list.copy()
        
        # Build mutation map: each char -> all other chars (case-preserving)
        self.mutation_map: dict[str, list[str]] = {
            c: [other for other in char_list if other != c]
            for c in char_list
        }
        
        # Add case variants to mutation_map if support_both_cases
        if support_both_cases:
            for c in char_list:
                if c.isalpha():
                    c_lower = c.lower()
                    c_upper = c.upper()
                    # Add lowercase variant if not already canonical
                    if c_lower not in self.mutation_map:
                        self.mutation_map[c_lower] = [
                            other.lower() if other.isalpha() else other
                            for other in char_list if other != c
                        ]
                    # Add uppercase variant if not already canonical
                    if c_upper not in self.mutation_map:
                        self.mutation_map[c_upper] = [
                            other.upper() if other.isalpha() else other
                            for other in char_list if other != c
                        ]
        
        # Store complement (validate if provided)
        if complement is not None:
            for c in char_list:
                if c not in complement:
                    raise ValueError(f"Complement mapping missing for character '{c}'")
                if complement[c] not in char_list:
                    raise ValueError(f"Complement '{complement[c]}' for '{c}' not in alphabet")
            # Create expanded complement with case variants if support_both_cases
            if support_both_cases:
                expanded_complement = dict(complement)
                for c, comp in complement.items():
                    if c.isalpha():
                        c_lower = c.lower()
                        c_upper = c.upper()
                        comp_lower = comp.lower() if comp.isalpha() else comp
                        comp_upper = comp.upper() if comp.isalpha() else comp
                        if c_lower not in expanded_complement:
                            expanded_complement[c_lower] = comp_lower
                        if c_upper not in expanded_complement:
                            expanded_complement[c_upper] = comp_upper
                self.complement: Optional[dict[str, str]] = expanded_complement
            else:
                self.complement = dict(complement)
        else:
            self.complement = None
        
        # Store ignore chars as a set
        self.ignore_chars: set[str] = set(ignore_chars)
    
    def validate_sequence(self, seq: str) -> None:
        """Validate that a sequence contains only alphabet characters (plus ignore chars)."""
        valid_chars = set(self.all_chars) | self.ignore_chars
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
        """Get complement of a character. Ignore chars return themselves."""
        if self.complement is None:
            raise ValueError("This alphabet has no complement mapping defined")
        if char in self.ignore_chars:
            return char
        if char not in self.complement:
            raise ValueError(f"Character '{char}' not in alphabet")
        return self.complement[char]
    
    def get_seq_length(self, seq: str) -> int:
        """Get the number of valid alphabet characters in a sequence.
        
        Counts only characters in the alphabet, ignoring any ignore_chars
        and marker tags.
        Useful for determining the effective length of a gapped alignment.
        """
        from .markers.parsing import strip_all_markers
        # Remove all marker tags first
        seq_no_markers = strip_all_markers(seq)
        char_set = set(self.all_chars)
        return sum(1 for c in seq_no_markers if c in char_set)
    
    def get_length_without_markers(self, seq: str) -> int:
        """Get sequence length excluding only marker tags.
        
        Counts all characters except those inside marker tags.
        Unlike get_seq_length(), this includes non-alphabet characters.
        """
        return _get_length_without_markers(seq)
    
    def get_positions_without_markers(self, seq: str) -> list[int]:
        """Get raw string positions of all characters excluding marker interiors.
        
        Returns positions of all characters that are not inside marker tags.
        Use for operations that work on the full sequence (like breakpoint_scan).
        """
        return _get_positions_without_markers(seq)
    
    def get_valid_seq_positions(self, seq: str) -> list[int]:
        """Get the indices of valid alphabet characters in a sequence.
        
        Returns positions of characters that are in the alphabet,
        skipping any ignore_chars (gaps, spaces, etc.) and positions
        that overlap with marker tags.
        Useful for determining which positions are eligible for mutagenesis.
        """
        # Get positions that are not inside marker tags
        positions_without_markers = set(_get_positions_without_markers(seq))
        
        char_set = set(self.all_chars)
        return [i for i, c in enumerate(seq) if c in char_set and i in positions_without_markers]
    
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
