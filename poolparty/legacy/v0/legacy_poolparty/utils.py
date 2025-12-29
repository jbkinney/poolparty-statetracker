from typing import Union, List, Iterable
import hashlib
import numpy as np
from .pool import Pool
from .shuffle_pool import ShufflePool

# Define named alphabets
named_alphabets_dict = {
    "dna": list("ACGT"),
    "rna": list("ACGU"),
    "protein": list("ACDEFGHIKLMNPQRSTVWY"),
    "protein*": list("*ACDEFGHIKLMNPQRSTVWY"),
}

codon_to_aa_dict = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "TAA": "*", "TAG": "*", "TGA": "*"
}

#  Codon order reflects prevalence in the human genome. 
aa_to_codon_dict = {
    "F": ["TTC", "TTT"],
    "L": ["CTG", "CTC", "CTT", "TTG", "TTA", "CTA"],
    "I": ["ATC", "ATT", "ATA"],
    "M": ["ATG"],
    "V": ["GTG", "GTC", "GTT", "GTA"],
    "S": ["AGC", "TCC", "TCT", "TCA", "AGT", "TCG"],
    "P": ["CCC", "CCT", "CCA", "CCG"],
    "T": ["ACC", "ACA", "ACT", "ACG"],
    "A": ["GCC", "GCT", "GCA", "GCG"],
    "Y": ["TAC", "TAT"],
    "H": ["CAC", "CAT"],
    "Q": ["CAG", "CAA"],
    "N": ["AAC", "AAT"],
    "K": ["AAG", "AAA"],
    "D": ["GAC", "GAT"],
    "E": ["GAG", "GAA"],
    "C": ["TGC", "TGT"],
    "W": ["TGG"],
    "R": ["AGA", "AGG", "CGG", "CGC", "CGA", "CGT"],
    "G": ["GGC", "GGA", "GGG", "GGT"],
    "*": ["TGA", "TAA", "TAG"]
}

# Create helper dictionaries for codon mutations
codon_to_synonymous_dict = {}
for codon, aa in codon_to_aa_dict.items():
    synonymous = [c for c in aa_to_codon_dict[aa] if c != codon]
    codon_to_synonymous_dict[codon] = synonymous

all_codons = set(codon_to_aa_dict.keys())
stop_codons = ["TAA", "TAG", "TGA"]

iupac_to_dna_dict = {
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

def validate_alphabet(alphabet: Union[str, List[str]] = 'dna') -> List[str]:
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
            f"alphabet must be either a str (alphabet name) or List[str] (list of characters), "
            f"got {type(alphabet).__name__}"
        )

def shuffle(seq: Union[Pool, str], preserve_dinucleotides: bool = False) -> Pool:
    """Create a ShufflePool from an input pool or string.
    
    Args:
        seq: Input oligonucleotide sequence, either as Pool object or string
        preserve_dinucleotides: If True, preserve dinucleotide frequencies
        
    Returns:
        ShufflePool object that generates shuffled versions of the input sequence
    """
    return ShufflePool(seq, preserve_dinucleotides=preserve_dinucleotides)

def mutate_codon(codon: str, mutation_type: str, state: int | None = None) -> Union[str, None]:
    """Mutate a codon according to the specified mutation type.
    
    Args:
        codon: A 3-nucleotide DNA codon string (must be valid codon with ACGT characters).
        mutation_type: Type of mutation to perform. Must be one of:
            - 'all_by_codon': Return any of the 63 codons different from input
            - 'missense_first_codon': Choose different AA, return its first codon
            - 'missense_random_codon': Choose different AA, randomly select one of its codons
            - 'synonymous': Choose different synonymous codon
            - 'nonsense': Mutate to a stop codon (or None if already stop)
        state: Integer state for deterministic selection. If None, randomly chosen.
    
    Returns:
        Mutated codon string, or None if mutation is not possible.
    
    Raises:
        ValueError: If codon is invalid or mutation_type is not recognized.
    """
    # Validate codon
    if not isinstance(codon, str) or len(codon) != 3:
        raise ValueError(f"codon must be a 3-character string, got '{codon}'")
    
    if not all(c in 'ACGT' for c in codon):
        raise ValueError(f"codon must contain only ACGT characters, got '{codon}'")
    
    if codon not in codon_to_aa_dict:
        raise ValueError(f"codon '{codon}' is not a valid codon")
    
    # Validate mutation_type
    valid_types = ['all_by_codon', 'missense_first_codon', 'missense_random_codon', 
                   'synonymous', 'nonsense']
    if mutation_type not in valid_types:
        raise ValueError(
            f"mutation_type must be one of {valid_types}, got '{mutation_type}'"
        )
    
    # Generate random state if not provided
    if state is None:
        state = np.random.randint(0, 10000000)
    
    # Perform mutation based on type
    if mutation_type == 'all_by_codon':
        # Get all codons except the input codon
        available_codons = [c for c in codon_to_aa_dict.keys() if c != codon]
        return available_codons[state % len(available_codons)]
    
    elif mutation_type == 'missense_first_codon':
        # Get current amino acid
        current_aa = codon_to_aa_dict[codon]
        # Get list of different amino acids
        available_aas = [aa for aa in aa_to_codon_dict.keys() if aa != current_aa]
        # Select amino acid deterministically
        selected_aa = available_aas[state % len(available_aas)]
        # Return first codon for that amino acid
        return aa_to_codon_dict[selected_aa][0]
    
    elif mutation_type == 'missense_random_codon':
        # Get current amino acid
        current_aa = codon_to_aa_dict[codon]
        # Get list of different amino acids
        available_aas = [aa for aa in aa_to_codon_dict.keys() if aa != current_aa]
        # Select amino acid using state % num_aas
        selected_aa = available_aas[state % len(available_aas)]
        # Get codons for selected amino acid
        aa_codons = aa_to_codon_dict[selected_aa]
        # Select codon using (state // num_aas) % num_codons_for_that_aa
        codon_idx = (state // len(available_aas)) % len(aa_codons)
        return aa_codons[codon_idx]
    
    elif mutation_type == 'synonymous':
        # Get synonymous codons (excluding input codon)
        synonymous = codon_to_synonymous_dict[codon]
        if len(synonymous) == 0:
            return None
        return synonymous[state % len(synonymous)]
    
    elif mutation_type == 'nonsense':
        # Check if already a stop codon
        if codon in stop_codons:
            return None
        # Return one of the three stop codons
        return stop_codons[state % 3]

