import random
from typing import Union
from .pool import Pool

# IUPAC nucleotide codes to DNA bases mapping
IUPAC_TO_DNA_DICT = {
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


class IUPACPool(Pool):
    """A class for generating DNA sequences from IUPAC notation.
    
    Supports both random generation and sequential iteration through
    all possible DNA sequences defined by the IUPAC string. The pool
    has a finite number of states equal to the product of possibilities
    at each position.
    
    For example, "RN" has 2 * 4 = 8 states (R=A/G, N=A/C/G/T).
    """
    def __init__(self, 
                 iupac_seq: str, 
                 max_num_states: int = None, 
                 mode: str = 'random', 
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize an IUPACPool.
        
        Args:
            iupac_seq: A string comprised of valid IUPAC characters
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
            metadata: Metadata level ('core', 'features', 'complete'). Default: 'features'
                
        Raises:
            ValueError: If iupac_seq is empty or contains invalid IUPAC characters
        """
        # Validate iupac_seq is not empty
        if not iupac_seq:
            raise ValueError("iupac_seq must be a non-empty string")
        
        # Validate all characters are valid IUPAC characters
        invalid_chars = set()
        for char in iupac_seq:
            if char not in IUPAC_TO_DNA_DICT:
                invalid_chars.add(char)
        
        if invalid_chars:
            invalid_list = sorted(invalid_chars)
            valid_chars = sorted(IUPAC_TO_DNA_DICT.keys())
            raise ValueError(
                f"iupac_seq contains invalid IUPAC character(s): {invalid_list}. "
                f"Valid IUPAC characters are: {valid_chars}"
            )
        
        # Store the IUPAC sequence
        self.iupac_seq = iupac_seq
        
        # Store the possible DNA bases at each position
        self.position_options = [IUPAC_TO_DNA_DICT[char] for char in iupac_seq]
        
        super().__init__(op='iupac', max_num_states=max_num_states, mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
    
    def _calculate_num_internal_states(self) -> int:
        """Calculate the number of internal states as product of possibilities at each position."""
        num_states = 1
        for options in self.position_options:
            num_states *= len(options)
        return num_states
    
    def _calculate_seq_length(self) -> int:
        """Sequence length equals the length of the IUPAC string."""
        return len(self.iupac_seq)
    
    def _compute_seq(self) -> str:
        """Compute sequence based on current state.
        
        For sequential mode, maps state directly to a specific sequence using
        mixed-radix conversion (similar to KmerPool).
        For random mode, randomly selects from possibilities at each position,
        ensuring equal coverage at each position regardless of which states are sampled.
        
        Returns:
            A DNA sequence string
        """
        if self.mode == 'sequential':
            # Sequential mode: use mixed-radix conversion
            state = self.get_state() % self.num_internal_states
            result = []
            
            # Mixed-radix conversion: each position can have different number of options
            for position_opts in reversed(self.position_options):
                result.append(position_opts[state % len(position_opts)])
                state //= len(position_opts)
            
            return ''.join(reversed(result))
        else:
            # Random mode: independently sample each position for equal coverage
            rng = random.Random(self.get_state())
            result = []
            for position_opts in self.position_options:
                result.append(rng.choice(position_opts))
            return ''.join(result)
    
    def __repr__(self) -> str:
        return f"IUPACPool(iupac_seq='{self.iupac_seq}')"

