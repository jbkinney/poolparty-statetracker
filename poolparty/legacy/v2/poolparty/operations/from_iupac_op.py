"""FromIupac operation - generate DNA sequences from IUPAC notation."""

from ..types import Optional, Sequence, ModeType, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


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


@beartype
class FromIupacOp(Operation):
    """Generate DNA sequences from IUPAC notation.
    
    This is a generator operation with no parent pools.
    Produces DNA sequences defined by the IUPAC string.
    
    The number of states equals the product of possibilities at each position.
    For example, "RN" has 2 * 4 = 8 states (R=A/G, N=A/C/G/T).
    
    In sequential mode, iterates through all possible combinations.
    In random mode, uniformly samples from allowed bases at each position.
    """
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        iupac_seq: str,
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize FromIupacOp.
        
        Args:
            iupac_seq: The IUPAC sequence string.
            mode: Either 'random' or 'sequential'.
            name: Optional name for this operation.
            design_card_keys: Keys to include in design cards.
        """
        # Validate iupac_seq is not empty
        if not iupac_seq:
            raise ValueError("iupac_seq must be a non-empty string")
        
        # Validate all characters and build position options
        iupac_seq = iupac_seq.upper()
        invalid_chars = set()
        position_options = []
        for char in iupac_seq:
            if char not in IUPAC_TO_DNA:
                invalid_chars.add(char)
            else:
                position_options.append(IUPAC_TO_DNA[char])
        
        if invalid_chars:
            raise ValueError(
                f"iupac_seq contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                f"Valid IUPAC characters are: {sorted(IUPAC_TO_DNA.keys())}"
            )
        
        self.iupac_seq = iupac_seq
        self.position_options = position_options
        
        # Compute num_states as product of possibilities at each position
        num_states = 1
        for options in position_options:
            num_states *= len(options)
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=len(iupac_seq),
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int
    ) -> dict:
        """Generate a DNA sequence and return dict.
        
        In sequential mode, uses mixed-radix conversion to enumerate all
        possible sequences. In random mode, uniformly samples from allowed
        bases at each position.
        """
        if self.mode == 'sequential':
            # Mixed-radix conversion: map state to specific sequence
            state = sequential_state % self.num_states
            result = []
            remaining = state
            for position_opts in reversed(self.position_options):
                result.append(position_opts[remaining % len(position_opts)])
                remaining //= len(position_opts)
            seq = ''.join(reversed(result))
        else:
            # Random mode: uniformly sample from allowed bases at each position
            seq = ''.join(
                self.rng.choice(options) 
                for options in self.position_options
            )
        
        return {'seq': seq}


#########################################################
# Public factory function
#########################################################

@beartype
def from_iupac(
    iupac_seq: str,
    mode: ModeType = 'random',
    name: str = 'from_iupac',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that generates DNA sequences from IUPAC notation.
    
    Supports both random generation and sequential iteration through
    all possible DNA sequences defined by the IUPAC string.
    
    Args:
        iupac_seq: A string of valid IUPAC characters.
            Valid characters: A, C, G, T, U, R, Y, S, W, K, M, B, D, H, V, N
        mode: Either 'random' or 'sequential' (default: 'random').
        name: Name for the pool.
        design_card_keys: Keys to include in design cards.
    
    Returns:
        A Pool that generates DNA sequences from the IUPAC pattern.
    
    Example:
        >>> pool = from_iupac('RN', mode='sequential')
        >>> pool.operation.num_states
        8  # R=2 options × N=4 options
        >>> seqs = pool.generate_library(num_seqs=8)['seq'].tolist()
        >>> set(seqs)
        {'AA', 'AC', 'AG', 'AT', 'GA', 'GC', 'GG', 'GT'}
    
    IUPAC Codes:
        A = Adenine
        C = Cytosine
        G = Guanine
        T = Thymine
        U = Uracil (treated as T)
        R = A or G (purine)
        Y = C or T (pyrimidine)
        S = G or C
        W = A or T
        K = G or T
        M = A or C
        B = C, G, or T (not A)
        D = A, G, or T (not C)
        H = A, C, or T (not G)
        V = A, C, or G (not T)
        N = A, C, G, or T (any base)
    """
    return Pool(
        operation=FromIupacOp(
            iupac_seq=iupac_seq,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        ),
    )

