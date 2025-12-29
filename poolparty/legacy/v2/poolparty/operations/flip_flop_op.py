"""Flip operation - reverse complement sequences with a specified probability."""

from ..types import Optional, Sequence, ModeType, Literal, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import reverse_complement_iupac


@beartype
class FlipFlopOp(Operation):
    """Reverse complement sequences with a specified probability.
    
    This is a transformer operation that takes sequences from a parent pool
    and optionally reverse complements them based on forward_probability.
    
    The operation has exactly 2 states:
        - State 0: Forward (unchanged)
        - State 1: Reverse complement
    
    In random mode, uses forward_probability to decide which state to use.
    In sequential mode, iterates through state 0 then state 1.
    """
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        parent: Pool,
        forward_probability: float = 0.5,
        mode: Literal['random', 'sequential'] = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize FlipOp.
        
        Args:
            parent: Parent pool to transform.
            forward_probability: Probability of keeping sequence unchanged (0.0-1.0).
                1.0 = always forward, 0.0 = always reverse complement.
                Only used in random mode.
            mode: 'random' or 'sequential'.
            name: Optional name for this operation.
            design_card_keys: Keys to include in design cards.
        """
        # Validate forward_probability
        if not 0.0 <= forward_probability <= 1.0:
            raise ValueError(
                f"forward_probability must be between 0 and 1, got {forward_probability}"
            )
        
        self.forward_probability = forward_probability
        
        super().__init__(
            parent_pools=[parent],
            num_states=2,  # State 0 = forward, State 1 = reverse complement
            mode=mode,
            seq_length=parent.seq_length,
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
        """Apply flip transformation to input sequence.
        
        Args:
            input_strings: Input sequences from parent pools.
            sequential_state: State index (0=forward, 1=reverse complement).
        
        Returns:
            Dict with 'seq' key containing the transformed sequence.
        """
        seq = input_strings[0]
        
        if self.mode == 'sequential':
            # Use sequential_state: 0 = forward, 1 = reverse complement
            if sequential_state == 1:
                seq = reverse_complement_iupac(seq)
        else:
            # Random mode: use forward_probability
            if self.rng.random() >= self.forward_probability:
                seq = reverse_complement_iupac(seq)
        
        return {'seq': seq}


#########################################################
# Public factory function
#########################################################

@beartype
def flip_flop(
    parent: Pool,
    forward_probability: float = 0.5,
    mode: Literal['random', 'sequential'] = 'random',
    name: str = 'flip',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that reverse complements sequences with a specified probability.
    
    Uses the full IUPAC DNA alphabet for complementing. Characters not in the
    IUPAC alphabet are passed through unchanged.
    
    Args:
        parent: Parent pool to transform.
        forward_probability: Probability of keeping sequence unchanged (0.0-1.0).
            1.0 = always forward, 0.0 = always reverse complement.
            Only used in random mode.
        mode: 'random' or 'sequential'.
        name: Name for the pool.
        design_card_keys: Keys to include in design cards.
    
    Returns:
        A Pool that applies the flip transformation.
    
    Example:
        >>> from poolparty.operations import from_seqs, flip
        >>> parent = from_seqs(['AACC', 'GGTT'])
        >>> pool = flip(parent, forward_probability=0.5)
        >>> # In random mode, each sequence has 50% chance of being flipped
        
        >>> pool = flip(parent, mode='sequential')
        >>> # In sequential mode: state 0 = forward, state 1 = reverse complement
    """
    return Pool(
        operation=FlipFlopOp(
            parent=parent,
            forward_probability=forward_probability,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        )
    )

