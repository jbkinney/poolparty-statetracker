"""Mutagenize operation - apply random mutations to a sequence."""

from __future__ import annotations
import random
from typing import List, Optional, Union, TYPE_CHECKING

from ..operation import Operation
from ..utils import validate_alphabet

if TYPE_CHECKING:
    from ..pool import Pool


class MutagenizeOp(Operation):
    """Apply random mutations to a parent sequence.
    
    This is a transformer operation - it has one parent pool.
    Each position independently has a probability of being mutated.
    Has infinite states (random sampling).
    """
    op_name = 'mutagenize'
    
    def __init__(
        self,
        parent: 'Pool',
        alphabet: list[str],
        mutation_rate: Union[float, list[float]],
        mark_changes: bool,
        is_uniform_rate: bool,
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize MutagenizeOp."""
        self.alphabet = alphabet
        self.mutation_rate = mutation_rate
        self.mark_changes = mark_changes
        self._is_uniform_rate = is_uniform_rate
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent],
            num_states=-1,  # Unknown number of states (random sampling)
            mode=mode,
            seq_length=parent.seq_length,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute mutated sequence with random mutations.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number (used as random seed)
        
        Returns:
            Mutated sequence
        """
        base_seq = input_strings[0]
        
        # Create local Random instance
        local_rng = random.Random(state)
        
        # Generate mutations
        result = []
        for i, char in enumerate(base_seq):
            rate = self.mutation_rate if self._is_uniform_rate else self.mutation_rate[i]
            
            if local_rng.random() < rate:
                available_chars = [c for c in self.alphabet if c != char]
                if available_chars:
                    mutated_char = local_rng.choice(available_chars)
                    result.append(mutated_char)
                else:
                    result.append(char)
            else:
                result.append(char)
        
        # Apply case change if requested
        if self.mark_changes:
            result = [
                result[i].swapcase() if result[i] != base_seq[i] else result[i]
                for i in range(len(result))
            ]
        
        return ''.join(result)


def mutagenize_op(
    seq: Union['Pool', str],
    alphabet: Union[str, list[str]] = 'dna',
    mutation_rate: Union[float, list[float]] = 0.1,
    mark_changes: bool = False,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Apply random mutations to a sequence.
    
    Each position in the sequence independently has a probability of being mutated
    to a different character from the alphabet. Has infinite states (random sampling).
    
    Note: This operation only supports random mode.
    
    Args:
        seq: Input sequence (string or Pool) to mutate
        alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
            or a list of single-character strings. Default: 'dna'
        mutation_rate: Probability of mutation at each position (0-1).
            Can be a single float for uniform rate, or a list of floats
            for position-specific rates. Default: 0.1
        mark_changes: If True, apply swapcase() to mutated positions. Default: False
        mode: Must be 'random'. Sequential mode is not supported.
        name: Optional name for this pool
    
    Returns:
        A Pool that generates mutated sequences.
    
    Example:
        >>> pool = mutagenize('ACGTACGT', mutation_rate=0.2)
        >>> pool.operation.num_states
        inf
        >>> seqs = pool.generate_library(num_seqs=10, seed=42)
        >>> len(seqs)
        10
    
    Raises:
        ValueError: If mode is 'sequential'
        ValueError: If mutation_rate is out of range [0, 1]
        ValueError: If mutation_rate array length doesn't match sequence length
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # Only supports random mode
    if mode == 'sequential':
        raise ValueError("mutagenize only supports mode='random'")
    
    # Validate and store alphabet
    alphabet_list = validate_alphabet(alphabet)
    
    # If seq is a string, wrap it in from_seqs first
    if isinstance(seq, str):
        parent = from_seqs_op([seq])
        seq_len = len(seq)
    else:
        parent = seq
        seq_len = parent.seq_length
    
    # Validate and normalize mutation_rate
    if isinstance(mutation_rate, (list, tuple)):
        if len(mutation_rate) != seq_len:
            raise ValueError(
                f"mutation_rate array length ({len(mutation_rate)}) "
                f"must match sequence length ({seq_len})"
            )
        for rate in mutation_rate:
            if not 0 <= rate <= 1:
                raise ValueError(f"mutation_rate values must be between 0 and 1, got {rate}")
        mutation_rate_normalized = list(mutation_rate)
        is_uniform = False
    else:
        if not 0 <= mutation_rate <= 1:
            raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
        mutation_rate_normalized = mutation_rate
        is_uniform = True
    
    return Pool(
        operation=MutagenizeOp(
            parent=parent,
            alphabet=alphabet_list,
            mutation_rate=mutation_rate_normalized,
            mark_changes=mark_changes,
            is_uniform_rate=is_uniform,
            mode=mode,
            name=name,
        ),
    )
