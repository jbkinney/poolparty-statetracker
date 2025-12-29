"""Operation base class for poolparty.

Operations are the computational units that transform sequences.
Each Operation has:
- mode: 'random', 'sequential', or 'fixed'
- parent_pools: list of input Pools
- num_states: number of sequential states (1 for fixed/random)
- num_outputs: number of output sequences (1 for most, >1 for multi-output)
"""

import numpy as np

from .types import Sequence, ModeType, beartype
from .pool import Pool


class Operation:
    """Base class for all operations.
    
    Subclasses must implement:
    - compute(parent_seqs, state, rng) -> dict with 'seq_0', 'seq_1', etc.
    
    Subclasses should set:
    - design_card_keys: list of keys to include in design card output
    - num_outputs: number of output sequences (default 1)
    """
    
    # Class-level counter for unique operation IDs
    _id_counter: int = 0
    
    # Subclasses override these
    design_card_keys: Sequence[str] = []
    num_outputs: int = 1
    max_num_sequential_states: int = 100_000
    
    @classmethod
    def validate_num_states(cls, num_states: int, mode: ModeType) -> int:
        """Validate num_states against max_num_sequential_states.
        
        Args:
            num_states: Computed number of states (must be >= 1 or -1)
            mode: Operation mode ('sequential', 'random', or 'fixed')
        
        Returns:
            num_states if within limit, or -1 if exceeds limit and mode is 'random'
        
        Raises:
            ValueError: If num_states invalid, or exceeds limit and mode is 'sequential'
        """
        if num_states < 1 and num_states != -1:
            raise ValueError(f"num_states must be >= 1 or -1, got {num_states}")
        if num_states > cls.max_num_sequential_states:
            if mode == 'sequential':
                raise ValueError(
                    f"Number of states ({num_states}) exceeds "
                    f"max_num_sequential_states ({cls.max_num_sequential_states}). "
                    f"Use mode='random' instead."
                )
            return -1
        return num_states
    
    @beartype
    def __init__(
        self,
        parent_pools: list[Pool],
        num_states: int,
        mode: ModeType,
        name: str | None = None,
    ) -> None:
        """Initialize Operation.
        
        Args:
            parent_pools: List of input Pool objects
            num_states: Number of sequential states (>= 1, or -1 if too many to enumerate)
            mode: 'random', 'sequential', or 'fixed'
            name: Optional name for this operation
        """
        self.parent_pools = list(parent_pools)
        self.num_states = num_states
        
        # Validate and set mode
        if mode not in ('random', 'sequential', 'fixed'):
            raise ValueError(f"mode must be 'random', 'sequential', or 'fixed', got {mode!r}")
        self.mode = mode
        
        # Set name
        self.name = name if name is not None else self.__class__.__name__
        
        # Assign unique ID
        self.id = Operation._id_counter
        Operation._id_counter += 1
        
        # RNG will be set by Party before execution (only for random mode)
        self.rng: np.random.Generator | None = None
    
    @beartype
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Compute output sequence(s) and design card data.
        
        Args:
            parent_seqs: Sequences from parent pools
            state: Current state index (for sequential mode)
            rng: Random number generator (for random mode)
        
        Returns:
            dict with:
            - 'seq_0', 'seq_1', ... for each output
            - Any design_card_keys defined by the subclass
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, mode={self.mode!r}, name={self.name!r})"


@beartype
def reset_op_id_counter() -> None:
    """Reset the operation ID counter to 0. Useful for testing."""
    Operation._id_counter = 0
