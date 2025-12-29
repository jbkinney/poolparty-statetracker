"""FromSeqs operation - create a pool from a list of sequences."""

import numpy as np

from ..types import Sequence, ModeType, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party


class FromSeqsOp(Operation):
    """Create a pool from a list of sequences.
    
    In sequential mode, iterates through sequences in order.
    In random mode, samples sequences according to probabilities.
    """
    
    design_card_keys = ['seq_name', 'seq_index']
    
    @beartype
    def __init__(
        self,
        seqs: Sequence[str],
        names: Sequence[str] | None = None,
        probs: Sequence[float] | None = None,
        mode: ModeType = 'sequential',
        name: str = 'from_seqs',
    ) -> None:
        """Initialize FromSeqsOp.
        
        Args:
            seqs: List of sequences
            names: Optional names for each sequence
            probs: Sampling probabilities (for random mode)
            mode: 'sequential' or 'random'
            name: Operation name
        """
        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")
        
        self.seqs = list(seqs)
        self.names = list(names) if names else [f"seq_{i}" for i in range(len(seqs))]
        
        if len(self.names) != len(self.seqs):
            raise ValueError("names must have same length as seqs")
        
        # Normalize probabilities
        if probs is not None:
            p = np.array(probs, dtype=float)
            if len(p) != len(seqs):
                raise ValueError("probs must have same length as seqs")
            if np.any(p < 0):
                raise ValueError("probs must be non-negative")
            self.probs = p / p.sum()
        else:
            self.probs = None
        
        super().__init__(
            parent_pools=[],
            num_states=len(seqs),
            mode=mode,
            name=name,
        )
        
        # Register with active party
        party = get_active_party()
        if party is not None:
            party._register_operation(self)
    
    @beartype
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Return a sequence and its metadata."""
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError("Random mode requires RNG")
            if self.probs is not None:
                idx = int(rng.choice(len(self.seqs), p=self.probs))
            else:
                idx = int(rng.integers(0, len(self.seqs)))
        else:  # sequential
            idx = state % len(self.seqs)
        
        return {
            'seq_0': self.seqs[idx],
            'seq_name': self.names[idx],
            'seq_index': idx,
        }


@beartype
def from_seqs(
    seqs: Sequence[str],
    names: Sequence[str] | None = None,
    probs: Sequence[float] | None = None,
    mode: ModeType = 'sequential',
    name: str = 'from_seqs',
) -> Pool:
    """Create a Pool from a list of sequences.
    
    Args:
        seqs: List of sequences
        names: Optional names for each sequence
        probs: Sampling probabilities (for random mode)
        mode: 'sequential' or 'random'
        name: Operation name
    
    Returns:
        Pool containing the sequences
    
    Example:
        >>> pool = from_seqs(['AAA', 'TTT', 'GGG'])
    """
    op = FromSeqsOp(seqs, names=names, probs=probs, mode=mode, name=name)
    return Pool(operation=op, output_index=0)
