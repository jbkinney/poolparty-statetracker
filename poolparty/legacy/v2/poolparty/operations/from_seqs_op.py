from ..types import Union, Optional, Literal, Sequence, Real, ModeType, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
class FromSeqsOp(Operation):
    """Create a pool from a list of sequences."""
    design_card_keys = ['seq_name', 'seq_index', 'seq']
  
    #########################################################
    # Constructor
    #########################################################
  
    def __init__(
        self,
        seqs: Sequence[str],
        seq_names: Optional[Sequence[str]] = None,
        mode: ModeType = 'random',
        name: Optional[str] = None,
        seq_probs: Optional[Sequence[Real]] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ) -> None:
        
        # Validate seqs
        if len(seqs)==0:
            raise ValueError("seqs must be non-empty")
        self.seqs = list(seqs)
        
        # Validate seq_names
        if seq_names is not None:
            if len(seq_names) != len(seqs):
                raise ValueError("seq_names must be the same length as seqs")
            self.seq_names = list(seq_names)
        else:
            self.seq_names = [f"seq_{i}" for i in range(len(seqs))]
        
        # Compute seq_length: fixed if all same length, None if variable
        seq_lengths = set(len(s) for s in seqs)
        seq_length = seq_lengths.pop() if len(seq_lengths) == 1 else None
    
        # Set seq_probs
        if seq_probs is None:
            arr = np.ones(len(seqs)) / len(seqs)
        else:
            if len(seq_probs) != len(seqs):
                raise ValueError(f"{len(seq_probs)=}) does not match {len(seqs)=}.")
            arr = np.array(seq_probs)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{seq_probs=} has non-finite values.")
            if not np.any(arr < 0):
                raise ValueError(f"{seq_probs=} has negative values.")
            if np.sum(arr) == 0:
                raise ValueError(f"seq_probs sums to zero.")
        self.seq_probs = arr / np.sum(arr)
    
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=len(seqs),
            mode=mode,
            seq_length=seq_length,
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
        """Return a dict with seq and design card data for one sequence."""
        if self.mode == 'sequential':
            index = sequential_state % len(self.seqs)
        elif self.mode == 'random':  
            index = self.rng.choice(len(self.seqs), p=self.seq_probs)
        else:
            raise ValueError(f"{self.mode=} is not 'sequential' or 'random'.")
        seq = self.seqs[index]
        return {'seq': seq, 'seq_name': self.seq_names[index], 'seq_index': index}

#########################################################
# Public factory function
#########################################################

@beartype
def from_seqs(
    strings: Sequence[str],
    string_names: Optional[Sequence[str]] = None,
    mode: ModeType = 'random',
    name: str = 'from_seqs',
    seq_probs: Optional[Sequence[Real]] = None,
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool from a list of sequences."""
    from ..pool import Pool
    return Pool(
        operation=FromSeqsOp(strings, 
                             seq_names=string_names, 
                             mode=mode, 
                             name=name, 
                             seq_probs=seq_probs,
                             design_card_keys=design_card_keys),
    )
