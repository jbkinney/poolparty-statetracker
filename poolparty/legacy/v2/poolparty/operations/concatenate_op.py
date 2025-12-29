from ..types import Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
from .from_seqs_op import from_seqs
import pandas as pd


@beartype
class ConcatenateOp(Operation):
    """Concatenate multiple sequences (fixed mode - no internal variability)."""
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        parents: Sequence[Union[Pool, str]],
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ) -> None:
        
        # Cast parents to pools as needed
        parent_pools = [(p if isinstance(p, Pool) else from_seqs([p],design_card_keys=[])) for p in parents ]
        
        # Compute seq_length
        parent_lengths = [p.seq_length for p in parent_pools]
        seq_length = None if any([l is None for l in parent_lengths]) else sum(parent_lengths)

        # Initialize base class attributes
        super().__init__(
            parent_pools=parent_pools,
            num_states=1,
            mode='fixed',
            seq_length=seq_length,
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results(
        self, 
        input_strings_lists: Sequence[Sequence[str]], 
        sequential_states: Sequence[int],
    ) -> None:
        seqs = [''.join(input_strings) for input_strings in zip(*input_strings_lists)]
        self._results_df = pd.DataFrame({'seq': seqs})

#########################################################
# Public factory function
#########################################################

@beartype
def concatenate(
    sequences: Sequence[Union[str,Pool]],
    design_card_keys: Optional[Sequence[str]] = None,
    name: str = 'concatenate',
) -> Pool:
    """Concatenate multiple sequences (Pool or str)"""
    return Pool(operation=ConcatenateOp(sequences, design_card_keys=design_card_keys, name=name))
