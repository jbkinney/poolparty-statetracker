from ..types import Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool

@beartype
class SliceOp(Operation):
    """Slice a sequence (fixed mode - no internal variability)."""
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor 
    #########################################################
    
    def __init__(
        self,
        parent: Pool,
        key: Union[int, slice],
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        self.key = key
        
        # Calculate sliced length
        if isinstance(key, int):
            seq_length = 1
        elif parent.seq_length is not None:
            dummy = 'x' * parent.seq_length
            seq_length = len(dummy[key])
        else:
            seq_length = None
        
        super().__init__(
            parent_pools=[parent],
            num_states=1,
            mode='fixed',
            seq_length=seq_length,
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(self, input_strings: Sequence[str], sequential_state: int) -> dict:
        """Apply slice to input sequence and return dict with seq."""
        result = input_strings[0][self.key]
        seq = result if isinstance(result, str) else str(result)
        return {'seq': seq}

#########################################################
# Public factory function
#########################################################

@beartype
def subseq(
    parent: Pool, 
    key: Union[int, slice],
    name: str = 'slice',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Slice a pool's sequence."""
    return Pool(operation=SliceOp(parent=parent, key=key, name=name, design_card_keys=design_card_keys))
