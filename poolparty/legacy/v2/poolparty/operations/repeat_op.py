from ..types import Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool


@beartype
class RepeatOp(Operation):
    """Repeat a sequence n times (fixed mode - no internal variability)."""
    design_card_keys = ['seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        parent_pool: Pool,
        n: int,
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        if not n > 0:
            raise ValueError(f'{n=} is not positive.')
        self.n = n
        seq_length = parent_pool.seq_length * n if parent_pool.seq_length is not None else None
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode='fixed',
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
        """Repeat input sequence n times and return dict with seq."""
        assert len(input_strings)==1
        seq = input_strings[0] * self.n
        return {'seq': seq}

#########################################################
# Public factory function
#########################################################

@beartype
def repeat(
    parent: Pool,
    n: int,
    name: str = 'repeat',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Repeat a pool's sequence n times."""    
    return Pool(operation=RepeatOp(parent_pool=parent, n=n, name=name, design_card_keys=design_card_keys))
