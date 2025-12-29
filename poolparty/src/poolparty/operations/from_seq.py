"""FromSeq operation - create a pool from a single sequence."""
from ..types import Pool_type, Optional, beartype
from ..operation import Operation
from ..pool import Pool


class FromSeqOp(Operation):
    """Create a pool from a single sequence."""
    factory_name = "from_seq"
    design_card_keys = ['seq']
    
    @beartype
    def __init__(
        self,
        seq: str,
        name: Optional[str] = None,
    ) -> None:
        """Initialize FromSeqOp."""
        self.seq = seq
        super().__init__(
            parent_pools=[],
            num_states=1,
            mode='fixed',
            seq_length=len(seq),
            name=name,
        )
    
    @beartype
    def compute_design_card(self, parent_seqs: list[str], rng=None) -> dict:
        """Return design card with the sequence."""
        return {'seq': self.seq}
    
    @beartype
    def compute_seq_from_card(self, parent_seqs: list[str], card: dict) -> dict:
        """Return the sequence from the design card."""
        return {'seq_0': card['seq']}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'seq': self.seq,
            'name': None,
        }


@beartype
def from_seq(
    seq: str,
    iteration_order: int = 0,
    op_name: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> Pool_type:
    """Create a Pool from a single sequence."""
    op = FromSeqOp(seq, name=op_name)
    op.counter.iteration_order = iteration_order
    pool = Pool(operation=op, output_index=0)
    if pool_name is not None:
        pool.name = pool_name
    return pool
