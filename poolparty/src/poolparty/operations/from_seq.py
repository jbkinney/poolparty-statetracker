"""FromSeq operation - create a pool from a single sequence."""
from numbers import Real
from ..types import Pool_type, Optional, beartype
from ..operation import Operation
from ..pool import Pool


@beartype
def from_seq(
    seq: str,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """
    Create a Pool containing a single, fixed sequence.

    Parameters
    ----------
    seq : str
        The sequence to include in the pool.
    op_name : Optional[str], default=None
        Name for the internal Operation (if None, a default is used).
    name : Optional[str], default=None
        Name for the resulting Pool (if None, a default is used).
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the internal Operation (has no real effect).
        
    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequence as its only state.
    """
    op = FromSeqOp(seq, name=op_name, iter_order=op_iter_order)
    pool = Pool(operation=op, output_index=0, iter_order=iter_order, name=name)
    return pool


@beartype
class FromSeqOp(Operation):
    """Create a pool from a single sequence."""
    factory_name = "from_seq"
    design_card_keys = ['seq']
    
    def __init__(
        self,
        seq: str,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize FromSeqOp."""
        from ..party import get_active_party
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_seq requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        self.seq = seq
        # Use length without markers (includes all chars except marker tags)
        seq_length = party._alphabet.get_length_without_markers(seq)
        super().__init__(
            parent_pools=[],
            num_states=1,
            mode='fixed',
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
        )
    
    def compute_design_card(self, parent_seqs: list[str], rng=None) -> dict:
        """Return design card with the sequence."""
        return {'seq': self.seq}
    
    def compute_seq_from_card(self, parent_seqs: list[str], card: dict) -> dict:
        """Return the sequence from the design card."""
        return {'seq_0': card['seq']}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'seq': self.seq,
            'name': None,
            'iter_order': self.iter_order,
        }
