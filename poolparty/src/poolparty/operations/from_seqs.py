"""FromSeqs operation - create a pool from a list of sequences."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class FromSeqsOp(Operation):
    """Create a pool from a list of sequences."""
    factory_name = "from_seqs"
    design_card_keys = ['seq_name', 'seq_index']
    
    @beartype
    def __init__(
        self,
        seqs: Sequence[str],
        seq_names: Optional[Sequence[str]] = None,
        mode: ModeType = 'sequential',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Real = 0,
    ) -> None:
        """Initialize FromSeqsOp."""
        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")
        if mode == 'fixed' and len(seqs) != 1:
            raise ValueError("mode='fixed' requires exactly 1 sequence")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        self.seqs = list(seqs)
        self.seq_names = list(seq_names) if seq_names else [f"seq_{i}" for i in range(len(seqs))]
        if len(self.seq_names) != len(self.seqs):
            raise ValueError("seq_names must have same length as seqs")
        match mode:
            case 'sequential':
                num_states = len(seqs)
            case 'hybrid':
                num_states = num_hybrid_states
            case _:
                num_states = 1
        seq_length = len(self.seqs[0]) if all(len(s) == len(self.seqs[0]) for s in self.seqs) else None
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
        )
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with sequence selection."""
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            idx = rng.integers(0, len(self.seqs))
        else:
            # Use state 0 when inactive (state is None)
            state = self.counter.state
            idx = (0 if state is None else state) % len(self.seqs)
        return {
            'seq_name': self.seq_names[idx],
            'seq_index': idx,
        }
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the sequence based on design card."""
        idx = card['seq_index']
        return {'seq_0': self.seqs[idx]}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'seqs': self.seqs,
            'seq_names': self.seq_names,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }


@beartype
def from_seqs(
    seqs: Sequence[str],
    seq_names: Optional[Sequence[str]] = None,
    mode: ModeType = 'sequential',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool_type:
    """
    Construct a Pool containing sequences provided in `seqs`.

    Parameters
    ----------
    seqs : Sequence[str]
        List of sequences to be included in the pool.
    seq_names : Optional[Sequence[str]], default=None
        Optional names for each sequence provided.
    mode : ModeType, default='sequential'
        Mode for sequence selection ('sequential', 'random', or 'hybrid').
    num_hybrid_states : Optional[int], default=None
        Number of states for 'hybrid' mode; ignored otherwise.
    name : Optional[str], default=None
        Name for the created Pool instance.
    op_name : Optional[str], default=None
        Name for the internal operation.
    iter_order : Real, default=0
        Pool's iteration order (priority in design).
    op_iter_order : Real, default=0
        Operation's internal iteration order.

    Returns
    -------
    Pool_type
        A Pool object containing the given sequences and configuration.
    """
    op = FromSeqsOp(seqs, seq_names=seq_names, mode=mode, 
                    num_hybrid_states=num_hybrid_states, name=op_name,
                    iter_order=op_iter_order)
    pool = Pool(operation=op, output_index=0, name=name, iter_order=iter_order)
    return pool
