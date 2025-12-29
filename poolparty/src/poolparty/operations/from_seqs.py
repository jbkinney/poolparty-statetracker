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
        hybrid_mode_num_states: Optional[int] = None,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize FromSeqsOp."""
        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")
        if mode == 'fixed' and len(seqs) != 1:
            raise ValueError("mode='fixed' requires exactly 1 sequence")
        if mode == 'hybrid' and hybrid_mode_num_states is None:
            raise ValueError("hybrid_mode_num_states is required when mode='hybrid'")
        self.seqs = list(seqs)
        self.seq_names = list(seq_names) if seq_names else [f"seq_{i}" for i in range(len(seqs))]
        if len(self.seq_names) != len(self.seqs):
            raise ValueError("seq_names must have same length as seqs")
        if mode == 'sequential':
            num_states = len(seqs)
        elif mode == 'hybrid':
            num_states = hybrid_mode_num_states
        else:
            num_states = 1
        # Compute seq_length if all sequences have the same length
        lengths = set(len(s) for s in self.seqs)
        seq_length = lengths.pop() if len(lengths) == 1 else None
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            op_iteration_order=op_iteration_order,
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
            'hybrid_mode_num_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'op_iteration_order': self.iteration_order,
        }


@beartype
def from_seqs(
    seqs: Sequence[str],
    seq_names: Optional[Sequence[str]] = None,
    mode: ModeType = 'sequential',
    hybrid_mode_num_states: Optional[int] = None,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> Pool_type:
    """Create a Pool from a list of sequences."""
    op = FromSeqsOp(seqs, seq_names=seq_names, mode=mode, 
                    hybrid_mode_num_states=hybrid_mode_num_states, name=op_name,
                    op_iteration_order=op_iteration_order)
    pool = Pool(operation=op, output_index=0)
    pool.iteration_order = pool_iteration_order
    if pool_name is not None:
        pool.name = pool_name
    return pool
