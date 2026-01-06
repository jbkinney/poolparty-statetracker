"""SeqShuffle operation - shuffle characters within a sequence region."""
from numbers import Real
from ..types import Pool_type, ModeType, Optional, Union, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def shuffle_seq(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool that shuffles characters within a specified region.
    
    Parameters
    ----------
    pool : Pool_type
        Parent pool or sequence to shuffle.
    region : RegionType, default=None
        Region to shuffle. Can be a marker name (str), explicit interval [start, stop],
        or None to shuffle entire sequence.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove the marker tags from output.
        If None, uses Party default ('remove_marker').
    mark_changes : Optional[bool], default=None
        If True, swapcase() is applied to the shuffled region. If None, uses party default.
    mode : ModeType, default='random'
        Shuffle mode: 'random' or 'hybrid'. Sequential is not supported.
    num_hybrid_states : Optional[int], default=None
        Number of pool states if mode is 'hybrid'.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.
    
    Returns
    -------
    Pool
        A Pool that yields shuffled sequences.
    """
    from ..fixed_ops.from_seq import from_seq
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = SeqShuffleOp(
        parent_pool=pool_obj,
        region=region,
        remove_marker=remove_marker,
        mark_changes=mark_changes,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool


@beartype
class SeqShuffleOp(Operation):
    """Randomly shuffle characters within a region of the parent sequence."""
    factory_name = "shuffle_seq"
    design_card_keys = ['permutation']
    
    def __init__(
        self,
        parent_pool: Pool,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        mark_changes: Optional[bool] = None,
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize SeqShuffleOp."""
        from ..party import get_active_party
        
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        if mode == 'sequential':
            raise ValueError("mode='sequential' is not supported for SeqShuffleOp")
        
        # Resolve mark_changes from party defaults if not explicitly set
        party = get_active_party()
        if mark_changes is None:
            mark_changes = party.get_default('mark_changes', False) if party else False
        self.mark_changes = mark_changes
        
        # Determine num_states
        if mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
            region=region,
            remove_marker=remove_marker,
        )
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card containing the permutation for the shuffle.
        
        Note: Region handling is done by base class wrapper methods.
        parent_seqs[0] is the region content when region is specified.
        """
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
        else:
            raise RuntimeError(f"Unsupported mode {self.mode!r}")
        
        seq = parent_seqs[0]
        
        # Get molecular positions only (excludes markers and ignore_chars)
        molecular_positions = self._get_molecular_positions(seq)
        num_molecular = len(molecular_positions)
        
        if num_molecular == 0:
            permutation = tuple()
        else:
            order = rng.permutation(num_molecular)
            # Convert order (new positions holding original indices) to mapping original->new
            permutation = [0] * num_molecular
            for new_pos, orig_idx in enumerate(order):
                permutation[orig_idx] = int(new_pos)
            permutation = tuple(permutation)
        return {'permutation': permutation}
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Apply the permutation to the sequence (molecular chars only).
        
        Note: Region handling is done by base class wrapper methods.
        parent_seqs[0] is the region content when region is specified.
        """
        seq = parent_seqs[0]
        permutation = card['permutation']
        
        # Get molecular positions (excludes markers and ignore_chars)
        molecular_positions = self._get_molecular_positions(seq)
        num_molecular = len(molecular_positions)
        
        if len(permutation) != num_molecular:
            raise ValueError(
                f"Permutation length ({len(permutation)}) does not match "
                f"molecular character count ({num_molecular})"
            )
        
        if num_molecular == 0:
            return {'seq_0': seq}
        
        # Extract molecular characters
        molecular_chars = [seq[pos] for pos in molecular_positions]
        
        # Apply permutation: permutation[i] tells us where char i should go
        shuffled_molecular = [''] * num_molecular
        for i, ch in enumerate(molecular_chars):
            dest = permutation[i]
            shuffled_molecular[dest] = ch
        
        # Apply swapcase to shuffled molecular chars if mark_changes is True
        if self.mark_changes:
            shuffled_molecular = [ch.swapcase() for ch in shuffled_molecular]
        
        # Place shuffled molecular chars back at their original positions
        seq_list = list(seq)
        for i, pos in enumerate(molecular_positions):
            seq_list[pos] = shuffled_molecular[i]
        shuffled_seq = ''.join(seq_list)
        
        return {'seq_0': shuffled_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'region': self._region,
            'remove_marker': self._remove_marker,
            'mark_changes': self.mark_changes,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }

