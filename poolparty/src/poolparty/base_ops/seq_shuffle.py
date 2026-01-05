"""SeqShuffle operation - shuffle characters within a sequence region."""
from numbers import Real
from ..types import Pool_type, ModeType, Optional, Union, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def seq_shuffle(
    pool: Union[Pool_type, str],
    region: RegionType = None,
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
    factory_name = "seq_shuffle"
    design_card_keys = ['permutation']
    
    def __init__(
        self,
        parent_pool: Pool,
        region: RegionType = None,
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize SeqShuffleOp."""
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        if mode == 'sequential':
            raise ValueError("mode='sequential' is not supported for SeqShuffleOp")
        
        # Store and validate region parameter using centralized validation
        self._region = region
        Operation._validate_region(region)
        
        self._seq_length = parent_pool.seq_length
        # Determine num_states
        if mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
        )
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card containing the permutation for the shuffle."""
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
        else:
            raise RuntimeError(f"Unsupported mode {self.mode!r}")
        
        seq = parent_seqs[0]
        # Extract region content using centralized helper
        _, region_seq, _ = self._extract_region_parts(seq, self._region)
        region_len = len(region_seq)
        
        if region_len == 0:
            permutation = tuple()
        else:
            order = rng.permutation(region_len)
            # Convert order (new positions holding original indices) to mapping original->new
            permutation = [0] * region_len
            for new_pos, orig_idx in enumerate(order):
                permutation[orig_idx] = int(new_pos)
            permutation = tuple(permutation)
        return {'permutation': permutation}
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Apply the permutation to the target region."""
        seq = parent_seqs[0]
        permutation = card['permutation']
        
        # Extract region parts using centralized helper
        prefix, region_seq, suffix = self._extract_region_parts(seq, self._region)
        region_len = len(region_seq)
        
        if len(permutation) != region_len:
            raise ValueError(
                f"Permutation length ({len(permutation)}) does not match region length ({region_len})"
            )
        
        if region_len == 0:
            return {'seq_0': seq}
        
        # Apply permutation: permutation[i] tells us where char i should go
        shuffled_chars = [''] * region_len
        for i, ch in enumerate(region_seq):
            dest = permutation[i]
            shuffled_chars[dest] = ch
        
        # Reassemble: prefix + shuffled_region + suffix
        result_seq = prefix + ''.join(shuffled_chars) + suffix
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'region': self._region,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }

