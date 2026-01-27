"""SeqShuffle operation - shuffle characters within a sequence region."""
from numbers import Real
from ..types import Pool_type, ModeType, Optional, Union, RegionType, beartype, Seq
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def shuffle_seq(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    _remove_tags: bool = False,
    style: Optional[str] = None,
    _factory_name: Optional[str] = None,
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
        Shuffle mode: 'random'. Sequential is not supported.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    _remove_tags : bool, default=False
        If True and region is a marker name, remove the marker tags from output.
    style : Optional[str], default=None
        Style to apply to shuffled characters (e.g., 'purple', 'red bold').
    
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
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        _remove_tags=_remove_tags,
        style=style,
        _factory_name=_factory_name,
    )
    result_pool = Pool(operation=op)
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
        spacer_str: str = '',
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _remove_tags: bool = False,
        style: Optional[str] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize SeqShuffleOp."""
        from ..party import get_active_party
        
        if mode == 'sequential':
            raise ValueError("mode='sequential' is not supported for SeqShuffleOp")
        
        # Set factory_name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
        
        # Store styling parameter
        self._style = style
        
        # Determine num_states
        if mode == 'random':
            # num_states stays None for pure random mode
            pass
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            remove_tags=_remove_tags,
        )
    
    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
        suppress_styles: bool = False,
    ) -> tuple[Seq, dict]:
        """Return shuffled Seq and design card.
        
        Note: Region handling is done by base class compute() method.
        parents[0] is the region content when region is specified.
        """
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
        else:
            raise RuntimeError(f"Unsupported mode {self.mode!r}")
        
        seq = parents[0].string
        
        # Get molecular positions only (excludes markers and ignore_chars)
        molecular_positions = self._get_molecular_positions(seq)
        num_molecular = len(molecular_positions)
        
        if num_molecular == 0:
            permutation = tuple()
            shuffled_seq = seq
        else:
            order = rng.permutation(num_molecular)
            # Convert order (new positions holding original indices) to mapping original->new
            permutation = [0] * num_molecular
            for new_pos, orig_idx in enumerate(order):
                permutation[orig_idx] = int(new_pos)
            permutation = tuple(permutation)
            
            # Extract molecular characters
            molecular_chars = [seq[pos] for pos in molecular_positions]
            
            # Apply permutation: permutation[i] tells us where char i should go
            shuffled_molecular = [''] * num_molecular
            for i, ch in enumerate(molecular_chars):
                dest = permutation[i]
                shuffled_molecular[dest] = ch
            
        # Place shuffled molecular chars back at their original positions
        seq_list = list(seq)
        for i, pos in enumerate(molecular_positions):
            seq_list[pos] = shuffled_molecular[i]
        shuffled_seq = ''.join(seq_list)
        
        # Pass through parent styles and add styling to shuffled characters if requested
        from ..utils.style_utils import SeqStyle
        if suppress_styles:
            output_style = SeqStyle.empty(len(shuffled_seq))
        else:
            output_style = parents[0].style
            if self._style and molecular_positions:
                output_style = output_style.add_style(
                    self._style, 
                    np.array(molecular_positions, dtype=np.int64)
                )
        
        output_seq = Seq(shuffled_seq, output_style)
        
        return output_seq, {
            'permutation': permutation,
        }
    
    

