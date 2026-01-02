"""ReplacementScan - replace a segment of background with insert sequences."""
from ..types import Union, ModeType, Optional, Integral, Real, PositionsType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def replacement_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Replace a segment of the background sequence with an insert at specified scanning positions.

    Parameters
    ----------
    bg_pool : Pool or str
        Background Pool or sequence string in which the replacement will occur.
    ins_pool : Pool or str
        Insert Pool or sequence string to replace the segment in the background.
    positions : PositionsType, default=None
        Positions at which to place the start of the replacement (0-based, inclusive). If None, all valid positions are considered.
    spacer_str : str, default=''
        String to insert as a spacer between segments when joining (optional).
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the insert sequence. If None, uses party default.
    mode : ModeType, default='random'
        Selection mode for replacement positions: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states to use when mode is 'hybrid' (ignored for other modes).
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
        A Pool yielding sequences where a segment of the background is replaced by the insert sequence
        at the specified scanning positions, using the defined selection mode.
    """
    from .from_seq import from_seq
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan
    from .swap_case import swap_case
    
    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool
    
    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False
    
    # Apply swap_case to insert pool if mark_changes is True
    if mark_changes:
        ins_pool = swap_case(ins_pool)
    
    # Validate that both pools have defined seq_length
    bg_length = bg_pool.seq_length
    ins_length = ins_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")
    if ins_length is None:
        raise ValueError("ins_pool must have a defined seq_length")
    
    # Validate positions (valid range: 0 to bg_length - ins_length)
    max_position = bg_length - ins_length
    validated_positions = validate_positions(positions, max_position, min_position=0)
    
    # Split background at breakpoint positions
    left, right = breakpoint_scan(
        pool=bg_pool,
        num_breakpoints=1,
        positions=validated_positions,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iter_order,
        op_name=op_name,
    )
    
    # Clip the right segment by removing the first ins_length characters
    right_clipped = seq_slice(right, slice(ins_length, None, None))
    
    # Join left, insert, and right_clipped
    result_pool = join([left, ins_pool, right_clipped], 
                       spacer_str=spacer_str, name=name, op_name=op_name,
                       iter_order=iter_order)
    return result_pool

