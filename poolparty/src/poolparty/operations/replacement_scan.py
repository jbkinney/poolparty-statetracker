"""ReplacementScan - replace a segment of background with insert sequences."""
from ..types import Union, ModeType, Optional, Integral, Real, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


@beartype
def replacement_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    spacer_str: str = '',
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """Replace a segment of background with insert at scanning positions."""
    from .from_seq import from_seq
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan
    
    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool
    
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

