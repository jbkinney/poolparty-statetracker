"""Insertion scan operation - insert a sequence at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..party import get_active_party
from ..pool import Pool


@beartype
def insertion_scan(
    pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    replace: bool = False,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_insert: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = 'insertion_scan',
) -> Pool:
    """
    Insert or replace a sequence at specified scanning positions.

    Parameters
    ----------
    pool : Pool or str
        The background Pool or sequence string.
    ins_pool : Pool or str
        The insert Pool or sequence string to be inserted.
    positions : PositionsType, default=None
        Positions for insertion/replacement (0-based). If None, all valid positions.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
    replace : bool, default=False
        If False, insert at position (output length = bg + ins).
        If True, replace content at position (output length = bg).
    style : Optional[str], default=None
        Style to apply to inserted content.
    prefix : Optional[str], default=None
        Prefix for cartesian product index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
    prefix_position : Optional[str], default=None
        Prefix for position index (e.g., 'pos_' produces 'pos_0', 'pos_1', ...).
    prefix_insert : Optional[str], default=None
        Prefix for insert index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with the insert placed at selected position(s).
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.swapcase import swapcase
    from ..region_ops import region_scan, replace_region

    # Convert string inputs to pools
    pool = from_seq(pool, _factory_name=f'{_factory_name}(from_seq)') if isinstance(pool, str) else pool
    ins_pool = from_seq(ins_pool, _factory_name=f'{_factory_name}(from_seq)') if isinstance(ins_pool, str) else ins_pool

    # Validate ins_pool has defined seq_length
    ins_length = ins_pool.seq_length
    if ins_length is None:
        raise ValueError("ins_pool must have a defined seq_length")

    # Validate bg_pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Capture site pool state
    # (Pool.state is valid even when wrapped with stylize/swapcase, unlike operation.state)
    original_ins_pool_state = ins_pool.state
    original_ins_pool_num_states = ins_pool.num_states

    # Determine marker configuration based on replace mode
    # replace=False: marker_length=0 (insert without removing background)
    # replace=True: marker_length=ins_length (replace background content)
    # Use different marker names to avoid conflicts when both are used in same Party
    marker_name = '_rep' if replace else '_ins'
    marker_length = ins_length if replace else 0

    # Check if any naming prefix is provided
    has_naming = any([prefix, prefix_position, prefix_insert])

    # 1. Insert tags at scanning positions
    # Don't pass prefix to region_scan - naming is handled by replace_region
    marked = region_scan(
        pool,
        region=marker_name,
        region_length=marker_length,
        positions=positions,
        region_constraint=region,
        remove_tags=False,  # Keep outer tags for now
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _factory_name=f'{_factory_name}(region_scan)',
    )
    marked = marked.named(f'{marked.name}:{_factory_name}(intermediate)')

    # If naming is enabled, block naming on both parent operations
    # and capture state references for composite naming in replace_region
    pos_state = None
    site_state = None
    num_sites = None
    if has_naming:
        # Block naming on marker_scan operation
        marked.operation._block_seq_names = True
        # Block naming on original ins_pool operation
        ins_pool.operation._block_seq_names = True
        # Capture state references for naming
        # Use operation.state to get the actual position state, not the product state
        pos_state = marked.operation.state
        site_state = original_ins_pool_state
        num_sites = original_ins_pool_num_states

    # 2. Replace marker with content
    return replace_region(
        marked,
        ins_pool,
        marker_name,
        iter_order=iter_order,
        _factory_name=f'{_factory_name}(replace_region)',
        _seq_name_prefix=prefix,
        _seq_name_pos_prefix=prefix_position,
        _seq_name_site_prefix=prefix_insert,
        _pos_state=pos_state,
        _site_state=site_state,
        _num_sites=num_sites,
        _style=style,
    )


@beartype
def replacement_scan(
    pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_insert: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = 'replacement_scan',
) -> Pool:
    """Replace a segment with insert at specified scanning positions.

    Equivalent to insertion_scan(..., replace=True).
    """
    return insertion_scan(
        pool=pool,
        ins_pool=ins_pool,
        positions=positions,
        region=region,
        replace=True,
        style=style,
        prefix=prefix,
        prefix_position=prefix_position,
        prefix_insert=prefix_insert,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _factory_name=_factory_name if _factory_name is not None else 'replacement_scan',
    )
