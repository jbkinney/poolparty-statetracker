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
    remove_marker: Optional[bool] = None,
    replace: bool = False,
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    seq_name_pos_prefix: Optional[str] = None,
    seq_name_site_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    replace : bool, default=False
        If False, insert at position (output length = bg + ins).
        If True, replace content at position (output length = bg).
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the inserted content.
    seq_name_prefix : Optional[str], default=None
        Prefix for cartesian product index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
    seq_name_pos_prefix : Optional[str], default=None
        Prefix for position index (e.g., 'pos_' produces 'pos_0', 'pos_1', ...).
    seq_name_site_prefix : Optional[str], default=None
        Prefix for site index (e.g., 'site_' produces 'site_0', 'site_1', ...).
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'.

    Returns
    -------
    Pool
        A Pool yielding sequences with the insert placed at selected position(s).
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.swapcase import swapcase
    from ..marker_ops import marker_scan, replace_marker_content

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

    # Resolve defaults from party
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    # Capture site operation reference BEFORE swapcase transformation
    # (swapcase creates a new fixed operation, we need the original operation with site states)
    original_ins_pool_op = ins_pool.operation
    original_ins_pool_num_states = ins_pool.num_states

    # Apply swapcase to insert if mark_changes
    if mark_changes:
        ins_pool = swapcase(ins_pool, _factory_name=f'{_factory_name}(swapcase)')

    # Determine marker configuration based on replace mode
    # replace=False: marker_length=0 (insert without removing background)
    # replace=True: marker_length=ins_length (replace background content)
    # Use different marker names to avoid conflicts when both are used in same Party
    marker_name = '_rep' if replace else '_ins'
    marker_length = ins_length if replace else 0

    # Check if any naming prefix is provided
    has_naming = any([seq_name_prefix, seq_name_pos_prefix, seq_name_site_prefix])

    # 1. Insert marker at scanning positions
    # Don't pass seq_name_prefix to marker_scan - naming is handled by replace_marker_content
    marked = marker_scan(
        pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=positions,
        region=region,
        remove_marker=False,  # Keep outer region marker for now
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
        _factory_name=f'{_factory_name}(marker_scan)',
    )
    marked = marked.named(f'{marked.name}:{_factory_name}(intermediate)')

    # If naming is enabled, block naming on both parent operations
    # and capture references for composite naming in replace_marker_content
    pos_op = None
    site_op = None
    num_sites = None
    if has_naming:
        # Block naming on marker_scan operation
        marked.operation._block_seq_names = True
        # Block naming on original ins_pool operation (before swapcase)
        original_ins_pool_op._block_seq_names = True
        # Also block on swapcase operation if it was applied
        if mark_changes:
            ins_pool.operation._block_seq_names = True
        # Capture operation references for naming
        # Use original ins_pool operation (has site states), not swapcase operation
        pos_op = marked.operation
        site_op = original_ins_pool_op
        num_sites = original_ins_pool_num_states

    # 2. Replace marker with content (spacer_str is handled by replace_marker_content)
    return replace_marker_content(
        marked,
        ins_pool,
        marker_name,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name=f'{_factory_name}(replace_marker_content)',
        _seq_name_prefix=seq_name_prefix,
        _seq_name_pos_prefix=seq_name_pos_prefix,
        _seq_name_site_prefix=seq_name_site_prefix,
        _pos_op=pos_op,
        _site_op=site_op,
        _num_sites=num_sites,
    )


@beartype
def replacement_scan(
    pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    seq_name_pos_prefix: Optional[str] = None,
    seq_name_site_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
        remove_marker=remove_marker,
        replace=True,
        mark_changes=mark_changes,
        seq_name_prefix=seq_name_prefix,
        seq_name_pos_prefix=seq_name_pos_prefix,
        seq_name_site_prefix=seq_name_site_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name=_factory_name if _factory_name is not None else 'replacement_scan',
    )
