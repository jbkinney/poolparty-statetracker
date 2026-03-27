"""Deletion multiscan operation - delete segments at multiple positions simultaneously."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, ModeType, MultiPositionsType, Optional, RegionType, Sequence, Union, beartype
from ..region_ops.region_multiscan import _is_per_insert_positions
from ..utils import validate_positions


@beartype
def deletion_multiscan(
    pool: Union[Pool, str],
    deletion_length: Integral,
    num_deletions: Integral,
    deletion_marker: Optional[str] = "-",
    positions: MultiPositionsType = None,
    region: RegionType = None,
    names: Optional[Sequence[str]] = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "deletion_multiscan",
) -> Pool:
    """
    Delete segments at multiple positions simultaneously.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    deletion_length : Integral
        Number of characters to delete at each position.
    num_deletions : Integral
        Number of simultaneous deletions to make.
    deletion_marker : Optional[str], default='-'
        Character to insert at each deletion site. If None, deleted segments
        are removed with no marker.
    positions : MultiPositionsType, default=None
        Valid positions for deletion starts (0-based). Can be a flat list
        (shared across all deletions) or a list of per-deletion sublists.
        If None, all valid positions are used.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
    names : Optional[Sequence[str]], default=None
        Custom names for the deletion regions. If None, auto-generated
        (``_del_0``, ``_del_1``, ...).
    min_spacing : Optional[Integral], default=None
        Minimum gap between end of one deletion and start of next.
    max_spacing : Optional[Integral], default=None
        Maximum gap between adjacent deletions. None = unbounded.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    style : Optional[str], default=None
        Style to apply to deletion marker characters (e.g., 'gray', 'red bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : CardsType, default=None
        Design card keys to include. Available keys: ``'combination_index'``,
        ``'starts'``, ``'ends'``, ``'names'``, ``'region_seqs'``.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple segments deleted simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_multiscan, replace_region

    if num_deletions < 1:
        raise ValueError(f"num_deletions must be >= 1, got {num_deletions}")

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    bg_length = pool_obj.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    if deletion_length <= 0:
        raise ValueError(f"deletion_length must be > 0, got {deletion_length}")
    if bg_length is not None and deletion_length >= bg_length:
        raise ValueError(
            f"deletion_length ({deletion_length}) must be < pool.seq_length ({bg_length})"
        )

    if bg_length is not None:
        min_required_length = num_deletions * deletion_length
        if min_required_length > bg_length:
            raise ValueError(
                f"Cannot fit {num_deletions} non-overlapping deletions of length "
                f"{deletion_length} in sequence of length {bg_length}"
            )

    del_char = deletion_marker if deletion_marker is not None else "-"

    markers = list(names) if names is not None else [f"_del_{i}" for i in range(num_deletions)]
    if len(markers) != num_deletions:
        raise ValueError(f"len(names) ({len(markers)}) must equal num_deletions ({num_deletions})")
    marker_length = int(deletion_length)

    if _is_per_insert_positions(positions) or region is not None:
        validated_positions = positions
    elif bg_length is not None:
        max_position = bg_length - deletion_length
        validated_positions = validate_positions(positions, max_position, min_position=0)
    else:
        validated_positions = positions

    marked = region_multiscan(
        pool_obj,
        tag_names=markers,
        num_insertions=int(num_deletions),
        positions=validated_positions,
        region=region,
        region_length=marker_length,
        insertion_mode="ordered",
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=f"{_factory_name}(region_multiscan)",
    )

    if deletion_marker is not None:
        marker_str = del_char * marker_length
        content = from_seq(marker_str, _factory_name=f"{_factory_name}(from_seq)")
        replacement_style = style
    else:
        content = from_seq("", _factory_name=f"{_factory_name}(from_seq)")
        replacement_style = None

    result = marked
    for region_name in markers:
        result = replace_region(
            result,
            content,
            region_name,
            iter_order=iter_order,
            _factory_name=f"{_factory_name}(replace_region)",
            _style=replacement_style,
        )

    return result
