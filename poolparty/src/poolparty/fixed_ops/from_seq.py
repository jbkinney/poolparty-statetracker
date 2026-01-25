"""FromSeq operation - create a pool from a single sequence."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from .. import dna


@beartype
def from_seq(
    seq: str,
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
) -> Pool_type:
    """
    Create a Pool containing a single, fixed sequence.

    If pool and region are provided, the sequence replaces the region content
    in pool. Otherwise, creates a standalone pool with the sequence.

    Parameters
    ----------
    seq : str
        The sequence to include in the pool (or to insert at region).
    pool : Optional[Union[Pool, str]], default=None
        Pool or sequence. If provided with region, seq replaces the region.
    region : RegionType, default=None
        Region to replace in pool. Can be marker name (str) or [start, stop].
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    style : Optional[str], default=None
        Style to apply to the sequence.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequence (or bg_pool with region replaced).
    """
    from ..party import get_active_party
    from ..marker_ops.parsing import _validate_markers
    from .fixed import fixed_operation
    
    party = get_active_party()
    if party is None:
        raise RuntimeError(
            "from_seq requires an active Party context. "
            "Use 'with pp.Party() as party:' to create one."
        )
    
    # Validate and register any markers in the sequence
    markers = _validate_markers(seq)
    seq_length = dna.get_length_without_markers(seq)
    
    # If bg_pool and region provided, replace region content with seq
    if (pool is not None) and (region is None):
        raise ValueError("region is required when pool is provided")
    
    # When replacing region content (pool + region provided), don't pass through
    # styles from the original region to the new content
    is_replacement = pool is not None and region is not None
    
    # Create the pool
    result_pool = fixed_operation(
        parent_pools=[pool] if pool is not None else [],
        seq_from_seqs_fn=lambda _: seq,
        seq_length_from_pool_lengths_fn=lambda _: seq_length,
        region=region,
        remove_marker=remove_marker,
        iter_order=iter_order,
        _factory_name=_factory_name if _factory_name is not None else 'from_seq',
        _pass_through_styles=not is_replacement,
    )
    
    # Add validated markers to the pool
    for marker in markers:
        result_pool.add_marker(marker)
    
    # Apply style if specified
    if style is not None:
        from .style import stylize
        result_pool = stylize(result_pool, style=style)
    
    return result_pool