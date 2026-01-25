"""Lower operation - convert sequence characters to lowercase."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from ..marker_ops.parsing import transform_nonmarker_chars


@beartype
def lower(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
) -> Pool:
    """
    Create a Pool containing lowercase sequences from the input pool.

    Preserves XML marker tags exactly as they appear (only transforms
    non-marker characters).

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to convert to lowercase.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

    Returns
    -------
    Pool
        A Pool containing lowercase sequences.
    """
    from .fixed import fixed_operation

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=lambda seqs: transform_nonmarker_chars(seqs[0], str.lower),
        seq_length_from_pool_lengths_fn=lambda lengths: lengths[0],
        region=region,
        remove_marker=remove_marker,
        iter_order=iter_order,
        _factory_name='lower',
    )
    
    # Apply style if specified
    if style is not None:
        from .style import stylize
        result_pool = stylize(result_pool, style=style)
    
    return result_pool
