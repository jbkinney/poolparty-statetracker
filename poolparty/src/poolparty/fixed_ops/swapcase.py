"""SwapCase operation - swap case of sequence characters."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from ..marker_ops.parsing import transform_nonmarker_chars


@beartype
def swapcase(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
    _factory_name: Optional[str] = None,
) -> Pool:
    """
    Create a Pool containing case-swapped sequences from the input pool.

    Preserves XML marker tags exactly as they appear (only transforms
    non-marker characters).

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to swap case.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to the resulting sequences (e.g., 'red', 'blue bold').
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation
    Returns
    -------
    Pool
        A Pool containing case-swapped sequences.
    """
    from .fixed import fixed_operation

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=lambda seqs: transform_nonmarker_chars(seqs[0], str.swapcase),
        seq_length_from_pool_lengths_fn=lambda lengths: lengths[0],
        region=region,
        remove_marker=remove_marker,
        iter_order=iter_order,
        _factory_name=_factory_name if _factory_name is not None else 'swapcase',
    )
    
    # Apply style if specified
    if style is not None:
        from .style import stylize
        result_pool = stylize(result_pool, style=style)
    
    return result_pool
