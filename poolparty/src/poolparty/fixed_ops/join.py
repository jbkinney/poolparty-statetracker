"""Join operation - join multiple sequences together."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, beartype


@beartype
def join(
    segment_pools: Sequence[Union[Pool_type, str]],
    spacer_str: str = '',
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
    _factory_name: Optional[str] = None,
) -> Pool_type:
    """
    Concatenate multiple Pools or string sequences into a single Pool.

    Parameters
    ----------
    segment_pools : Sequence[Union[Pool_type, str]]
        List of Pool objects and/or strings to be joined in order.
        Any provided string is automatically converted to a constant Pool.
    spacer_str : str, default=''
        String to insert between joined sequences.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to the resulting concatenated sequences (e.g., 'red', 'blue bold').
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation
    Returns
    -------
    Pool_type
        A Pool whose states yield joined sequences from the specified inputs.
    """
    from .fixed import fixed_operation

    def seq_length_from_pool_lengths_fn(lengths: Sequence[Optional[int]]) -> Optional[int]:
        if all(L is not None for L in lengths):
            n_spacers = max(0, len(lengths) - 1)
            return sum(lengths) + len(spacer_str) * n_spacers
        return None

    result_pool = fixed_operation(
        parent_pools=segment_pools,
        seq_from_seqs_fn=lambda seqs: spacer_str.join(seqs),
        seq_length_from_pool_lengths_fn=seq_length_from_pool_lengths_fn,
        iter_order=iter_order,
        _factory_name=_factory_name if _factory_name is not None else 'join',
    )
    
    # Apply style if specified
    if style is not None:
        from .style import stylize
        result_pool = stylize(result_pool, style=style)
    
    return result_pool
