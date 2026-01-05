"""Join operation - join multiple sequences together."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, beartype


@beartype
def join(
    segment_pools: Sequence[Union[Pool_type, str]],
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
    name : Optional[str], default=None
        Name to assign to the resulting Pool.
    op_name : Optional[str], default=None
        Name to assign to the internal Operation.
    iter_order : Real, default=0
        Iteration priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration priority for the internal Operation (typically unused).

    Returns
    -------
    Pool_type
        A Pool whose states yield joined sequences from the specified inputs.
    """
    from .fixed import fixed_operation

    def seq_length_from_pools_fn(pools: Sequence[Pool_type]) -> Optional[int]:
        lengths = [p.seq_length for p in pools]
        if all(L is not None for L in lengths):
            n_spacers = max(0, len(pools) - 1)
            return sum(lengths) + len(spacer_str) * n_spacers
        return None

    return fixed_operation(
        parents=segment_pools,
        seq_from_seqs_fn=lambda seqs: spacer_str.join(seqs),
        seq_length_from_pools_fn=seq_length_from_pools_fn,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='join',
    )
