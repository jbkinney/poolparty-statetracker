"""Lower operation - convert sequence characters to lowercase."""
from numbers import Real
from ..types import Pool_type, Union, Optional, beartype
from ..pool import Pool
from ..marker_ops.parsing import transform_nonmarker_chars


@beartype
def lower(
    pool: Union[Pool_type, str],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool containing lowercase sequences from the input pool.

    Preserves XML marker tags exactly as they appear (only transforms
    non-marker characters).

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to convert to lowercase.
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
        A Pool containing lowercase sequences.
    """
    from .fixed import fixed_operation

    return fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=lambda seqs: transform_nonmarker_chars(seqs[0], str.lower),
        seq_length_from_pools_fn=lambda pools: pools[0].seq_length,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='lower',
    )
