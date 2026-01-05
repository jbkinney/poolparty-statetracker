"""FromSeq operation - create a pool from a single sequence."""
from numbers import Real
from ..types import Pool_type, Optional, beartype


@beartype
def from_seq(
    seq: str,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """
    Create a Pool containing a single, fixed sequence.

    Parameters
    ----------
    seq : str
        The sequence to include in the pool.
    op_name : Optional[str], default=None
        Name for the internal Operation (if None, a default is used).
    name : Optional[str], default=None
        Name for the resulting Pool (if None, a default is used).
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the internal Operation (has no real effect).

    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequence as its only state.
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
    
    seq_length = party._alphabet.get_length_without_markers(seq)
    
    pool = fixed_operation(
        parents=[],
        seq_from_seqs_fn=lambda _: seq,
        seq_length_from_pools_fn=lambda _: seq_length,
        name=name,
        op_name='from_seq',
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
    
    # Add validated markers to the pool
    for marker in markers:
        pool.add_marker(marker)
    
    return pool