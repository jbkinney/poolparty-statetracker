"""ReverseComplement operation - reverse complement a sequence."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..pool import Pool
from ..marker_ops.parsing import reverse_complement_with_markers


@beartype
def reverse_complement(
    pool: Union[Pool_type, str],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool containing the reverse complement of sequences from the input pool.

    Preserves XML marker tags, repositioning them based on reversed content
    coordinates.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to reverse complement.
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
        A Pool containing reverse-complemented sequences.
    """
    from ..party import get_active_party
    from .fixed import fixed_operation

    alphabet = get_active_party().alphabet

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        return reverse_complement_with_markers(seq, alphabet.get_complement)

    return fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=lambda pools: pools[0].seq_length,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='reverse_complement',
    )
