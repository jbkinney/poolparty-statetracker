"""rc operation - reverse complement a sequence."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from ..marker_ops.parsing import reverse_complement_with_markers
from .. import dna


@beartype
def rc(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    spacer_str: str = '',
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
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
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
    from .fixed import fixed_operation

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        return reverse_complement_with_markers(seq, dna.complement)

    return fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: lengths[0],
        region=region,
        remove_marker=remove_marker,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='rc',
    )
