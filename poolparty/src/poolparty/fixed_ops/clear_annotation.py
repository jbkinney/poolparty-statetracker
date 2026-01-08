"""ClearAnnotation operation - remove all markers/non-molecular chars and uppercase."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from ..marker_ops.parsing import strip_all_markers


@beartype
def clear_annotation(
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
    Create a Pool with all annotations cleared and sequences uppercased.

    Removes all XML marker tags and non-molecular characters, then uppercases
    the result. When a region is specified, only transforms content within
    that region (nested markers and non-molecular chars inside are cleared).

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to transform.
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
        A Pool with cleared annotations and uppercase sequences.
    """
    from ..party import get_active_party
    from .fixed import fixed_operation

    alphabet = get_active_party().alphabet
    all_chars_set = set(alphabet.all_chars)

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # Strip all marker tags
        seq_no_markers = strip_all_markers(seq)
        # Filter to molecular chars only and uppercase
        return ''.join(c.upper() for c in seq_no_markers if c in all_chars_set)

    return fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Length may vary
        region=region,
        remove_marker=remove_marker,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='clear_annotation',
    )
