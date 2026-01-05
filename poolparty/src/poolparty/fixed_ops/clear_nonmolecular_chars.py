"""ClearNonmolecularChars operation - remove all non-molecular characters from sequences."""
from numbers import Real
from ..types import Pool_type, Union, Optional, beartype
from ..pool import Pool


@beartype
def clear_nonmolecular_chars(
    pool: Union[Pool_type, str],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with all non-molecular characters removed from sequences.

    This removes everything that is NOT in the alphabet's all_chars, including:
    - Ignore characters (gaps '-', dots '.', spaces ' ', etc.)
    - All marker tags (XML-style markers like <marker>...</marker>)
    - Any other characters not in the molecular alphabet

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to filter.
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
        A Pool containing only molecular alphabet characters.
    """
    from ..party import get_active_party
    from ..marker_ops.parsing import strip_all_markers
    from .fixed import fixed_operation

    alphabet = get_active_party().alphabet
    all_chars_set = set(alphabet.all_chars)

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # First strip all marker tags
        seq_no_markers = strip_all_markers(seq)
        # Then filter to only alphabet characters
        return ''.join(c for c in seq_no_markers if c in all_chars_set)

    return fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=lambda pools: None,  # Length may vary
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='clear_nonmolecular_chars',
    )
