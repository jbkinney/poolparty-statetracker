"""ClearIgnoreChars operation - remove ignore characters from sequences."""
from numbers import Real
from ..types import Pool_type, Union, Optional, beartype
from ..pool import Pool


@beartype
def clear_ignore_chars(
    pool: Union[Pool_type, str],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with ignore characters removed from sequences.

    This removes only the alphabet's ignore_chars (gaps '-', dots '.', spaces ' ', etc.)
    while preserving marker tags intact.

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
        A Pool with ignore characters removed but markers preserved.
    """
    from ..party import get_active_party
    from .fixed import fixed_operation

    alphabet = get_active_party().alphabet
    ignore_chars_set = alphabet.ignore_chars

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # Remove only ignore characters, preserving markers
        return ''.join(c for c in seq if c not in ignore_chars_set)

    return fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=lambda pools: None,  # Length may vary
        name=name,
        op_name='clear_ignore_chars',
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
