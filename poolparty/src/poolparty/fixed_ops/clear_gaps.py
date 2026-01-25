"""ClearGaps operation - remove all gap/non-molecular characters from sequences."""
from numbers import Real
from ..types import Pool_type, Union, Optional, RegionType, beartype
from ..pool import Pool
from ..marker_ops.parsing import TAG_PATTERN


@beartype
def clear_gaps(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool with all gap/non-molecular characters removed from sequences.

    This removes everything that is NOT in the alphabet's all_chars, including:
    - Ignore characters (gaps '-', dots '.', spaces ' ', etc.)
    - Any other characters not in the molecular alphabet

    Marker tags are preserved intact.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to filter.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool containing only molecular alphabet characters (markers preserved).
    """
    from ..party import get_active_party
    from .fixed import fixed_operation

    alphabet = get_active_party().alphabet
    all_chars_set = set(alphabet.all_chars)

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # Remove non-molecular chars while preserving marker tags
        result = []
        last_end = 0
        for match in TAG_PATTERN.finditer(seq):
            # Filter non-marker text to only molecular chars
            result.append(''.join(c for c in seq[last_end:match.start()] if c in all_chars_set))
            # Keep marker tag unchanged
            result.append(match.group(0))
            last_end = match.end()
        # Handle remaining text after last marker
        result.append(''.join(c for c in seq[last_end:] if c in all_chars_set))
        return ''.join(result)

    return fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Length may vary
        region=region,
        remove_marker=remove_marker,
        iter_order=iter_order,
        _factory_name='clear_gaps',
    )
