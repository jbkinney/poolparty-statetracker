"""ClearGaps operation - remove all gap/non-molecular characters from sequences."""

from numbers import Real

from ..pool import Pool
from ..types import Optional, Pool_type, RegionType, Union, beartype
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import TAG_PATTERN
from ..utils.protein_seq import ProteinSeq

_MOLECULAR_CHARS: frozenset[str] = DnaSeq.VALID_CHARS | ProteinSeq.VALID_CHARS


@beartype
def clear_gaps(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> Pool:
    """
    Create a Pool with all gap/non-molecular characters removed from sequences.

    This removes everything that is NOT a valid molecular character (DNA or
    protein), including gaps '-', dots '.', spaces ' ', and any other
    non-molecular characters.

    Marker tags are preserved intact.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to filter.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_tags : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        A Pool containing only molecular alphabet characters (markers preserved).
        Always has ``seq_length=None`` because output length depends on how many
        non-molecular characters each sequence contains.
    """
    from .fixed import fixed_operation

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # Remove non-molecular chars while preserving marker tags
        result = []
        last_end = 0
        for match in TAG_PATTERN.finditer(seq):
            # Filter non-marker text to only molecular chars
            result.append("".join(c for c in seq[last_end : match.start()] if c in _MOLECULAR_CHARS))
            # Keep marker tag unchanged
            result.append(match.group(0))
            last_end = match.end()
        # Handle remaining text after last marker
        result.append("".join(c for c in seq[last_end:] if c in _MOLECULAR_CHARS))
        return "".join(result)

    return fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Length may vary
        region=region,
        remove_tags=remove_tags,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name="clear_gaps",
    )
