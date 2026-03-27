"""ClearAnnotation operation - remove all markers/non-molecular chars and uppercase."""

from numbers import Real

from ..pool import Pool
from ..types import Optional, Pool_type, RegionType, Union, beartype
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import strip_all_tags
from ..utils.protein_seq import ProteinSeq

_MOLECULAR_CHARS: frozenset[str] = DnaSeq.VALID_CHARS | ProteinSeq.VALID_CHARS


@beartype
def clear_annotation(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
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
    remove_tags : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        A Pool with cleared annotations and uppercase sequences.
        Always has ``seq_length=None`` because output length depends on how many
        tags and non-molecular characters each sequence contains.
    """
    from .fixed import fixed_operation

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        # Strip all marker tags
        seq_no_markers = strip_all_tags(seq)
        # Filter to molecular chars only and uppercase
        return "".join(c.upper() for c in seq_no_markers if c in _MOLECULAR_CHARS)

    return fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Length may vary
        region=region,
        remove_tags=remove_tags,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name="clear_annotation",
    )
