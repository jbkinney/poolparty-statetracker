"""rc operation - reverse complement a sequence."""

from numbers import Real

from ..pool import Pool
from ..types import Optional, Pool_type, RegionType, Union, beartype
from ..utils import dna_utils


@beartype
def rc(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    style: Optional[str] = None,
) -> Pool:
    """
    Create a Pool containing the reverse complement of sequences from the input pool.

    Note: Region tags are not preserved in the output. If you need to preserve
    regions, use extract_region with rc=True instead.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to reverse complement.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_tags : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    style : Optional[str], default=None
        Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

    Returns
    -------
    Pool
        A Pool containing reverse-complemented sequences.
    """
    from .fixed import fixed_operation

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        from ..utils.parsing_utils import strip_all_tags

        seq = seqs[0]
        # Strip tags before reverse complementing
        clean_seq = strip_all_tags(seq)
        return dna_utils.reverse_complement(clean_seq)

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: lengths[0],
        region=region,
        remove_tags=remove_tags,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name="rc",
    )

    # Apply style if specified
    if style is not None:
        from .stylize import stylize

        result_pool = stylize(result_pool, style=style)

    return result_pool
