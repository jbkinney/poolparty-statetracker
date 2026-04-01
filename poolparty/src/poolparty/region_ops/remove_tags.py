"""Remove region tags and optionally their content from sequences."""

from numbers import Real

from ..pool import Pool
from ..types import Optional, Pool_type, RegionType, Union, beartype
from ..utils.parsing_utils import validate_single_region


@beartype
def remove_tags(
    pool: Union[Pool_type, str],
    region_name: str,
    keep_content: bool = True,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> Pool:
    """
    Remove region tags from sequences.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the region.
    region_name : str
        Name of the region to remove.
    keep_content : bool, default=True
        If True, keep the content inside the region (just remove tags).
        If False, remove both the region tags and their content.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        A Pool yielding sequences with the region tags removed.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
    ...
    ...     # Keep content (just remove tags)
    ...     result = pp.remove_tags(bg, 'region', keep_content=True)
    ...     # Result: 'ACGTTTAAGCGC'
    ...
    ...     # Remove content too
    ...     result = pp.remove_tags(bg, 'region', keep_content=False)
    ...     # Result: 'ACGTGCGC'
    """
    from ..fixed_ops.fixed import fixed_operation
    from ..fixed_ops.from_seq import from_seq
    from ..party import get_active_party

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Look up registered region length for seq_length computation
    party = get_active_party()
    region_seq_length = None
    if party is not None and party.has_region(region_name):
        region_seq_length = party.get_region(region_name).seq_length

    if keep_content:
        # Stripping tags doesn't change biological length
        def _seq_length_fn(lengths):
            return lengths[0]
    else:
        # Biological length shrinks by the region's content length
        def _seq_length_fn(lengths):
            parent_len = lengths[0]
            if parent_len is None or region_seq_length is None:
                return None
            return parent_len - region_seq_length

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        region = validate_single_region(seq, region_name)

        prefix = seq[: region.start]
        suffix = seq[region.end :]

        if keep_content:
            return prefix + region.content + suffix
        else:
            return prefix + suffix

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=_seq_length_fn,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name="remove_tags",
    )

    # The region is removed, so remove it from the pool's region set
    result_pool._untrack_region(region_name)

    return result_pool
