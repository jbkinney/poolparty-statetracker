"""Annotate a region in pool sequences, optionally applying styling."""

from numbers import Real
from typing import TypeVar

from ..pool import Pool
from ..types import Optional, Pool_type

T = TypeVar("T", bound=Pool)


def annotate_region(
    pool: T,
    region_name: str,
    extent: Optional[tuple[int, int]] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> T:
    """
    Annotate a region in the pool's sequences, optionally applying a style.

    If the region already exists in the pool, extent must be None (can't change
    bounds), but styling can still be applied.

    If region doesn't exist, insert XML tags at the specified extent. If extent
    is None, the entire sequence is marked as the region.

    Parameters
    ----------
    pool : Pool
        The pool to annotate.
    region_name : str
        Name for the region.
    extent : Optional[tuple[int, int]]
        Start and stop positions (0-indexed, stop exclusive) for the region.
        If None and region doesn't exist, uses the entire sequence.
        Must be None if region already exists.
    style : Optional[str]
        Style to apply to the region (e.g., 'red', 'bold blue').
        Applied via stylize() operation.
    iter_order : Optional[Real]
        Iteration order priority for the Operation.
    prefix : Optional[str]
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        Pool with region annotated and optionally styled.

    Raises
    ------
    ValueError
        If extent is specified for an existing region.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.stylize import stylize
    from ..party import get_active_party
    from .insert_tags import insert_tags

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    party = get_active_party()
    if party is None:
        raise RuntimeError(
            "annotate_region requires an active Party context. "
            "Use 'with pp.Party() as party:' or 'pp.init()' to create one."
        )

    # Check if region already exists
    if pool.has_region(region_name):
        if extent is not None:
            raise ValueError(
                f"Region '{region_name}' already exists. Cannot specify extent when "
                f"annotating an existing region. To change bounds, create a new region."
            )
        result_pool = pool
    else:
        # Region doesn't exist - create it
        if extent is None and pool.seq_length is None:
            from ..fixed_ops.fixed import fixed_operation
            from ..utils.parsing_utils import build_region_tags

            region = party.register_region(region_name, None)

            def wrap_full_seq(seqs):
                return build_region_tags(region_name, seqs[0])

            result_pool = fixed_operation(
                parent_pools=[pool],
                seq_from_seqs_fn=wrap_full_seq,
                seq_length_from_pool_lengths_fn=lambda lengths: lengths[0] if lengths else None,
                iter_order=iter_order,
                prefix=prefix,
            )
            result_pool.add_region(region)
        else:
            if extent is None:
                start, stop = 0, pool.seq_length
            else:
                start, stop = extent

            result_pool = insert_tags(
                pool, region_name, start=start, stop=stop, iter_order=iter_order, prefix=prefix
            )

    # Apply styling if requested
    if style is not None:
        result_pool = stylize(result_pool, region=region_name, style=style, iter_order=iter_order)

    return result_pool
