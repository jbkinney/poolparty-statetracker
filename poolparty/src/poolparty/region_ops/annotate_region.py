"""Annotate a region in pool sequences, optionally applying styling."""

from numbers import Real

from ..types import Optional, Pool_type


def annotate_region(
    pool: Pool_type,
    name: str,
    extent: Optional[tuple[int, int]] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> Pool_type:
    """
    Annotate a region in the pool's sequences, optionally applying a style.

    If region 'name' already exists in the pool, extent must be None (can't change
    bounds), but styling can still be applied.

    If region doesn't exist, insert XML tags at the specified extent. If extent
    is None, the entire sequence is marked as the region.

    Parameters
    ----------
    pool : Pool
        The pool to annotate.
    name : str
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
    if pool.has_region(name):
        if extent is not None:
            raise ValueError(
                f"Region '{name}' already exists. Cannot specify extent when "
                f"annotating an existing region. To change bounds, create a new region."
            )
        result_pool = pool
    else:
        # Region doesn't exist - create it
        if extent is None:
            # Use full sequence
            start, stop = 0, pool.seq_length
        else:
            start, stop = extent

        # Insert tags to create the region
        result_pool = insert_tags(
            pool, name, start=start, stop=stop, iter_order=iter_order, prefix=prefix
        )

    # Apply styling if requested
    if style is not None:
        result_pool = stylize(result_pool, region=name, style=style, iter_order=iter_order)

    return result_pool
