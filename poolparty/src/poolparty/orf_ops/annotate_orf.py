"""Annotate an ORF region in pool sequences with frame and optional styling."""

from numbers import Real

from ..dna_pool import DnaPool
from ..region import VALID_FRAMES, OrfRegion
from ..types import Optional, Pool_type, beartype


@beartype
def annotate_orf(
    pool: Pool_type,
    region_name: str,
    extent: Optional[tuple[int, int]] = None,
    frame: int = 1,
    style: Optional[str] = None,
    style_codons: Optional[list[str]] = None,
    style_frames: Optional[list[str]] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> DnaPool:
    """
    Annotate an ORF region with reading frame, optionally applying styling.

    If region already exists as a plain Region, extent must be None and the region
    is upgraded to an OrfRegion with the specified frame.

    If region already exists as an OrfRegion, extent must be None and frame must
    match the existing frame (can't change frame of an immutable OrfRegion).
    Styling can still be applied.

    If region doesn't exist, insert XML tags and register as OrfRegion.

    Parameters
    ----------
    pool : Pool
        The pool to annotate.
    region_name : str
        Name for the ORF region.
    extent : Optional[tuple[int, int]]
        Start and stop positions (0-indexed, stop exclusive) for the region.
        If None and region doesn't exist, uses the entire sequence.
        Must be None if region already exists.
    frame : int
        Reading frame (+1, +2, +3, -1, -2, -3). Default +1.
    style : Optional[str]
        Flat style to apply to the region (e.g., 'red'). Applied via stylize().
        Mutually exclusive with style_codons and style_frames.
    style_codons : Optional[list[str]]
        List of styles for codon-based coloring. Applied via stylize_orf().
        Mutually exclusive with style and style_frames.
    style_frames : Optional[list[str]]
        List of styles for frame-based coloring (length must be multiple of 3).
        Applied via stylize_orf(). Mutually exclusive with style and style_codons.
    iter_order : Optional[Real]
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        Pool with ORF region annotated and optionally styled.

    Raises
    ------
    ValueError
        If frame is invalid, extent is specified for existing region,
        frame differs from existing OrfRegion's frame, or multiple
        style options are provided.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.stylize import stylize
    from ..party import get_active_party
    from ..region_ops.insert_tags import insert_tags
    from .stylize_orf import stylize_orf

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Validate frame
    if frame not in VALID_FRAMES:
        raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")

    # Validate style exclusivity
    style_count = sum(x is not None for x in [style, style_codons, style_frames])
    if style_count > 1:
        raise ValueError("At most one of style, style_codons, or style_frames can be provided")

    party = get_active_party()
    if party is None:
        raise RuntimeError(
            "annotate_orf requires an active Party context. "
            "Use 'with pp.Party() as party:' or 'pp.init()' to create one."
        )

    # Check if region already exists
    if pool.has_region(region_name):
        if extent is not None:
            raise ValueError(
                f"Region '{region_name}' already exists. Cannot specify extent when "
                f"annotating an existing region. To change bounds, create a new region."
            )

        # Get existing region from Party
        existing = party.get_region(region_name)

        if isinstance(existing, OrfRegion):
            # Already an OrfRegion - check frame matches
            if existing.frame != frame:
                raise ValueError(
                    f"OrfRegion '{region_name}' already exists with frame={existing.frame}. "
                    f"Cannot change frame to {frame} (OrfRegion is immutable)."
                )
            # Frame matches, can proceed with styling
        else:
            # Plain Region - upgrade to OrfRegion
            party.upgrade_to_orf_region(region_name, frame)

        result_pool = pool
    else:
        # Region doesn't exist - create it
        if extent is None and pool.seq_length is None:
            from ..fixed_ops.fixed import fixed_operation
            from ..utils.parsing_utils import build_region_tags

            orf_region = party.register_orf_region(region_name, None, frame)

            def wrap_full_seq(seqs):
                return build_region_tags(region_name, seqs[0])

            result_pool = fixed_operation(
                parent_pools=[pool],
                seq_from_seqs_fn=wrap_full_seq,
                seq_length_from_pool_lengths_fn=lambda lengths: None,
                iter_order=iter_order,
                prefix=prefix,
            )
            result_pool.add_region(orf_region)
        else:
            if extent is None:
                start, stop = 0, pool.seq_length
            else:
                start, stop = extent

            # Insert tags to create the region
            result_pool = insert_tags(
                pool, region_name, start=start, stop=stop, iter_order=iter_order, prefix=prefix
            )

            # Register as OrfRegion with Party
            existing = party.get_region(region_name)
            if not isinstance(existing, OrfRegion):
                party.upgrade_to_orf_region(region_name, frame)

    # Apply styling if requested
    if style is not None:
        # Flat style via stylize()
        result_pool = stylize(result_pool, region=region_name, style=style, iter_order=iter_order)
    elif style_codons is not None or style_frames is not None:
        # ORF-aware styling via stylize_orf()
        # Get the frame from the registered OrfRegion (in case we need to use it)
        orf_region = party.get_region(region_name)
        actual_frame = orf_region.frame if isinstance(orf_region, OrfRegion) else frame
        result_pool = stylize_orf(
            result_pool,
            region=region_name,
            style_codons=style_codons,
            style_frames=style_frames,
            frame=actual_frame,
            iter_order=iter_order,
        )

    return result_pool
