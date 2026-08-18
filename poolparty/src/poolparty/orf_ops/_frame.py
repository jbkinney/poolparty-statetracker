"""Shared reading-frame helpers for ORF operations.

All ORF-aware operations must derive codon geometry from :func:`frame_offset`
so that a single ``OrfRegion.frame`` identifies the same codons for every
operation. Historically each operation carried its own copy of this logic and
they disagreed for ``|frame| != 1``; see ``resolve_frame`` callers.
"""

from ..party import get_active_party
from ..region import VALID_FRAMES, OrfRegion
from ..types import Optional, RegionType


def frame_offset(frame: int) -> int:
    """Number of bases skipped before the first complete codon.

    ``frame=+N`` skips ``N-1`` bases at the region's 5' end. ``frame=-N`` skips
    ``N-1`` bases at the region's 3' end, the codons then being read 3'->5' as
    the reverse complement.

    Parameters
    ----------
    frame : int
        Reading frame and orientation: +1, +2, +3, -1, -2, -3.

    Returns
    -------
    int
        Bases skipped before the first complete codon (0, 1, or 2).
    """
    return abs(frame) - 1


def resolve_frame(region: RegionType, frame: Optional[int]) -> int:
    """Resolve the frame value, looking up from OrfRegion if needed.

    Backward compatibility: defaults to frame=1 when region is None or an interval.
    When region is a named OrfRegion, uses the stored frame.
    When region is a named plain Region, raises an error (must specify frame).
    """
    # If frame is explicitly provided, validate and use it
    if frame is not None:
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")
        return frame

    # frame is None - try to get from OrfRegion or use default
    if region is None or not isinstance(region, str):
        # Backward compatibility: default to frame=1 for non-named regions
        return 1

    # region is a string (region name) - look it up
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")

    if not party.has_region(region):
        # Region doesn't exist yet - use default frame=1
        return 1

    registered_region = party.get_region(region)
    if isinstance(registered_region, OrfRegion):
        return registered_region.frame
    else:
        raise ValueError(
            f"Region '{region}' is a plain Region, not an OrfRegion. "
            f"frame must be specified explicitly, or use annotate_orf() to "
            f"upgrade the region to an OrfRegion with a frame."
        )
