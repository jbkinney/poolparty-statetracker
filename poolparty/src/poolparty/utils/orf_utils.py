"""Utility functions for ORF handling."""
from numbers import Integral
from ..types import Optional, Sequence, beartype


@beartype
def validate_orf_extent(
    orf_extent: Optional[Sequence[Integral]],
    seq_length: int,
) -> tuple[int, int, int]:
    """Validate and parse orf_extent parameter.
    
    Args:
        orf_extent: ORF boundaries as (start, end) or None for entire sequence.
        seq_length: Length of the parent sequence.
    
    Returns:
        Tuple of (orf_start, orf_end, num_codons)
    
    Raises:
        ValueError: If orf_extent is invalid.
    """
    if orf_extent is None:
        orf_start = 0
        orf_end = seq_length
    else:
        if len(orf_extent) != 2:
            raise ValueError(f"orf_extent must have exactly 2 elements, got {len(orf_extent)}")
        orf_start = int(orf_extent[0])
        orf_end = int(orf_extent[1])
    
    # Validate boundaries
    if orf_start < 0:
        raise ValueError(f"orf_extent start must be >= 0, got {orf_start}")
    if orf_end > seq_length:
        raise ValueError(f"orf_extent end ({orf_end}) cannot exceed sequence length ({seq_length})")
    if orf_start >= orf_end:
        raise ValueError(f"orf_extent start ({orf_start}) must be < end ({orf_end})")
    
    # Validate divisible by 3
    orf_length = orf_end - orf_start
    if orf_length % 3 != 0:
        raise ValueError(
            f"ORF region length must be divisible by 3, got {orf_length} "
            f"(orf_extent=({orf_start}, {orf_end}))"
        )
    
    num_codons = orf_length // 3
    return orf_start, orf_end, num_codons
