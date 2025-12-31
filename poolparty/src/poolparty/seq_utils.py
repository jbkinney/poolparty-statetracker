"""Sequence utilities for poolparty."""
from .types import PositionsType, Sequence, Integral, beartype


@beartype
def validate_positions(
    positions: PositionsType,
    max_position: int,
    min_position: int = 0,
) -> list[int]:
    """Validate and resolve positions to a list of integers.
    
    Parameters
    ----------
    positions
        None (all positions), a slice, or an explicit list of positions.
    max_position
        Maximum valid position (inclusive).
    min_position
        Minimum valid position (inclusive).
    
    Returns
    -------
    list[int]
        Validated list of positions in [min_position, max_position].
    
    Raises
    ------
    ValueError
        If any position is out of range or duplicates exist.
    """
    num_positions = max_position - min_position + 1
    
    if positions is None:
        return list(range(min_position, max_position + 1))
    
    if isinstance(positions, slice):
        start, stop, step = positions.indices(num_positions)
        return [min_position + i for i in range(start, stop, step)]
    
    result = list(positions)
    for pos in result:
        if pos < min_position or pos > max_position:
            raise ValueError(
                f"Position {pos} is out of range [{min_position}, {max_position}]"
            )
    if len(result) != len(set(result)):
        raise ValueError("Positions must not contain duplicates")
    return result
