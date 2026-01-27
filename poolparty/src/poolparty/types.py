"""Centralized type definitions for poolparty."""
from typing import TypeAlias, Literal, Union, Optional, Any
from collections.abc import Sequence, Callable
from beartype import beartype
from beartype.typing import Tuple
from numbers import Real, Integral
import numpy as np

# Forward reference type aliases (resolve circular imports)
Pool_type: TypeAlias = "poolparty.pool.Pool"
Operation_type: TypeAlias = "poolparty.operation.Operation"
State_type: TypeAlias = "statetracker.state.State"

# Mode type for operations
ModeType: TypeAlias = Literal['random', 'sequential', 'fixed']

# Alphabet type alias
AlphabetType: TypeAlias = Union[str, Sequence[str]]

# Filter function type
FilterFunc: TypeAlias = Callable[[str, list[str] | None], bool]

# Positions type for scan operations
PositionsType: TypeAlias = Union[Sequence[Integral], slice, None]

# Region type for operations that can target a subsequence
# str = marker name, Sequence[Integral] = [start, stop] interval, None = full sequence
RegionType: TypeAlias = Union[str, Sequence[Integral], None]

# Inline styling types for per-sequence style tracking
# StyleTuple: (style_spec, positions) where style_spec is parsed like highlighter.py
StyleTuple: TypeAlias = tuple[str, np.ndarray]
StyleList: TypeAlias = list[StyleTuple]

# Import SeqStyle and Seq classes for convenience
from .utils.style_utils import SeqStyle
from .utils.seq import Seq

__all__ = [
    'beartype',
    'Union',
    'Optional', 
    'Sequence',
    'Callable',
    'Literal',
    'Integral',
    'Real',
    'Any',
    'Pool_type',
    'Operation_type',
    'State_type',
    'ModeType',
    'AlphabetType',
    'FilterFunc',
    'PositionsType',
    'RegionType',
    'StyleTuple',
    'StyleList',
    'SeqStyle',
    'Seq',
]
