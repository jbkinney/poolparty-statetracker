"""Centralized type definitions for poolparty."""

from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any, Literal, Optional, TypeAlias, Union

import numpy as np
from beartype import beartype

# Forward reference type aliases (resolve circular imports)
Pool_type: TypeAlias = Union[
    "poolparty.pool.Pool", "poolparty.dna_pool.DnaPool", "poolparty.protein_pool.ProteinPool"
]
Operation_type: TypeAlias = "poolparty.operation.Operation"
State_type: TypeAlias = "statetracker.state.State"

# Mode type for operations
ModeType: TypeAlias = Literal["random", "sequential", "fixed"]

# Alphabet type alias
AlphabetType: TypeAlias = str | Sequence[str]

# Filter function type
FilterFunc: TypeAlias = Callable[[str, list[str] | None], bool]

# Positions type for scan operations
PositionsType: TypeAlias = Sequence[Integral] | slice | None

# Positions type for multiscan operations (also allows per-insert list-of-lists)
MultiPositionsType: TypeAlias = Sequence[Integral] | Sequence[Sequence[Integral]] | slice | None

# Region type for operations that can target a subsequence
# str = marker name, Sequence[Integral] = [start, stop] interval, None = full sequence
RegionType: TypeAlias = str | Sequence[Integral] | None

# Style assignment mode for recombination
StyleByForRecombineType: TypeAlias = Literal["source", "order"]

# Design cards type for opt-in card reporting
# None = no cards, list[str] = card keys with default names, dict[str,str] = key->column_name mapping
CardsType: TypeAlias = None | list[str] | dict[str, str]

# Inline styling types for per-sequence style tracking
# StyleTuple: (style_spec, positions) where style_spec is parsed like highlighter.py
StyleTuple: TypeAlias = tuple[str, np.ndarray]
StyleList: TypeAlias = list[StyleTuple]

# Import SeqStyle and Seq classes for convenience
from .utils.dna_seq import DnaSeq
from .utils.protein_seq import VALID_PROTEIN_CHARS, ProteinSeq
from .utils.seq import NullSeq, Seq, is_null_seq
from .utils.style_utils import SeqStyle

__all__ = [
    "beartype",
    "Union",
    "Optional",
    "Sequence",
    "Callable",
    "Literal",
    "Integral",
    "Real",
    "Any",
    "Pool_type",
    "Operation_type",
    "State_type",
    "ModeType",
    "AlphabetType",
    "FilterFunc",
    "PositionsType",
    "MultiPositionsType",
    "RegionType",
    "StyleByForRecombineType",
    "CardsType",
    "StyleTuple",
    "StyleList",
    "SeqStyle",
    "Seq",
    "DnaSeq",
    "NullSeq",
    "is_null_seq",
    "ProteinSeq",
    "VALID_PROTEIN_CHARS",
]
