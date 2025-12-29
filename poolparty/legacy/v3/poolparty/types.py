"""Centralized type definitions for poolparty.

This module provides type aliases and re-exports common typing constructs.
All other modules should import typing constructs from here rather than
directly from typing or collections.abc.
"""
from typing import TypeAlias, Literal, Union, Optional
from collections.abc import Sequence, Callable

# Re-export beartype for convenient imports across the package
from beartype import beartype

# Mode type for operations
ModeType: TypeAlias = Literal['random', 'sequential', 'fixed']

# Alphabet type alias - either a named alphabet string or sequence of characters
AlphabetType: TypeAlias = Union[str, Sequence[str]]

# Filter function type: (seq, filtered_seqs) -> bool
FilterFunc: TypeAlias = Callable[[str, list[str] | None], bool]

# Export all type aliases and typing constructs
__all__ = [
    # Beartype
    'beartype',
    # Typing constructs
    'Union',
    'Optional', 
    'Sequence',
    'Callable',
    # Custom type aliases
    'ModeType',
    'AlphabetType',
    'FilterFunc',
]
