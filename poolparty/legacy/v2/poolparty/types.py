"""Centralized type definitions for poolparty.

This module provides TypeAlias definitions that work with both:
- Static type checkers (mypy, pyright)
- Runtime type checkers (beartype)

Using full module paths allows beartype to resolve types at runtime
without circular import issues.
"""
from __future__ import annotations
from typing import TypeAlias, Optional, Literal, Union
from collections.abc import Sequence, Callable
from numbers import Real
from beartype import beartype

# Forward reference type aliases (resolve circular imports)
Pool_type: TypeAlias = "poolparty.pool.Pool"
Operation_type: TypeAlias = "poolparty.operation.Operation"

# Common type aliases
ModeType: TypeAlias = Literal['random', 'sequential', 'fixed']

# Alphabet type alias
AlphabetType: TypeAlias = Union[str, Sequence[str]]

# Filter function type: (seq, filtered_seqs) -> bool
FilterFunc: TypeAlias = Callable[[str, list[str] | None], bool]