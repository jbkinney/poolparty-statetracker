"""Multiscan operations - operations that use marker_multiscan() to operate on multiple positions."""

from .deletion_multiscan import deletion_multiscan
from .insertion_multiscan import insertion_multiscan, replacement_multiscan

__all__ = [
    "deletion_multiscan",
    "insertion_multiscan",
    "replacement_multiscan",
]
