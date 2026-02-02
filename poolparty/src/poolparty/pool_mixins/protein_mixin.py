"""Protein-specific operation mixins for ProteinPool class."""

from ..types import Optional, Pool_type, Real


class ProteinMixin:
    """Mixin providing protein-specific operation methods for ProteinPool.

    Currently a placeholder for future protein-specific operations like:
    - reverse_translate() - convert protein sequence to DNA
    """

    # Future: reverse_translate() will be added here
    # def reverse_translate(
    #     self,
    #     codon_table: str = "standard",
    #     optimization: str = "random",
    #     ...
    # ) -> "DnaPool":
    #     """Reverse translate protein sequence to DNA."""
    #     pass
    pass
