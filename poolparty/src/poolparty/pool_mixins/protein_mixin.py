"""Protein-specific operation mixins for ProteinPool class."""

from ..types import Literal, Optional, RegionType, Union


class ProteinMixin:
    """Mixin providing protein-specific operation methods for ProteinPool."""

    def reverse_translate(
        self,
        region: RegionType = None,
        *,
        codon_selection: Literal["first", "random"] = "first",
        num_states: Optional[int] = None,
        genetic_code: Union[str, dict] = "standard",
        iter_order: Optional[float] = None,
        prefix: Optional[str] = None,
    ):
        """Reverse translate protein sequence to DNA.

        Parameters
        ----------
        region : RegionType, default=None
            Region to reverse translate. If None, reverse translates the entire sequence.
        codon_selection : Literal["first", "random"], default="first"
            How to select codons for each amino acid:
            - "first": Use the most frequent codon (deterministic)
            - "random": Randomly select from synonymous codons (stochastic)
        num_states : Optional[int], default=None
            Number of states to generate. Only relevant when codon_selection="random".
        genetic_code : Union[str, dict], default="standard"
            Genetic code to use for reverse translation.
        iter_order : Optional[float], default=None
            Iteration order priority.
        prefix : Optional[str], default=None
            Prefix for sequence names.

        Returns
        -------
        DnaPool
            Pool containing reverse-translated DNA sequences.
        """
        from ..orf_ops.reverse_translate import reverse_translate

        return reverse_translate(
            self,
            region=region,
            codon_selection=codon_selection,
            num_states=num_states,
            genetic_code=genetic_code,
            iter_order=iter_order,
            prefix=prefix,
        )
