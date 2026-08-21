"""Stats mixin for DnaPool - summarises the library a pool produces."""

from typing import Any

from ..types import Integral, Optional


class StatsMixin:
    """Mixin providing the ``stats`` readout for DnaPool."""

    def stats(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        seed: Optional[Integral] = None,
        max_hamming_seqs: Optional[Integral] = 2000,
        max_homopolymer_run: Optional[Integral] = 6,
        enzymes: Optional[list[str]] = None,
        sites: Optional[list[str]] = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Summarise the library this pool produces.

        See :func:`poolparty.stats` for the full parameter list and the keys of
        the returned dict.

        Examples
        --------
        >>> print(pool.stats())
        >>> pool.stats(num_seqs=100_000)["frac_duplicate_seqs"]
        """
        from ..stats import stats

        return stats(
            self,
            num_seqs=num_seqs,
            num_cycles=num_cycles,
            seed=seed,
            max_hamming_seqs=max_hamming_seqs,
            max_homopolymer_run=max_homopolymer_run,
            enzymes=enzymes,
            sites=sites,
            show_progress=show_progress,
        )
