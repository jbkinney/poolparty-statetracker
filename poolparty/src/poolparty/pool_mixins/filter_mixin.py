"""Filter operation mixins for DnaPool class.

Provides convenience methods for filtering sequences based on
various properties like GC content, complexity, and restriction sites.
"""

from typing_extensions import Self

from ..types import Optional, Pool_type, Sequence


class FilterMixin:
    """Mixin providing filter convenience methods for DnaPool.

    All filter methods wrap the base `filter()` method with appropriate
    predicate functions. Filtered sequences become NullSeq and can be
    removed from output using `generate_library(discard_null_seqs=True)`.
    """

    def filter_gc(
        self,
        min_gc: float = 0.0,
        max_gc: float = 1.0,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Filter sequences by GC content.

        Parameters
        ----------
        min_gc : float, default 0.0
            Minimum GC content (inclusive), as fraction 0.0-1.0.
        max_gc : float, default 1.0
            Maximum GC content (inclusive), as fraction 0.0-1.0.
        name : str, optional
            Name for the filter operation.
        prefix : str, optional
            Prefix for sequence naming.

        Returns
        -------
        Pool
            Filtered pool where sequences outside the GC range are NullSeq.

        Examples
        --------
        >>> pool.filter_gc(min_gc=0.4, max_gc=0.6)  # Keep 40-60% GC
        >>> pool.filter_gc(max_gc=0.5)  # Keep <= 50% GC
        """
        from ..utils.seq_properties import calc_gc

        if not 0.0 <= min_gc <= 1.0:
            raise ValueError(f"min_gc must be between 0.0 and 1.0, got {min_gc}")
        if not 0.0 <= max_gc <= 1.0:
            raise ValueError(f"max_gc must be between 0.0 and 1.0, got {max_gc}")
        if min_gc > max_gc:
            raise ValueError(f"min_gc ({min_gc}) cannot be greater than max_gc ({max_gc})")

        def predicate(seq: str) -> bool:
            gc = calc_gc(seq)
            return min_gc <= gc <= max_gc

        return self.filter(predicate, name=name, prefix=prefix)

    def filter_homopolymer(
        self,
        max_length: int,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Filter out sequences containing long homopolymer runs.

        Homopolymers (runs of repeated single bases like AAAA or GGGG)
        can cause synthesis problems and sequencing errors.

        Parameters
        ----------
        max_length : int
            Maximum allowed homopolymer length. Sequences with runs
            longer than this will be filtered out.
        name : str, optional
            Name for the filter operation.
        prefix : str, optional
            Prefix for sequence naming.

        Returns
        -------
        Pool
            Filtered pool where sequences with long homopolymers are NullSeq.

        Examples
        --------
        >>> pool.filter_homopolymer(max_length=4)  # No runs > 4 bases
        >>> pool.filter_homopolymer(max_length=6)  # Allow up to 6-base runs
        """
        from ..utils.seq_properties import has_homopolymer

        if max_length < 1:
            raise ValueError(f"max_length must be at least 1, got {max_length}")

        def predicate(seq: str) -> bool:
            return not has_homopolymer(seq, max_length)

        return self.filter(predicate, name=name, prefix=prefix)

    def filter_complexity(
        self,
        min_complexity: float,
        k_range: tuple[int, ...] = (1, 2, 3),
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Filter sequences by linguistic complexity.

        Linguistic complexity measures the ratio of observed unique k-mers
        to maximum possible, averaged across k values. Low complexity indicates
        repetitive sequences.

        Parameters
        ----------
        min_complexity : float
            Minimum complexity score (0.0-1.0). Sequences below this
            threshold will be filtered out.
        k_range : tuple[int, ...], default (1, 2, 3)
            Tuple of k values for k-mer analysis.
        name : str, optional
            Name for the filter operation.
        prefix : str, optional
            Prefix for sequence naming.

        Returns
        -------
        Pool
            Filtered pool where low-complexity sequences are NullSeq.

        Examples
        --------
        >>> pool.filter_complexity(min_complexity=0.5)
        >>> pool.filter_complexity(min_complexity=0.3, k_range=(1, 2))
        """
        from ..utils.seq_properties import calc_complexity

        if not 0.0 <= min_complexity <= 1.0:
            raise ValueError(f"min_complexity must be between 0.0 and 1.0, got {min_complexity}")

        def predicate(seq: str) -> bool:
            return calc_complexity(seq, k_range) >= min_complexity

        return self.filter(predicate, name=name, prefix=prefix)

    def filter_dust(
        self,
        max_score: float = 2.0,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Filter sequences by DUST complexity score.

        The DUST algorithm identifies low-complexity regions based on
        triplet frequencies. This is the standard algorithm used by
        NCBI BLAST for sequence masking.

        Parameters
        ----------
        max_score : float, default 2.0
            Maximum DUST score allowed. Sequences with scores above this
            (indicating low complexity) will be filtered out.
            Typical thresholds: 2.0 (stringent) to 4.0 (permissive).
        name : str, optional
            Name for the filter operation.
        prefix : str, optional
            Prefix for sequence naming.

        Returns
        -------
        Pool
            Filtered pool where high-DUST-score sequences are NullSeq.

        Examples
        --------
        >>> pool.filter_dust(max_score=2.0)  # Stringent filtering
        >>> pool.filter_dust(max_score=4.0)  # More permissive
        """
        from ..utils.seq_properties import calc_dust

        if max_score < 0:
            raise ValueError(f"max_score must be non-negative, got {max_score}")

        def predicate(seq: str) -> bool:
            return calc_dust(seq) <= max_score

        return self.filter(predicate, name=name, prefix=prefix)

    def filter_restriction_sites(
        self,
        enzymes: Sequence[str] | None = None,
        sites: Sequence[str] | None = None,
        check_rc: bool = True,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Filter out sequences containing restriction enzyme recognition sites.

        Useful for designing sequences compatible with cloning strategies
        that use specific restriction enzymes.

        Parameters
        ----------
        enzymes : Sequence[str], optional
            List of enzyme names or preset names. Enzyme names are looked up
            in the built-in restriction enzyme database. Preset names
            (e.g., "golden_gate", "common", "mcs") expand to predefined
            enzyme lists.

            Available presets:
            - "golden_gate": BsaI, BsmBI, BbsI, SapI, etc.
            - "common": EcoRI, BamHI, HindIII, XhoI, etc.
            - "mcs": Standard multiple cloning site enzymes
            - "gibson": Common enzymes to avoid in Gibson assembly
            - "frequent_cutters": 4-base cutters like DpnI, AluI
            - "rare_cutters": 8-base cutters like NotI, PacI
            - "blunt": Blunt-end cutters

        sites : Sequence[str], optional
            List of explicit recognition sequences (IUPAC format allowed).
            Use this for custom sites not in the enzyme database.
        check_rc : bool, default True
            If True, also check for reverse complement of each site.
            Important for non-palindromic enzymes like BsaI.
        name : str, optional
            Name for the filter operation.
        prefix : str, optional
            Prefix for sequence naming.

        Returns
        -------
        Pool
            Filtered pool where sequences with restriction sites are NullSeq.

        Raises
        ------
        ValueError
            If an enzyme name is not recognized and not a preset name.
            If neither enzymes nor sites is provided.

        Examples
        --------
        >>> # Filter out common cloning sites
        >>> pool.filter_restriction_sites(enzymes=["EcoRI", "BamHI", "HindIII"])

        >>> # Use a preset for Golden Gate cloning
        >>> pool.filter_restriction_sites(enzymes=["golden_gate"])

        >>> # Mix enzymes, presets, and custom sites
        >>> pool.filter_restriction_sites(
        ...     enzymes=["EcoRI", "golden_gate"],
        ...     sites=["GAATTC", "CUSTOM"]
        ... )

        >>> # Check only forward strand (rare use case)
        >>> pool.filter_restriction_sites(enzymes=["BsaI"], check_rc=False)
        """
        from ..utils.seq_properties import get_sites_for_enzymes, has_restriction_site

        if enzymes is None and sites is None:
            raise ValueError("At least one of 'enzymes' or 'sites' must be provided")

        # Convert to lists for the helper function
        enzyme_list = list(enzymes) if enzymes else None
        site_list = list(sites) if sites else None

        # Get all recognition sites (resolves enzyme names and presets)
        all_sites = get_sites_for_enzymes(enzymes=enzyme_list, sites=site_list)

        if not all_sites:
            raise ValueError("No valid restriction sites found")

        def predicate(seq: str) -> bool:
            return not has_restriction_site(seq, all_sites, check_rc=check_rc)

        return self.filter(predicate, name=name, prefix=prefix)
