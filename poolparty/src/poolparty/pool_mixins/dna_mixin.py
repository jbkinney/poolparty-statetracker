"""DNA-specific operation mixins for DnaPool class."""

from typing import TYPE_CHECKING

from typing_extensions import Self

import pandas as pd

if TYPE_CHECKING:
    from ..protein_pool import ProteinPool

from ..types import (
    CardsType,
    Integral,
    Literal,
    ModeType,
    Optional,
    Pool_type,
    Real,
    RegionType,
    Sequence,
    Union,
)


class DnaMixin:
    """Mixin providing DNA-specific operation methods for DnaPool."""

    # =========================================================================
    # DNA-specific base operations
    # =========================================================================

    def insert_from_iupac(
        self,
        iupac_seq: str,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> Self:
        """Insert IUPAC-generated DNA sequences into a region.

        Parameters
        ----------
        iupac_seq : str
            IUPAC sequence string (e.g., 'RN' for purine + any base).
            Valid characters: A, C, G, T, U, R, Y, S, W, K, M, B, D, H, V, N.
        region : RegionType, default=None
            Region to replace. Can be a marker name or [start, stop] interval.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Sequence selection mode: 'sequential' or 'random'.
        num_states : Optional[int], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        style : Optional[str], default=None
            Style to apply to generated sequences (e.g., 'red', 'blue bold').
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'iupac_state'``.

        Returns
        -------
        Pool
            A Pool yielding DNA sequences from the IUPAC pattern.
        """
        from ..base_ops.from_iupac import from_iupac

        return from_iupac(
            iupac_seq=iupac_seq,
            pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
            cards=cards,
        )

    def insert_from_motif(
        self,
        prob_df: pd.DataFrame,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> Self:
        """Insert sequences sampled from a position probability matrix into a region.

        Parameters
        ----------
        prob_df : pd.DataFrame
            DataFrame with probability values for each position.
            Columns should be alphabet characters (e.g., 'A', 'C', 'G', 'T').
            Rows represent positions. Values are probabilities (auto-normalized).
        region : RegionType, default=None
            Region to replace. Can be a marker name or [start, stop] interval.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Sequence selection mode. Only 'random' is supported.
        num_states : Optional[int], default=None
            Number of states. If None, defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        style : Optional[str], default=None
            Style to apply to generated sequences (e.g., 'red', 'blue bold').
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'prob_state'``.

        Returns
        -------
        Pool
            A Pool yielding sequences sampled from the probability matrix.
        """
        from ..base_ops.from_motif import from_motif

        return from_motif(
            prob_df=prob_df,
            pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
            cards=cards,
        )

    def insert_kmers(
        self,
        length: Integral,
        region: RegionType = None,
        style: Optional[str] = None,
        case: Literal["lower", "upper"] = "upper",
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Insert DNA k-mers (all possible sequences of length k) into a region.

        Parameters
        ----------
        length : int
            Length of k-mers to generate.
        region : RegionType, default=None
            Region to replace. Can be a marker name or [start, stop] interval.
        style : Optional[str], default=None
            Style to apply to generated k-mers (e.g., 'red', 'blue bold').
        case : Literal['lower', 'upper'], default='upper'
            Case of output k-mers.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Sequence selection mode: 'sequential' or 'random'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'kmer_index'``, ``'kmer'``.

        Returns
        -------
        Pool
            A Pool whose states yield DNA k-mers of the specified length.
        """
        from ..base_ops.get_kmers import get_kmers

        return get_kmers(
            length=length,
            pool=self,
            region=region,
            style=style,
            case=case,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            cards=cards,
        )

    def flip(
        self,
        region: RegionType = None,
        rc_prob: Real = 0.5,
        prefix: Optional[str] = None,
        mode: ModeType = "sequential",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> Self:
        """Randomly reverse-complement sequences based on a probability.

        Parameters
        ----------
        region : RegionType, default=None
            Region to flip. Can be marker name (str), [start, stop], or None.
        rc_prob : Real, default=0.5
            Probability of reverse-complementing each sequence.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='sequential'
            Selection mode: 'sequential' (enumerate forward/RC) or 'random'.
        num_states : Optional[int], default=None
            Number of states. In sequential mode defaults to 2 (forward + RC).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        style : Optional[str], default=None
            Style to apply to reverse-complemented sequences (e.g., 'red', 'blue bold').
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'flip'``.

        Returns
        -------
        Pool
            A Pool that yields forward or reverse-complemented sequences.
        """
        from ..base_ops.flip import flip

        return flip(
            pool=self,
            region=region,
            rc_prob=rc_prob,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
            cards=cards,
        )

    # =========================================================================
    # DNA-specific fixed operations
    # =========================================================================

    def rc(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Self:
        """Reverse-complement sequences deterministically.

        Parameters
        ----------
        region : RegionType, default=None
            Region to reverse-complement. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        style : Optional[str], default=None
            Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

        Returns
        -------
        Pool
            A Pool containing reverse-complemented sequences.
        """
        from ..fixed_ops.rc import rc

        return rc(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
        )

    # =========================================================================
    # ORF operations (DNA-specific)
    # =========================================================================

    def annotate_orf(
        self,
        region_name: str,
        extent: Optional[tuple[int, int]] = None,
        frame: int = 1,
        style: Optional[str] = None,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Annotate an ORF region with reading frame, optionally applying styling.

        Parameters
        ----------
        region_name : str
            Name for the ORF region.
        extent : Optional[tuple[int, int]], default=None
            Start and stop positions (0-indexed, stop exclusive) for the region.
            If None and region doesn't exist, uses the entire sequence.
            Must be None if region already exists.
        frame : int, default=1
            Reading frame (+1, +2, +3, -1, -2, -3).
        style : Optional[str], default=None
            Style to apply to the region (e.g., 'red', 'bold blue').
            Mutually exclusive with style_codons and style_frames.
        style_codons : Optional[list[str]], default=None
            List of styles for codon-based coloring. Applied via stylize_orf().
            Mutually exclusive with style and style_frames.
        style_frames : Optional[list[str]], default=None
            List of styles for frame-based coloring (length must be multiple
            of 3). Mutually exclusive with style and style_codons.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            Pool with ORF region annotated and optionally styled.
        """
        from ..orf_ops.annotate_orf import annotate_orf

        return annotate_orf(
            self,
            region_name,
            extent=extent,
            frame=frame,
            style=style,
            style_codons=style_codons,
            style_frames=style_frames,
            iter_order=iter_order,
            prefix=prefix,
        )

    def stylize_orf(
        self,
        region: RegionType = None,
        *,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        frame: Optional[int] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Apply ORF-aware inline styling to sequences.

        Parameters
        ----------
        region : RegionType, default=None
            Region to style. Can be a marker name or [start, stop] interval.
            If None, styles the entire sequence.
        style_codons : Optional[list[str]], default=None
            List of styles to apply to codons in sequence, cycling through
            the list. Mutually exclusive with style_frames.
        style_frames : Optional[list[str]], default=None
            List of styles with length a multiple of 3. Each group of 3
            styles is applied to frames 0, 1, 2 of a codon, cycling through
            groups. Mutually exclusive with style_codons.
        frame : Optional[int], default=None
            Reading frame: +1, +2, +3, -1, -2, -3.
            If None and region is a named OrfRegion, uses the OrfRegion's frame.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool with ORF-aware inline styling attached to sequences.
        """
        from ..orf_ops.stylize_orf import stylize_orf

        return stylize_orf(
            pool=self,
            region=region,
            style_codons=style_codons,
            style_frames=style_frames,
            frame=frame,
            iter_order=iter_order,
            prefix=prefix,
        )

    def mutagenize_orf(
        self,
        region: RegionType = None,
        *,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        mutation_type: str = "missense_only_first",
        codon_positions: Union[Sequence[Integral], slice, None] = None,
        style: Optional[str] = None,
        frame: Optional[int] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Apply codon-level mutations to an ORF sequence.

        Parameters
        ----------
        region : RegionType, default=None
            Region to mutate. Can be marker name (e.g., "orf") or [start, stop].
            If None, mutates the entire sequence.
        num_mutations : Optional[Integral], default=None
            Fixed number of codon mutations (mutually exclusive with mutation_rate).
        mutation_rate : Optional[Real], default=None
            Per-codon mutation probability (mutually exclusive with num_mutations).
        mutation_type : str, default='missense_only_first'
            Type of mutation: 'any_codon', 'nonsynonymous_first', 'nonsynonymous_random',
            'missense_only_first', 'missense_only_random', 'synonymous', 'nonsense'.
        codon_positions : Union[Sequence[Integral], slice, None], default=None
            Eligible codon indices: None (all), list of indices, or slice.
        style : Optional[str], default=None
            Style to apply to mutated codon positions (e.g., 'red', 'bold').
        frame : Optional[int], default=None
            Reading frame and orientation (+1/+2/+3/-1/-2/-3).
            If None and region is a named OrfRegion, uses the OrfRegion's frame.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'. Sequential requires
            ``num_mutations`` (not ``mutation_rate``) and a uniform ``mutation_type``.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'codon_positions'``,
            ``'wt_codons'``, ``'mut_codons'``, ``'wt_aas'``, ``'mut_aas'``.

        Returns
        -------
        Pool
            A Pool that generates codon-mutated sequences.
        """
        from ..orf_ops.mutagenize_orf import mutagenize_orf

        return mutagenize_orf(
            pool=self,
            region=region,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            mutation_type=mutation_type,
            codon_positions=codon_positions,
            style=style,
            frame=frame,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            cards=cards,
        )

    def translate(
        self,
        region: RegionType = None,
        *,
        frame: Optional[int] = None,
        include_stop: bool = True,
        preserve_codon_styles: bool = True,
        genetic_code: Union[str, dict] = "standard",
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> "ProteinPool":
        """Translate DNA sequence to protein.

        Parameters
        ----------
        region : RegionType, default=None
            Region to translate. Can be a marker name or [start, stop] interval.
            If None, translates the entire sequence.
        frame : Optional[int], default=None
            Reading frame: +1, +2, +3, -1, -2, -3.
            If None and region is an OrfRegion, uses its frame; otherwise +1.
        include_stop : bool, default=True
            Whether to include stop codon (*) in output.
        preserve_codon_styles : bool, default=True
            If True, propagate styles to amino acids when all 3 nucleotides
            of a codon share the same style.
        genetic_code : Union[str, dict], default="standard"
            Genetic code to use for translation.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        ProteinPool
            A Pool containing translated protein sequences.
        """
        from ..orf_ops.translate import translate

        return translate(
            pool=self,
            region=region,
            frame=frame,
            include_stop=include_stop,
            preserve_codon_styles=preserve_codon_styles,
            genetic_code=genetic_code,
            iter_order=iter_order,
            prefix=prefix,
        )
