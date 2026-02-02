"""DNA-specific operation mixins for DnaPool class."""

import pandas as pd

from ..types import (
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
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..base_ops.from_iupac import from_iupac

        return from_iupac(
            iupac_seq=iupac_seq,
            bg_pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
        )

    def insert_from_motif(
        self,
        prob_df: pd.DataFrame,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..base_ops.from_motif import from_motif

        return from_motif(
            prob_df=prob_df,
            bg_pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
        name: str,
        extent: Optional[tuple[int, int]] = None,
        frame: int = 1,
        style: Optional[str] = None,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Pool_type:
        """Annotate an ORF region with frame, optionally applying styling."""
        from ..orf_ops.annotate_orf import annotate_orf

        return annotate_orf(
            self,
            name,
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
    ) -> Pool_type:
        """Apply ORF-aware styling."""
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
    ) -> Pool_type:
        """Apply codon-level mutations."""
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
    ):
        """Translate DNA to protein."""
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
