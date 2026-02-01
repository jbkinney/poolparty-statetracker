"""ORF operation mixins for Pool class."""

from ..types import Integral, ModeType, Optional, Pool_type, Real, RegionType, Sequence, Union


class OrfOpsMixin:
    """Mixin providing ORF operation methods for Pool."""

    def annotate_orf(
        self,
        name: str,
        extent: Optional[tuple[int, int]] = None,
        frame: int = 1,
        style: Optional[str] = None,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Annotate an ORF region with frame, optionally applying styling. See annotate_orf() for details."""
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
        )

    def stylize_orf(
        self,
        region: RegionType = None,
        *,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        frame: Optional[int] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Apply ORF-aware styling. See stylize_orf() for details."""
        from ..orf_ops.stylize_orf import stylize_orf

        return stylize_orf(
            pool=self,
            region=region,
            style_codons=style_codons,
            style_frames=style_frames,
            frame=frame,
            iter_order=iter_order,
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
        """Apply codon-level mutations. See mutagenize_orf() for details."""
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
