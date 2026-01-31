"""Base operation mixins for Pool class."""

from typing import Literal

import pandas as pd

from ..types import (
    Callable,
    Integral,
    ModeType,
    Optional,
    Pool_type,
    Real,
    RegionType,
    Sequence,
    StyleByForRecombineType,
    Union,
)


class BaseOpsMixin:
    """Mixin providing base operation methods for Pool."""

    def mutagenize(
        self,
        region: RegionType = None,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        allowed_chars: Optional[str] = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..base_ops.mutagenize import mutagenize

        return mutagenize(
            pool=self,
            region=region,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            allowed_chars=allowed_chars,
            style=style,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )

    def shuffle_seq(
        self,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..base_ops.shuffle_seq import shuffle_seq

        return shuffle_seq(
            pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
        )

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

    def recombine(
        self,
        region: RegionType = None,
        sources: Sequence[Union[Pool_type, str]] = (),
        num_breakpoints: Integral = 1,
        positions: Optional[Sequence[Integral]] = None,
        mode: ModeType = "random",
        num_states: Optional[int] = None,
        prefix: Optional[str] = None,
        styles: Optional[list[str]] = None,
        style_by: StyleByForRecombineType = "order",
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..base_ops.recombine import recombine

        return recombine(
            pool=self,
            region=region,
            sources=sources,
            num_breakpoints=num_breakpoints,
            positions=positions,
            mode=mode,
            num_states=num_states,
            prefix=prefix,
            styles=styles,
            style_by=style_by,
            iter_order=iter_order,
        )

    def filter(
        self,
        predicate: Callable[[str], bool],
        name: Optional[str] = None,
    ) -> Pool_type:
        """Filter sequences based on a predicate function.

        Sequences for which the predicate returns False are replaced with NullSeq.
        Use generate_library with discard_null_seqs=True to exclude them.
        """
        from ..base_ops.filter_seq import filter_seq

        return filter_seq(self, predicate=predicate, name=name)
