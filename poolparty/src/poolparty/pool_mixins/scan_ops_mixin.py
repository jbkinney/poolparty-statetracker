"""Scan operation mixins for Pool class."""

from ..types import (
    Integral,
    Literal,
    ModeType,
    MultiPositionsType,
    Optional,
    Pool_type,
    PositionsType,
    Real,
    RegionType,
    Sequence,
    Union,
)


class ScanOpsMixin:
    """Mixin providing scan operation methods for Pool."""

    def mutagenize_scan(
        self,
        mutagenize_length: Integral,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[Union[str, Sequence[str]]] = None,
        mode: Union[ModeType, tuple[ModeType, ModeType]] = "random",
        num_states: Optional[Union[Integral, Sequence[Integral]]] = None,
        style: Optional[str] = None,
        iter_order: Optional[Union[Real, Sequence[Real]]] = None,
    ) -> Pool_type:
        from ..scan_ops.mutagenize_scan import mutagenize_scan

        return mutagenize_scan(
            pool=self,
            mutagenize_length=mutagenize_length,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            positions=positions,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
        )

    def deletion_scan(
        self,
        deletion_length: Integral,
        deletion_marker: Optional[str] = "-",
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..scan_ops.deletion_scan import deletion_scan

        return deletion_scan(
            pool=self,
            deletion_length=deletion_length,
            deletion_marker=deletion_marker,
            region=region,
            positions=positions,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
        )

    def insertion_scan(
        self,
        ins_pool: Union[Pool_type, str],
        positions: PositionsType = None,
        region: RegionType = None,
        replace: bool = False,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_insert: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..scan_ops.insertion_scan import insertion_scan

        return insertion_scan(
            pool=self,
            ins_pool=ins_pool,
            positions=positions,
            region=region,
            replace=replace,
            style=style,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_insert=prefix_insert,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )

    def replacement_scan(
        self,
        ins_pool: Union[Pool_type, str],
        positions: PositionsType = None,
        region: RegionType = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_insert: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..scan_ops.insertion_scan import replacement_scan

        return replacement_scan(
            pool=self,
            ins_pool=ins_pool,
            positions=positions,
            region=region,
            style=style,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_insert=prefix_insert,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )

    def shuffle_scan(
        self,
        shuffle_length: Integral,
        positions: PositionsType = None,
        region: RegionType = None,
        shuffles_per_position: Integral = 1,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_shuffle: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..scan_ops.shuffle_scan import shuffle_scan

        return shuffle_scan(
            pool=self,
            shuffle_length=shuffle_length,
            positions=positions,
            region=region,
            shuffles_per_position=shuffles_per_position,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_shuffle=prefix_shuffle,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
        )

    def subseq_scan(
        self,
        seq_length: Integral,
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..scan_ops.subseq_scan import subseq_scan

        return subseq_scan(
            pool=self,
            seq_length=seq_length,
            positions=positions,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )

    def deletion_multiscan(
        self,
        deletion_length: Integral,
        num_deletions: Integral,
        deletion_marker: Optional[str] = "-",
        positions: MultiPositionsType = None,
        region: RegionType = None,
        names: Optional[Sequence[str]] = None,
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..multiscan_ops.deletion_multiscan import deletion_multiscan

        return deletion_multiscan(
            pool=self,
            deletion_length=deletion_length,
            num_deletions=num_deletions,
            deletion_marker=deletion_marker,
            positions=positions,
            region=region,
            names=names,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
        )

    def insertion_multiscan(
        self,
        num_insertions: Integral,
        insertion_pools: Union[Pool_type, Sequence[Pool_type]],
        positions: MultiPositionsType = None,
        region: RegionType = None,
        names: Optional[Sequence[str]] = None,
        replace: bool = False,
        insertion_mode: Literal["ordered", "unordered"] = "ordered",
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..multiscan_ops.insertion_multiscan import insertion_multiscan

        return insertion_multiscan(
            pool=self,
            num_insertions=num_insertions,
            insertion_pools=insertion_pools,
            positions=positions,
            region=region,
            names=names,
            replace=replace,
            insertion_mode=insertion_mode,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )

    def replacement_multiscan(
        self,
        num_replacements: Integral,
        replacement_pools: Union[Pool_type, Sequence[Pool_type]],
        positions: MultiPositionsType = None,
        region: RegionType = None,
        names: Optional[Sequence[str]] = None,
        insertion_mode: Literal["ordered", "unordered"] = "ordered",
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..multiscan_ops.insertion_multiscan import replacement_multiscan

        return replacement_multiscan(
            pool=self,
            num_replacements=num_replacements,
            replacement_pools=replacement_pools,
            positions=positions,
            region=region,
            names=names,
            insertion_mode=insertion_mode,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
