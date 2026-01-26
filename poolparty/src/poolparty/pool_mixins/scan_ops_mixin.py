"""Scan operation mixins for Pool class."""
from ..types import Pool_type, RegionType, Optional, Real, Integral, ModeType, beartype, Sequence, Union, PositionsType


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
        mode: Union[ModeType, tuple[ModeType, ModeType]] = 'random',
        num_states: Optional[Union[Integral, Sequence[Integral]]] = None,
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
            iter_order=iter_order,
        )
    
    def deletion_scan(
        self,
        deletion_length: Integral,
        deletion_marker: Optional[str] = '-',
        region: RegionType = None,
        positions: PositionsType = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
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
        mode: ModeType = 'random',
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
        mode: ModeType = 'random',
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
        mode: ModeType = 'random',
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
