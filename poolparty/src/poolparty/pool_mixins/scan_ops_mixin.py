"""Scan operation mixins for Pool class."""

from typing_extensions import Self

from ..types import (
    CardsType,
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
        num_states: Optional[Union[Integral, Sequence[Optional[Integral]]]] = None,
        style: Optional[str] = None,
        iter_order: Optional[Union[Real, Sequence[Real]]] = None,
        cards: Optional[tuple[CardsType, CardsType]] = None,
    ) -> Self:
        """Apply mutagenesis within a window at specified scanning positions.

        Parameters
        ----------
        mutagenize_length : Integral
            Length of the region to mutagenize at each position.
        num_mutations : Optional[Integral], default=None
            Fixed number of mutations to apply (mutually exclusive with mutation_rate).
        mutation_rate : Optional[Real], default=None
            Probability of mutation at each position (mutually exclusive with num_mutations).
        positions : PositionsType, default=None
            Positions to consider for the start of the mutagenize region (0-based).
            If None, all valid positions are used.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        prefix : Optional[Union[str, Sequence[str]]], default=None
            Prefix for sequence names.
            If a 2-tuple, first element is for scanning positions, second for mutagenization.
        mode : Union[ModeType, tuple[ModeType, ModeType]], default='random'
            Selection mode: 'random' or 'sequential'. A scalar value is broadcast
            to both scan and mutagenize sub-operations. If a 2-tuple, first element
            is for scanning positions, second for mutagenization.
        num_states : Optional[Union[Integral, Sequence[Optional[Integral]]]], default=None
            Number of states. A scalar value is broadcast to both sub-operations.
            If a 2-tuple, first element is for scanning positions, second for mutagenization.
            In sequential mode, overrides the computed count (cycling if greater,
            clipping if less). In random mode, if None defaults to 1 (pure random sampling).
        style : Optional[str], default=None
            Style to apply to mutated characters (e.g., 'red', 'blue bold').
        iter_order : Optional[Union[Real, Sequence[Real]]], default=None
            Iteration order priority for the Operation.
            If a 2-tuple, first element is for scanning positions, second for mutagenization.
        cards : Optional[tuple[CardsType, CardsType]], default=None
            Design card keys as a 2-tuple ``(scan_cards, mutagenize_cards)``.
            Scan keys: ``'position_index'``, ``'start'``, ``'end'``, ``'name'``,
            ``'region_seq'``. Mutagenize keys: ``'positions'``, ``'wt_chars'``,
            ``'mut_chars'``.

        Returns
        -------
        Pool
            A Pool yielding sequences where a region of the specified length is mutagenized
            at each allowed position.
        """
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
            cards=cards,
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
        cards: CardsType = None,
    ) -> Self:
        """Scan for all possible single deletions of a fixed length.

        Parameters
        ----------
        deletion_length : Integral
            Number of characters to delete at each valid position.
        deletion_marker : Optional[str], default='-'
            Character to insert at the deletion site. If None, segment is removed.
        positions : PositionsType, default=None
            Positions to consider for the start of the deletion (0-based, relative to region).
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        style : Optional[str], default=None
            Style to apply to deletion gap characters (e.g., 'gray', 'red bold').
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys: ``'position_index'``,
            ``'start'``, ``'end'``, ``'name'``, ``'region_seq'``.

        Returns
        -------
        Pool
            A Pool yielding sequences where a segment of the specified length is removed
            from the source at each allowed position.
        """
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
            cards=cards,
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
        cards: CardsType = None,
    ) -> Self:
        """Insert a sequence at specified scanning positions.

        Parameters
        ----------
        ins_pool : Union[Pool, str]
            The insert Pool or sequence string to be inserted.
        positions : PositionsType, default=None
            Positions for insertion (0-based). If None, all valid positions.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        replace : bool, default=False
            If False, insert at position (output length = bg + ins).
            If True, replace content at position (output length = bg).
        style : Optional[str], default=None
            Style to apply to inserted content (e.g., 'red', 'blue bold').
        prefix : Optional[str], default=None
            Prefix for cartesian product index.
        prefix_position : Optional[str], default=None
            Prefix for position index.
        prefix_insert : Optional[str], default=None
            Prefix for insert index.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys: ``'position_index'``,
            ``'start'``, ``'end'``, ``'name'``, ``'region_seq'``.

        Returns
        -------
        Pool
            A Pool yielding sequences with the insert placed at selected position(s).
        """
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
            cards=cards,
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
        cards: CardsType = None,
    ) -> Self:
        """Replace a segment with insert at specified scanning positions.

        Equivalent to ``insertion_scan(..., replace=True)``.

        Parameters
        ----------
        ins_pool : Union[Pool, str]
            The insert Pool or sequence string to be inserted.
        positions : PositionsType, default=None
            Positions for replacement (0-based). If None, all valid positions.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or
            [start, stop] interval.
        style : Optional[str], default=None
            Style to apply to inserted content (e.g., 'red', 'blue bold').
        prefix : Optional[str], default=None
            Prefix for cartesian product index.
        prefix_position : Optional[str], default=None
            Prefix for position index.
        prefix_insert : Optional[str], default=None
            Prefix for insert index.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed
            count (cycling if greater, clipping if less). In random mode,
            if None defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys:
            ``'position_index'``, ``'start'``, ``'end'``, ``'name'``,
            ``'region_seq'``.

        Returns
        -------
        Pool
            A Pool yielding sequences with the insert replacing content at
            selected position(s).
        """
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
            cards=cards,
        )

    def shuffle_scan(
        self,
        shuffle_length: Integral,
        positions: PositionsType = None,
        region: RegionType = None,
        shuffle_type: Literal["mono", "dinuc"] = "mono",
        shuffles_per_position: Integral = 1,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_shuffle: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: Optional[tuple[CardsType, CardsType]] = None,
    ) -> Self:
        """Shuffle characters within a window at specified scanning positions.

        Parameters
        ----------
        shuffle_length : Integral
            Length of the region to shuffle at each position.
        positions : PositionsType, default=None
            Positions to consider for the start of the shuffle region (0-based).
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        shuffle_type : Literal["mono", "dinuc"], default="mono"
            Type of shuffle: ``"mono"`` for random permutation or ``"dinuc"``
            for Euler-path shuffle preserving dinucleotide frequencies.
        shuffles_per_position : Integral, default=1
            Number of shuffles to perform at each position.
        prefix : Optional[str], default=None
            Prefix for cartesian product index.
        prefix_position : Optional[str], default=None
            Prefix for position index.
        prefix_shuffle : Optional[str], default=None
            Prefix for shuffle variant index.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        style : Optional[str], default=None
            Style to apply to shuffled characters (e.g., 'purple', 'red bold').
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : Optional[tuple[CardsType, CardsType]], default=None
            Design card keys as a 2-tuple ``(scan_cards, shuffle_cards)``.
            Scan keys: ``'position_index'``, ``'start'``, ``'end'``, ``'name'``,
            ``'region_seq'``. Shuffle keys: ``'permutation'``.

        Returns
        -------
        Pool
            A Pool yielding sequences where a region of the specified length is shuffled
            at each allowed position.
        """
        from ..scan_ops.shuffle_scan import shuffle_scan

        return shuffle_scan(
            pool=self,
            shuffle_length=shuffle_length,
            positions=positions,
            region=region,
            shuffle_type=shuffle_type,
            shuffles_per_position=shuffles_per_position,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_shuffle=prefix_shuffle,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
            cards=cards,
        )

    def subseq_scan(
        self,
        subseq_length: Integral,
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Extract subsequences of a specified length at scanning positions.

        Parameters
        ----------
        subseq_length : Integral
            Length of subsequence to extract at each position.
        positions : PositionsType, default=None
            Positions to consider for the start of extraction (0-based).
            If None, all valid positions are used.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys: ``'position_index'``,
            ``'start'``, ``'end'``, ``'name'``, ``'region_seq'``.

        Returns
        -------
        Pool
            A Pool yielding subsequences extracted at each allowed position.
        """
        from ..scan_ops.subseq_scan import subseq_scan

        return subseq_scan(
            pool=self,
            subseq_length=subseq_length,
            positions=positions,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            cards=cards,
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
        cards: CardsType = None,
    ) -> Self:
        """Delete segments at multiple positions simultaneously.

        Parameters
        ----------
        deletion_length : Integral
            Number of characters to delete at each position.
        num_deletions : Integral
            Number of simultaneous deletions to make.
        deletion_marker : Optional[str], default='-'
            Character to insert at each deletion site. If None, deleted segments
            are removed with no marker.
        positions : MultiPositionsType, default=None
            Valid positions for deletion starts (0-based). Can be a flat list
            (shared across all deletions) or a list of per-deletion sublists.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        names : Optional[Sequence[str]], default=None
            Custom names for the deletion regions. If None, auto-generated.
        min_spacing : Optional[Integral], default=None
            Minimum gap between end of one deletion and start of next.
        max_spacing : Optional[Integral], default=None
            Maximum gap between adjacent deletions. None = unbounded.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        style : Optional[str], default=None
            Style to apply to deletion marker characters (e.g., 'gray', 'red bold').
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys: ``'combination_index'``,
            ``'starts'``, ``'ends'``, ``'names'``, ``'region_seqs'``.

        Returns
        -------
        Pool
            A Pool yielding sequences with multiple segments deleted simultaneously.
        """
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
            cards=cards,
        )

    def insertion_multiscan(
        self,
        num_insertions: Integral,
        insertion_pools: Union[Pool_type, Sequence[Pool_type]],
        positions: MultiPositionsType = None,
        region: RegionType = None,
        names: Optional[Sequence[str]] = None,
        replace: bool = False,
        style: Optional[str] = None,
        insertion_mode: Literal["ordered", "unordered"] = "ordered",
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Insert sequences at multiple positions simultaneously.

        Parameters
        ----------
        num_insertions : Integral
            Number of simultaneous insertions to make.
        insertion_pools : Union[Pool, Sequence[Pool]]
            Pool(s) providing content. If a single Pool is provided,
            it will be deepcopied ``num_insertions - 1`` times. If a Sequence,
            its length must equal ``num_insertions``.
        positions : MultiPositionsType, default=None
            Valid positions (0-based). Can be a flat list (shared across all
            insertions) or a list of per-insertion sublists.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        names : Optional[Sequence[str]], default=None
            Custom names for the insertion regions. If None, auto-generated.
        replace : bool, default=False
            If True, replace existing content at each position.
            If False, insert at zero-width positions.
        style : Optional[str], default=None
            Style to apply to inserted content (e.g., 'red', 'blue bold').
        insertion_mode : Literal['ordered', 'unordered'], default='ordered'
            How to assign pools to positions. ``'ordered'`` preserves position
            order; ``'unordered'`` uses all permutations.
        min_spacing : Optional[Integral], default=None
            Minimum gap between adjacent positions.
        max_spacing : Optional[Integral], default=None
            Maximum gap between adjacent positions. None = unbounded.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys: ``'combination_index'``,
            ``'starts'``, ``'ends'``, ``'names'``, ``'region_seqs'``.

        Returns
        -------
        Pool
            A Pool yielding sequences with multiple insertions.
        """
        from ..multiscan_ops.insertion_multiscan import insertion_multiscan

        return insertion_multiscan(
            pool=self,
            num_insertions=num_insertions,
            insertion_pools=insertion_pools,
            positions=positions,
            region=region,
            names=names,
            replace=replace,
            style=style,
            insertion_mode=insertion_mode,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            cards=cards,
        )

    def replacement_multiscan(
        self,
        num_replacements: Integral,
        replacement_pools: Union[Pool_type, Sequence[Pool_type]],
        positions: MultiPositionsType = None,
        region: RegionType = None,
        names: Optional[Sequence[str]] = None,
        style: Optional[str] = None,
        insertion_mode: Literal["ordered", "unordered"] = "ordered",
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Replace segments at multiple positions simultaneously.

        Equivalent to ``insertion_multiscan(..., replace=True)``.

        Parameters
        ----------
        num_replacements : Integral
            Number of simultaneous replacements to make.
        replacement_pools : Union[Pool, Sequence[Pool]]
            Pool(s) providing replacement content. If a single Pool is
            provided, it will be deepcopied ``num_replacements - 1`` times.
            If a Sequence, its length must equal ``num_replacements``.
        positions : MultiPositionsType, default=None
            Valid positions (0-based). Can be a flat list (shared across
            all replacements) or a list of per-replacement sublists.
        region : RegionType, default=None
            Region to constrain the scan to. Can be a marker name or
            [start, stop] interval.
        names : Optional[Sequence[str]], default=None
            Custom names for the replacement regions. If None,
            auto-generated.
        style : Optional[str], default=None
            Style to apply to replaced content (e.g., 'red', 'blue bold').
        insertion_mode : Literal['ordered', 'unordered'], default='ordered'
            How to assign pools to positions. ``'ordered'`` preserves
            position order; ``'unordered'`` uses all permutations.
        min_spacing : Optional[Integral], default=None
            Minimum gap between adjacent positions.
        max_spacing : Optional[Integral], default=None
            Maximum gap between adjacent positions. None = unbounded.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[Integral], default=None
            Number of states. In sequential mode, overrides the computed
            count (cycling if greater, clipping if less). In random mode,
            if None defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : CardsType, default=None
            Design card keys to include. Available keys:
            ``'combination_index'``, ``'starts'``, ``'ends'``,
            ``'names'``, ``'region_seqs'``.

        Returns
        -------
        Pool
            A Pool yielding sequences with multiple replacements.
        """
        from ..multiscan_ops.insertion_multiscan import replacement_multiscan

        return replacement_multiscan(
            pool=self,
            num_replacements=num_replacements,
            replacement_pools=replacement_pools,
            positions=positions,
            region=region,
            names=names,
            style=style,
            insertion_mode=insertion_mode,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            cards=cards,
        )
