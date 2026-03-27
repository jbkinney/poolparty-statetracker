"""Common operation mixins for Pool class - generic operations that work on any sequence type."""

from ..types import (
    Any,
    Callable,
    CardsType,
    Integral,
    Literal,
    ModeType,
    Optional,
    Pool_type,
    Real,
    RegionType,
    Sequence,
    StyleByForRecombineType,
    Union,
)


class CommonOpsMixin:
    """Mixin providing common operation methods for Pool (works on any sequence type)."""

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
        cards: CardsType = None,
    ) -> Pool_type:
        """Apply mutations to a sequence.

        Parameters
        ----------
        region : Union[str, Sequence[Integral], None], default=None
            Region to mutagenize. Can be a marker name (str), explicit interval
            [start, stop], or None to mutagenize entire sequence.
        num_mutations : Optional[Integral], default=None
            Fixed number of mutations to apply (mutually exclusive with mutation_rate).
        mutation_rate : Optional[Real], default=None
            Probability of mutation at each position (mutually exclusive with num_mutations).
        allowed_chars : Optional[str], default=None
            IUPAC string of same length as sequence, specifying allowed bases at each
            position. Positions where only the wild-type is allowed are non-mutable.
        style : Optional[str], default=None
            Style to apply to mutated positions (e.g., 'red', 'blue bold').
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'. Sequential only available
            with num_mutations.
        num_states : Optional[int], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'positions'``,
            ``'wt_chars'``, ``'mut_chars'``.

        Returns
        -------
        Pool
            A Pool that generates mutated sequences.
        """
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
            cards=cards,
        )

    def shuffle_seq(
        self,
        region: RegionType = None,
        shuffle_type: Literal["mono", "dinuc"] = "mono",
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> Pool_type:
        from ..base_ops.shuffle_seq import shuffle_seq

        return shuffle_seq(
            pool=self,
            region=region,
            shuffle_type=shuffle_type,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
            cards=cards,
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
        cards: CardsType = None,
    ) -> Pool_type:
        """Recombine segments from multiple source pools at breakpoints.

        Parameters
        ----------
        region : Union[str, Sequence[Integral], None], default=None
            Region where recombined sequences will be inserted. Region content
            is discarded (not used as a source pool).
        sources : Sequence[Union[Pool, str]], default=()
            Source pools for recombination. All must have the same seq_length.
        num_breakpoints : Integral, default=1
            Number of recombination breakpoints. Must be <= seq_length - 1.
        positions : Optional[Sequence[Integral]], default=None
            Valid breakpoint positions. If None, defaults to range(seq_length - 1).
            Position i means "breakpoint after index i".
        mode : ModeType, default='random'
            Selection mode: 'random' (random breakpoints and pool assignments) or
            'sequential' (enumerate all combinations).
        num_states : Optional[int], default=None
            Number of states. In sequential mode, overrides the computed count
            (cycling if greater, clipping if less). In random mode, if None
            defaults to 1 (pure random sampling).
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        styles : Optional[list[str]], default=None
            List of styles to apply to segments. Cycles through the list.
            Use style_by to control whether cycling is by segment position or source.
        style_by : StyleByForRecombineType, default='order'
            How styles are assigned: 'order' (by segment position) or
            'source' (by source pool index).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'breakpoints'``,
            ``'pool_assignments'``.

        Returns
        -------
        Pool
            A Pool that generates recombined sequences.
        """
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
            cards=cards,
        )

    def filter(
        self,
        predicate: Callable[[str], bool],
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> Pool_type:
        """Filter sequences based on a predicate function.

        Sequences for which the predicate returns False are replaced with NullSeq.
        Use generate_library with discard_null_seqs=True to exclude them.
        """
        from ..base_ops.filter_seq import filter

        return filter(self, predicate=predicate, name=name, prefix=prefix, cards=cards)

    def score(
        self,
        fn: Callable[[str], Any],
        card_key: str = "score",
        region: RegionType = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> Pool_type:
        from ..fixed_ops.score import score

        return score(
            pool=self,
            fn=fn,
            card_key=card_key,
            region=region,
            prefix=prefix,
            cards=cards,
        )

    def materialize(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        seed: Optional[Integral] = None,
        discard_null_seqs: bool = True,
        max_iterations: Optional[Integral] = None,
        min_acceptance_rate: Optional[Real] = None,
        attempts_per_rate_assessment: Integral = 100,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> Pool_type:
        """Materialize sequences into a new pool with fixed states.

        Generates sequences from this pool and creates a new pool that stores
        them. The resulting pool has a well-defined num_states and no parent
        references (severed DAG).
        """
        from ..base_ops.materialize import materialize

        return materialize(
            pool=self,
            num_seqs=num_seqs,
            num_cycles=num_cycles,
            seed=seed,
            discard_null_seqs=discard_null_seqs,
            max_iterations=max_iterations,
            min_acceptance_rate=min_acceptance_rate,
            attempts_per_rate_assessment=attempts_per_rate_assessment,
            name=name,
            prefix=prefix,
            cards=cards,
        )
