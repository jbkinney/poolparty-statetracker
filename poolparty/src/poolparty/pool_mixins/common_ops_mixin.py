"""Common operation mixins for Pool class - generic operations that work on any sequence type."""

from typing_extensions import Self

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
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
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
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
    ) -> Self:
        """Shuffle sequence characters within a region.

        Parameters
        ----------
        region : RegionType, default=None
            Region to shuffle. Can be marker name (str), [start, stop], or None.
        shuffle_type : Literal['mono', 'dinuc'], default='mono'
            Type of shuffle: 'mono' (mononucleotide) or 'dinuc' (dinucleotide-preserving).
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        mode : ModeType, default='random'
            Selection mode: 'random' or 'sequential'.
        num_states : Optional[int], default=None
            Number of states for random mode. If None, defaults to 1.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        style : Optional[str], default=None
            Style to apply to shuffled characters (e.g., 'red', 'blue bold').
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'permutation'``.

        Returns
        -------
        Pool
            A Pool containing shuffled sequences.
        """
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
        num_states: Optional[Integral] = None,
        prefix: Optional[str] = None,
        styles: Optional[list[str]] = None,
        style_by: StyleByForRecombineType = "order",
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
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
    ) -> Self:
        """Filter sequences based on a predicate function.

        Sequences for which the predicate returns False are replaced with
        NullSeq, which propagates through downstream operations. Use
        generate_library with discard_null_seqs=True to exclude filtered
        sequences from output.

        Parameters
        ----------
        predicate : Callable[[str], bool]
            Function taking sequence string (clean, no tags), returning
            True to keep.
        name : Optional[str], default=None
            Optional name for the operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'passed'``.

        Returns
        -------
        Pool
            New pool that may contain NullSeq for filtered sequences.
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
    ) -> Self:
        """Score sequences using a custom function and record values on design cards.

        Parameters
        ----------
        fn : Callable[[str], Any]
            Scoring function that takes a sequence string and returns a value.
        card_key : str, default='score'
            Key name for the score column in design cards.
        region : RegionType, default=None
            Region to extract before scoring. Can be marker name (str),
            [start, stop], or None to score the full sequence.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        cards : list[str] or dict, optional
            Design card keys to include. The available key is the ``card_key``
            value (default ``'score'``).

        Returns
        -------
        Pool
            The same Pool, with scores recorded on design cards.
        """
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
    ) -> Self:
        """Materialize sequences into a new pool with fixed states.

        Generates sequences from this pool and creates a new pool that stores
        them. The resulting pool has a well-defined num_states and no parent
        references (severed DAG), making it a leaf node in any future
        computation.

        Parameters
        ----------
        num_seqs : Optional[Integral], default=None
            Number of sequences to generate and store. Mutually exclusive
            with num_cycles.
        num_cycles : Optional[Integral], default=None
            Number of complete cycles through the source pool's state space.
            If both num_seqs and num_cycles are None, defaults to 1 cycle.
            Mutually exclusive with num_seqs.
        seed : Optional[Integral], default=None
            Random seed for reproducible generation.
        discard_null_seqs : bool, default=True
            If True, filtered/null sequences are excluded. If False, they
            are included as NullSeq objects.
        max_iterations : Optional[Integral], default=None
            Maximum iterations before stopping (only used with num_seqs).
        min_acceptance_rate : Optional[Real], default=None
            Minimum fraction of sequences that must pass filters.
        attempts_per_rate_assessment : Integral, default=100
            Iterations between acceptance rate checks.
        name : Optional[str], default=None
            Optional name for the operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        cards : list[str] or dict, optional
            Design card keys to include.

        Returns
        -------
        Pool
            A new Pool containing the materialized sequences with a fixed
            number of states equal to the number of stored sequences.
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
