"""Common operation mixins for Pool class - generic operations that work on any sequence type."""

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
        prefix: Optional[str] = None,
    ) -> Pool_type:
        """Filter sequences based on a predicate function.

        Sequences for which the predicate returns False are replaced with NullSeq.
        Use generate_library with discard_null_seqs=True to exclude them.
        """
        from ..base_ops.filter_seq import filter

        return filter(self, predicate=predicate, name=name, prefix=prefix)

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
        )
