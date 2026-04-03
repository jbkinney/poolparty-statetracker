"""Bravo audit tests for state operations (C7).

Fills gaps from the alpha pass by testing:
  - Mixin forwarding (F1: repeat missing cards)
  - I2 explicit exhaustion for sample, state_slice, state_shuffle
  - I4 tag preservation through all state ops
  - I7 composition: repeat + downstream sequential op
  - Pool type preservation (DnaPool stays DnaPool)
  - Domain-specific: state count arithmetic, pass-through content identity
  - Stack adversarial: tagged parents, composed upstream, single-pool
  - Sync adversarial: downstream chain, 3-pool lockstep, trivial, single
  - Contract tracing: stack orphaned state, Cartesian product downstream
"""

import pytest

import poolparty as pp
from poolparty.dna_pool import DnaPool
from poolparty.pool import Pool
from poolparty.utils.parsing_utils import strip_all_tags


def _bio_len(seq: str) -> int:
    return len(strip_all_tags(seq))


# ===================================================================
# F1: repeat mixin missing cards forwarding
# ===================================================================


def test_f1_repeat_mixin_missing_cards() -> None:
    with pp.Party():
        base = pp.from_seqs(["AA", "BB"], mode="sequential")
        repeated = base.repeat(2, cards=["repeat_index"]).named("rep")
        df = repeated.generate_library(num_cycles=1)
    card_col = [c for c in df.columns if c.endswith(".repeat_index")][0]
    assert list(df[card_col]) == [0, 1, 0, 1]


# ===================================================================
# Mixin forwarding: verify other params work at runtime
# ===================================================================


class TestMixinForwardingRuntime:

    def test_repeat_mixin_prefix(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            repeated = base.repeat(2, prefix="rep").named("r")
        df = repeated.generate_library(num_cycles=1)
        assert all("rep" in n for n in df["name"] if n)

    def test_repeat_mixin_iter_order(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            repeated = base.repeat(2, iter_order=5.0)
        assert repeated.operation.state.iter_order == 5.0

    def test_sample_mixin_all_params(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sampled = base.sample(num_seqs=2, seed=42, with_replacement=False, prefix="s")
        assert sampled.num_states == 2

    def test_shuffle_states_mixin_all_params(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C"], mode="sequential")
            shuffled = base.shuffle_states(seed=99, prefix="sh")
        assert shuffled.num_states == 3

    def test_slice_states_mixin_all_params(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sliced = base.slice_states(slice(0, 2), prefix="sl")
        assert sliced.num_states == 2


# ===================================================================
# I2: Explicit exhaustion for sample, state_slice, state_shuffle
# ===================================================================


class TestI2Exhaustion:

    def test_sample_num_seqs_exhaustion(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = pp.sample(base, num_seqs=3, seed=7).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == sampled.num_states == 3

    def test_sample_seq_states_exhaustion(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sampled = pp.sample(base, seq_states=[0, 2, 3]).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == sampled.num_states == 3
        assert df["seq"].tolist() == ["A", "C", "D"]

    def test_state_slice_exhaustion(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = pp.state_slice(base, slice(1, 4)).named("sl")
        df = sliced.generate_library(num_cycles=1)
        assert len(df) == sliced.num_states == 3
        assert df["seq"].tolist() == ["B", "C", "D"]

    def test_state_shuffle_exhaustion(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled = pp.state_shuffle(base, seed=42).named("sh")
        df = shuffled.generate_library(num_cycles=1)
        assert len(df) == shuffled.num_states == 5
        assert sorted(df["seq"]) == ["A", "B", "C", "D", "E"]


# ===================================================================
# I4: Tag preservation through state ops
# ===================================================================


class TestI4TagPreservation:

    def _make_tagged_pool(self):
        return pp.from_seq("AA<rgn>CC</rgn>GG")

    def test_stack_preserves_tags(self) -> None:
        with pp.Party():
            a = self._make_tagged_pool()
            b = pp.from_seq("TTTTTT")
            stacked = pp.stack([a, b]).named("stacked")
        df = stacked.generate_library(num_cycles=1)
        tagged_seq = df["seq"].iloc[0]
        assert "<rgn>" in tagged_seq and "</rgn>" in tagged_seq

    def test_repeat_preserves_tags(self) -> None:
        with pp.Party():
            base = self._make_tagged_pool()
            repeated = pp.repeat(base, times=2).named("rep")
        df = repeated.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "<rgn>" in seq and "</rgn>" in seq

    def test_sample_preserves_tags(self) -> None:
        with pp.Party():
            base = self._make_tagged_pool()
            sampled = pp.sample(base, seq_states=[0]).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert "<rgn>" in df["seq"].iloc[0]

    def test_state_slice_preserves_tags(self) -> None:
        with pp.Party():
            base = self._make_tagged_pool()
            sliced = pp.state_slice(base, 0).named("sl")
        df = sliced.generate_library(num_cycles=1)
        assert "<rgn>" in df["seq"].iloc[0]

    def test_state_shuffle_preserves_tags(self) -> None:
        with pp.Party():
            base = self._make_tagged_pool()
            shuffled = pp.state_shuffle(base, seed=1).named("sh")
        df = shuffled.generate_library(num_cycles=1)
        assert "<rgn>" in df["seq"].iloc[0]


# ===================================================================
# I7: Composition — repeat + downstream sequential op
# ===================================================================


def test_i7_repeat_composition_with_mutagenize() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        repeated = pp.repeat(base, times=2).named("rep")
        downstream = pp.mutagenize(repeated, num_mutations=1, mode="sequential").named("m")
        df = downstream.generate_library(num_cycles=1)

    assert downstream.seq_length == 4
    assert len(df) == downstream.num_states
    for seq in df["seq"]:
        assert _bio_len(seq) == 4


# ===================================================================
# Pool type preservation
# ===================================================================


class TestPoolTypePreservation:

    def test_stack_preserves_dna_pool_type(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["AA", "BB"], mode="sequential")
            b = pp.from_seqs(["XX", "YY"], mode="sequential")
            stacked = pp.stack([a, b])
        assert isinstance(stacked, DnaPool)

    def test_repeat_preserves_dna_pool_type(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            repeated = pp.repeat(base, times=2)
        assert isinstance(repeated, DnaPool)

    def test_sample_preserves_dna_pool_type(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            sampled = pp.sample(base, seq_states=[0])
        assert isinstance(sampled, DnaPool)

    def test_state_slice_preserves_dna_pool_type(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            sliced = pp.state_slice(base, slice(0, 1))
        assert isinstance(sliced, DnaPool)

    def test_state_shuffle_preserves_dna_pool_type(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            shuffled = pp.state_shuffle(base, seed=1)
        assert isinstance(shuffled, DnaPool)

    def test_stack_with_base_pool_returns_base_pool(self) -> None:
        with pp.Party():
            dna_a = pp.from_seqs(["AA", "BB"], mode="sequential")
            dna_b = pp.from_seqs(["XX", "YY"], mode="sequential")
            a = Pool(operation=dna_a.operation)
            b = Pool(operation=dna_b.operation)
            stacked = pp.stack([a, b])
        assert type(stacked) is Pool


# ===================================================================
# Domain-specific: state count arithmetic
# ===================================================================


class TestStateCountArithmetic:

    def test_stack_num_states_equals_sum(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
            c = pp.from_seq("Q")
            stacked = pp.stack([a, b, c])
        assert stacked.num_states == 2 + 3 + 1

    def test_repeat_num_states_equals_parent_times(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C"], mode="sequential")
            repeated = pp.repeat(base, times=4)
        assert repeated.num_states == 3 * 4

    def test_sample_num_states_equals_num_seqs(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sampled = pp.sample(base, num_seqs=7, seed=1, with_replacement=True)
        assert sampled.num_states == 7

    def test_sample_num_states_equals_len_seq_states(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sampled = pp.sample(base, seq_states=[0, 1, 1, 3])
        assert sampled.num_states == 4

    def test_state_slice_num_states_matches_slice(self) -> None:
        with pp.Party():
            base = pp.from_seqs([chr(65 + i) for i in range(10)], mode="sequential")
            sliced = pp.state_slice(base, slice(2, 8, 2))
        assert sliced.num_states == 3  # indices 2, 4, 6

    def test_state_shuffle_preserves_num_states(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            shuffled = pp.state_shuffle(base, seed=10)
        assert shuffled.num_states == 4


# ===================================================================
# Domain-specific: pass-through content identity
# ===================================================================


class TestPassThroughContentIdentity:

    def test_sample_seq_states_returns_exact_parent_seqs(self) -> None:
        with pp.Party():
            seqs = ["ACGT", "TGCA", "GGGG"]
            base = pp.from_seqs(seqs, mode="sequential")
            sampled = pp.sample(base, seq_states=[2, 0]).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert df["seq"].tolist() == ["GGGG", "ACGT"]

    def test_state_slice_returns_exact_parent_seqs(self) -> None:
        with pp.Party():
            seqs = ["ACGT", "TGCA", "GGGG"]
            base = pp.from_seqs(seqs, mode="sequential")
            sliced = pp.state_slice(base, slice(1, 3)).named("sl")
        df = sliced.generate_library(num_cycles=1)
        assert df["seq"].tolist() == ["TGCA", "GGGG"]

    def test_state_shuffle_returns_same_set_of_seqs(self) -> None:
        with pp.Party():
            seqs = ["ACGT", "TGCA", "GGGG", "CCCC"]
            base = pp.from_seqs(seqs, mode="sequential")
            shuffled = pp.state_shuffle(base, seed=7).named("sh")
        df = shuffled.generate_library(num_cycles=1)
        assert sorted(df["seq"]) == sorted(seqs)

    def test_repeat_returns_parent_seqs_repeated(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            repeated = pp.repeat(base, times=3).named("rep")
        df = repeated.generate_library(num_cycles=1)
        assert df["seq"].tolist() == ["AA", "AA", "AA", "BB", "BB", "BB"]


# ===================================================================
# Stack adversarial
# ===================================================================


class TestStackAdversarial:

    def test_stack_tagged_parents_region_union(self) -> None:
        with pp.Party():
            a = pp.from_seq("AA<r1>CC</r1>GG")
            b = pp.from_seq("TT<r2>AA</r2>CC")
            stacked = pp.stack([a, b]).named("stacked")

        assert any(r.name == "r1" for r in stacked.regions)
        assert any(r.name == "r2" for r in stacked.regions)
        df = stacked.generate_library(num_cycles=1)
        assert "<r1>" in df["seq"].iloc[0]
        assert "<r2>" in df["seq"].iloc[1]

    def test_stack_composed_upstream_cartesian_with_downstream(self) -> None:
        with pp.Party():
            src = pp.from_seqs(["AC", "GT"], mode="sequential")
            mutated = pp.mutagenize(src, num_mutations=1, mode="sequential")
            other = pp.from_seq("XX")
            stacked = pp.stack([mutated, other]).named("stacked")
            downstream = pp.repeat(stacked, times=2).named("ds")
        df = downstream.generate_library(num_cycles=1)
        expected = stacked.num_states * 2
        assert downstream.num_states == expected
        assert len(df) == expected

    def test_stack_single_pool_identity(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB", "CC"], mode="sequential")
            stacked = pp.stack([base]).named("stacked")
        df = stacked.generate_library(num_cycles=1)
        assert stacked.num_states == 3
        assert df["seq"].tolist() == ["AA", "BB", "CC"]


# ===================================================================
# Sync adversarial
# ===================================================================


class TestSyncAdversarial:

    def test_sync_then_downstream_chain_collapses_state_count(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A1", "A2", "A3"], mode="sequential")
            b = pp.from_seqs(["B1", "B2", "B3"], mode="sequential")
            pp.sync([a, b])
            joined = pp.join([a, "-", b]).named("joined")
            repeated = pp.repeat(joined, times=2).named("rep")
        df = repeated.generate_library(num_cycles=1)
        assert joined.num_states == 3
        assert repeated.num_states == 6
        assert len(df) == 6

    def test_sync_three_pools_all_lockstep(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            c = pp.from_seqs(["1", "2"], mode="sequential")
            pp.sync([a, b, c])
            joined = pp.join([a, b, c]).named("joined")
        df = joined.generate_library(num_cycles=1)
        assert joined.num_states == 2
        assert df["seq"].tolist() == ["AX1", "BY2"]

    def test_sync_trivial_num_states_1(self) -> None:
        with pp.Party():
            a = pp.from_seq("AA")
            b = pp.from_seq("BB")
            pp.sync([a, b])
            joined = pp.join([a, b]).named("joined")
        df = joined.generate_library(num_cycles=1)
        assert joined.num_states == 1
        assert df["seq"].iloc[0] == "AABB"

    def test_sync_single_pool_no_crash(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["AA", "BB"], mode="sequential")
            pp.sync([a])
            df = a.generate_library(num_cycles=1)
        assert len(df) == 2
        assert df["seq"].tolist() == ["AA", "BB"]

    def test_sync_different_num_states_raises(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
            with pytest.raises(ValueError, match="different num_states"):
                pp.sync([a, b])

    def test_sync_empty_raises(self) -> None:
        with pp.Party():
            with pytest.raises(ValueError, match="empty"):
                pp.sync([])


# ===================================================================
# Contract tracing: stack orphaned state and downstream Cartesian
# ===================================================================


class TestStackContractTracing:

    def test_stack_op_state_num_values_is_branch_count(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
            stacked = pp.stack([a, b]).named("stacked")

        assert stacked.operation.num_states == 2
        assert stacked.operation.state._num_values == 2
        assert stacked.num_states == 5

    def test_stack_downstream_cartesian_product(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = pp.stack([a, b]).named("stacked")
            downstream = pp.repeat(stacked, times=3).named("rep")
        df = downstream.generate_library(num_cycles=1)
        assert downstream.num_states == 4 * 3
        assert len(df) == 12

    def test_stack_active_parent_maps_to_correct_pool(self) -> None:
        with pp.Party():
            a = pp.from_seqs(["A1", "A2"], mode="sequential")
            b = pp.from_seqs(["B1", "B2", "B3"], mode="sequential")
            stacked = pp.stack([a, b], cards=["active_parent"]).named("stacked")
        df = stacked.generate_library(num_cycles=1)
        card_col = [c for c in df.columns if c.endswith(".active_parent")][0]
        for _, row in df.iterrows():
            if row["seq"].startswith("A"):
                assert row[card_col] == 0
            else:
                assert row[card_col] == 1


# ===================================================================
# I10: State-space immutability for repeat
# ===================================================================


def test_i10_state_space_immutability_for_repeat() -> None:
    with pp.Party():
        base = pp.from_seqs(["A", "B"], mode="sequential")
        repeated = pp.repeat(base, times=3).named("rep")
        ns_before = repeated.num_states
        sv_before = repeated.operation.state._num_values
        repeated.generate_library(num_cycles=1)
    assert repeated.num_states == ns_before
    assert repeated.operation.state._num_values == sv_before


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:

    def test_repeat_times_1_is_identity(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB", "CC"], mode="sequential")
            repeated = pp.repeat(base, times=1).named("rep")
        df = repeated.generate_library(num_cycles=1)
        assert repeated.num_states == 3
        assert df["seq"].tolist() == ["AA", "BB", "CC"]

    def test_repeat_times_0_raises(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["AA", "BB"], mode="sequential")
            with pytest.raises(ValueError, match="times must be >= 1"):
                pp.repeat(base, times=0)

    def test_state_slice_negative_index(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pp.state_slice(base, -2).named("sl")
        df = sliced.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "C"

    def test_state_slice_empty_result_raises(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="slice produces 0 states"):
                pp.state_slice(base, slice(5, 10))

    def test_sample_with_replacement_allows_duplicates(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B"], mode="sequential")
            sampled = pp.sample(base, num_seqs=6, seed=42, with_replacement=True).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == 6
        assert set(df["seq"]) <= {"A", "B"}

    def test_sample_without_replacement_all_unique(self) -> None:
        with pp.Party():
            base = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = pp.sample(
                base, num_seqs=5, seed=42, with_replacement=False
            ).named("s")
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == 5
        assert len(df["seq"].unique()) == 5
