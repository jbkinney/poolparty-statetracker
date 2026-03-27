"""Audit tests for state operations (alpha pass, C7)."""

import inspect

import pytest

import poolparty as pp
from poolparty.utils.parsing_utils import strip_all_tags


def _bio_len(seq: str) -> int:
    return len(strip_all_tags(seq))


def _assert_exhaustion(pool, *, seed: int = 13) -> None:
    df = pool.generate_library(num_cycles=1, seed=seed)
    assert len(df) == pool.num_states


def test_signature_inventory_factories_and_beartype() -> None:
    factories = [pp.stack, pp.sample, pp.state_slice, pp.state_shuffle, pp.repeat, pp.sync]
    for fn in factories:
        assert hasattr(fn, "__wrapped__"), f"{fn.__name__} should be @beartype-wrapped"
        assert str(inspect.signature(fn))


def test_state_ops_mixin_surface_and_operator_aliases() -> None:
    with pp.Party():
        pool = pp.from_seqs(["AA", "BB"], mode="sequential")
        assert hasattr(pool, "repeat")
        assert hasattr(pool, "sample")
        assert hasattr(pool, "shuffle_states")
        assert hasattr(pool, "slice_states")
        assert not hasattr(pool, "sync")  # top-level in-place API

        # Operator aliases should route to state ops.
        assert isinstance((pool + pool).operation, pp.StackOp)
        assert isinstance((pool * 2).operation, pp.RepeatOp)
        assert isinstance((pool[0:1]).operation, pp.StateSliceOp)


def test_smoke_all_factories_construct_and_generate() -> None:
    with pp.Party():
        base = pp.from_seqs(["AA", "BB", "CC"], mode="sequential")
        other = pp.from_seqs(["XX", "YY", "ZZ"], mode="sequential")
        ops = [
            pp.stack([base, other]),
            pp.sample(base, seq_states=[0, 2]),
            pp.state_slice(base, slice(0, 2)),
            pp.state_shuffle(base, seed=5),
            pp.repeat(base, times=2),
        ]
        pp.sync([base, other])  # smoke for top-level in-place API

    for pool in ops:
        _assert_exhaustion(pool)


def test_i1_i2_stack_known_length() -> None:
    with pp.Party():
        a = pp.from_seqs(["AA", "BB"], mode="sequential")
        b = pp.from_seqs(["XX", "YY", "ZZ"], mode="sequential")
        stacked = pp.stack([a, b]).named("stacked")
        df = stacked.generate_library(num_cycles=1)

    assert stacked.seq_length == 2
    assert len(df) == stacked.num_states == 5
    for seq in df["seq"]:
        assert _bio_len(seq) == stacked.seq_length


def test_i1_stack_seq_length_none_is_justified_for_mixed_parents() -> None:
    with pp.Party():
        a = pp.from_seq("A")
        b = pp.from_seq("TT")
        stacked = pp.stack([a, b]).named("stacked")
        df = stacked.generate_library(num_cycles=1)

    assert stacked.seq_length is None
    assert len(df) == stacked.num_states == 2
    assert {_bio_len(seq) for seq in df["seq"]} == {1, 2}


def test_i1_i2_repeat_preserves_length_and_exhausts() -> None:
    with pp.Party():
        base = pp.from_seqs(["AA", "BB"], mode="sequential")
        repeated = pp.repeat(base, times=3).named("rep")
        df = repeated.generate_library(num_cycles=1)

    assert repeated.seq_length == 2
    assert len(df) == repeated.num_states == 6
    for seq in df["seq"]:
        assert _bio_len(seq) == repeated.seq_length


def test_sample_should_preserve_seq_length() -> None:
    with pp.Party():
        base = pp.from_seq("ACGT")
        sampled = pp.sample(base, seq_states=[0]).named("s")
    assert sampled.seq_length == 4


def test_state_slice_should_preserve_seq_length() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        sliced = pp.state_slice(base, slice(0, 1)).named("sl")
    assert sliced.seq_length == 4


def test_state_shuffle_should_preserve_seq_length() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        shuffled = pp.state_shuffle(base, seed=3).named("sh")
    assert shuffled.seq_length == 4


def test_i3_stack_card_sequence_agreement() -> None:
    with pp.Party():
        a = pp.from_seqs(["A", "B"], mode="sequential")
        b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
        stacked = pp.stack([a, b], cards=["active_parent"]).named("stacked")
        df = stacked.generate_library(num_cycles=1)

    card_col = [c for c in df.columns if c.endswith(".active_parent")][0]
    pairs = list(zip(df["seq"].tolist(), df[card_col].tolist()))
    assert pairs == [("A", 0), ("B", 0), ("X", 1), ("Y", 1), ("Z", 1)]


def test_i3_repeat_card_sequence_agreement() -> None:
    with pp.Party():
        base = pp.from_seqs(["A", "B"], mode="sequential")
        repeated = pp.repeat(base, times=2, cards=["repeat_index"]).named("rep")
        df = repeated.generate_library(num_cycles=1)

    card_col = [c for c in df.columns if c.endswith(".repeat_index")][0]
    assert list(df["seq"]) == ["A", "A", "B", "B"]
    assert list(df[card_col]) == [0, 1, 0, 1]


def test_i5_sample_determinism_with_seed() -> None:
    with pp.Party():
        p1 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
        s1 = pp.sample(p1, num_seqs=4, seed=42).named("s1")
    df1 = s1.generate_library(num_cycles=1)

    with pp.Party():
        p2 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
        s2 = pp.sample(p2, num_seqs=4, seed=42).named("s2")
    df2 = s2.generate_library(num_cycles=1)

    assert df1["seq"].tolist() == df2["seq"].tolist()


def test_i5_state_shuffle_determinism_with_seed() -> None:
    with pp.Party():
        p1 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
        sh1 = pp.state_shuffle(p1, seed=7).named("sh1")
    df1 = sh1.generate_library(num_cycles=1)

    with pp.Party():
        p2 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
        sh2 = pp.state_shuffle(p2, seed=7).named("sh2")
    df2 = sh2.generate_library(num_cycles=1)

    assert df1["seq"].tolist() == df2["seq"].tolist()


def test_i7_sample_composition_with_mutagenize_should_work() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        sampled = pp.sample(base, seq_states=[0, 1]).named("sampled")
        downstream = pp.mutagenize(sampled, num_mutations=1, mode="sequential").named("m")
        df = downstream.generate_library(num_cycles=1)
    assert len(df) == downstream.num_states


def test_i7_state_slice_composition_with_mutagenize_should_work() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        sliced = pp.state_slice(base, slice(0, 1)).named("sliced")
        downstream = pp.mutagenize(sliced, num_mutations=1, mode="sequential").named("m")
        df = downstream.generate_library(num_cycles=1)
    assert len(df) == downstream.num_states


def test_i7_state_shuffle_composition_with_mutagenize_should_work() -> None:
    with pp.Party():
        base = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
        shuffled = pp.state_shuffle(base, seed=2).named("shuffled")
        downstream = pp.mutagenize(shuffled, num_mutations=1, mode="sequential").named("m")
        df = downstream.generate_library(num_cycles=1)
    assert len(df) == downstream.num_states


def test_i10_state_space_immutability_for_stack() -> None:
    with pp.Party():
        a = pp.from_seqs(["A", "B"], mode="sequential")
        b = pp.from_seqs(["X", "Y"], mode="sequential")
        stacked = pp.stack([a, b]).named("stacked")
        ns_before = stacked.num_states
        sv_before = stacked.operation.state._num_values
        stacked.generate_library(num_cycles=1)

    assert stacked.num_states == ns_before
    assert stacked.operation.state._num_values == sv_before


def test_sync_fix_20_and_21_pairwise_three_pools_and_return_none() -> None:
    with pp.Party():
        a = pp.from_seqs(["A", "B", "C"], mode="sequential")
        b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
        c = pp.from_seqs(["1", "2", "3"], mode="sequential")
        ret = pp.sync([a, b, c])
        out = pp.join([a, b, c]).generate_library(num_cycles=1)["seq"].tolist()

    assert ret is None
    assert out == ["AX1", "BY2", "CZ3"]


def test_sync_fix_36_and_37_preserve_non_leaf_variation_and_lockstep() -> None:
    with pp.Party():
        # Non-leaf pools (fixed op over sequential source)
        left = pp.from_seqs(["AA", "BB"], mode="sequential").upper()
        right = pp.from_seqs(["xx", "yy"], mode="sequential").lower()

        # Before sync, join should be Cartesian (4 states).
        cart = pp.join([left, "-", right]).named("cart")
        assert cart.num_states == 4

        pp.sync([left, right])

        # After sync, join should be lockstep (2 states).
        paired = pp.join([left, "-", right]).named("paired")
        df = paired.generate_library(num_cycles=1)

    assert paired.num_states == 2
    assert df["seq"].tolist() == ["AA-xx", "BB-yy"]


def test_sync_fix_38_rejects_ancestor_descendant_pairs() -> None:
    with pp.Party():
        base = pp.from_seqs(["AC", "GT"], mode="sequential")
        derived = base.upper()
        with pytest.raises(ValueError, match="ancestor-descendant relationship"):
            pp.sync([base, derived])


def test_stack_adversarial_mixed_cardinality_exhaustion() -> None:
    with pp.Party():
        a = pp.from_seqs(["A", "B"], mode="sequential")
        b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
        c = pp.from_seqs(["1"], mode="sequential")
        stacked = pp.stack([a, b, c]).named("stacked")
        df = stacked.generate_library(num_cycles=1)

    assert stacked.num_states == 6
    assert len(df) == 6
    assert df["seq"].tolist() == ["A", "B", "X", "Y", "Z", "1"]


def test_stack_adversarial_variable_length_composition() -> None:
    with pp.Party():
        mixed = pp.stack([pp.from_seq("A"), pp.from_seq("TT")]).named("mixed")
        # Downstream random mode should still work when seq_length=None is justified.
        downstream = pp.mutagenize(mixed, num_mutations=1, mode="random", num_states=5).named("m")
        df = downstream.generate_library(num_cycles=1, seed=11)

    assert mixed.seq_length is None
    assert len(df) == downstream.num_states


def test_stack_empty_pools_raises() -> None:
    with pp.Party():
        with pytest.raises(ValueError, match="num_states must be >= 1"):
            pp.stack([])
