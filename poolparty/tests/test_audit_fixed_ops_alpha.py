"""Audit tests for fixed operations (alpha pass).

Scope:
- fixed_ops: rc, upper, lower, swapcase, join, slice_seq, stylize,
  clear_gaps, clear_annotation, add_prefix, score
- base op: filter
- base op: flip
- region op: remove_tags

This module encodes operation_audit.mdc Steps 2-4 runtime checks:
- mixin forwarding checks (runtime, not signature-only)
- core invariants (I1/I2 + trigger-based checks)
- adversarial diagonals for high-risk ops
- known-bug repros as strict xfails with runtime evidence
"""

import pytest

import poolparty as pp
from poolparty.utils.parsing_utils import strip_all_tags


def _clean_len(seq: str) -> int:
    return len(strip_all_tags(seq))


def _assert_i1_i2(pool, *, num_cycles=1, seed=42):
    """Assert I1 and I2 for a generated pool."""
    df = pool.generate_library(num_cycles=num_cycles, seed=seed)
    assert len(df) == pool.num_states
    if pool.seq_length is not None:
        for seq in df["seq"]:
            if isinstance(seq, str):
                assert _clean_len(seq) == pool.seq_length
    return df


class TestStep2RuntimeForwarding:
    """Step 2: mixin forwarding at runtime."""

    def test_runtime_forwarding_for_supported_mixin_methods(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CC</r>TT")
            dna = pp.from_seq("ACGT")

            # Generic fixed ops mixin
            base.slice_seq(
                region="r",
                start=0,
                stop=1,
                step=1,
                keep_context=False,
                iter_order=0,
                prefix="slice",
                style="red",
            )
            base.add_prefix(prefix="tag", iter_order=0)
            base.swapcase(region="r", remove_tags=False, iter_order=0, prefix="sw", style="red")
            base.upper(region="r", remove_tags=False, iter_order=0, prefix="up", style="blue")
            base.lower(region="r", remove_tags=False, iter_order=0, prefix="low", style="green")
            base.stylize(
                region="r",
                style="bold",
                which="contents",
                regex=None,
                iter_order=0,
                prefix="sty",
            )

            # Common ops mixin
            base.filter(predicate=lambda s: True, name="f", prefix="flt", cards=["passed"])
            base.score(fn=len, card_key="len", region="r", prefix="sc", cards=["len"])

            # DNA mixin
            dna.flip(
                region=None,
                rc_prob=0.5,
                prefix="flip",
                mode="random",
                num_states=2,
                iter_order=0,
                style="red",
                cards=["flip"],
            )
            dna.rc(region=None, remove_tags=None, iter_order=0, prefix="rc", style="blue")

            # Region mixin
            base.remove_tags(region_name="r", keep_content=True, iter_order=0, prefix="rt")

    def test_runtime_forwarding_clear_gaps(self):
        with pp.Party():
            pp.from_seq("AA--CC").clear_gaps(region=None, remove_tags=False, iter_order=0, prefix="cg")

    def test_runtime_forwarding_clear_annotation(self):
        with pp.Party():
            pp.from_seq("AA--cc").clear_annotation(
                region=None, remove_tags=False, iter_order=0, prefix="ca"
            )

    def test_join_is_factory_only_not_mixin(self):
        with pp.Party():
            a = pp.from_seq("A")
            b = pp.from_seq("T")
            out = pp.join([a, b])
            assert not hasattr(a, "join")
            assert out is not None


class TestStep3Invariants:
    """Step 3: core invariants across audited ops."""

    def test_i1_i2_rc_upper_lower_swapcase_add_prefix(self):
        with pp.Party():
            base = pp.from_seq("AA<r>CC</r>TT")
            _assert_i1_i2(pp.rc(base))
            _assert_i1_i2(pp.upper(base))
            _assert_i1_i2(pp.lower(base))
            _assert_i1_i2(pp.swapcase(base))
            _assert_i1_i2(pp.add_prefix(base, "pfx"))

    def test_i1_i2_join_slice_stylize_remove_tags(self):
        with pp.Party():
            left = pp.from_seqs(["AA", "TT"], mode="sequential")
            right = pp.from_seqs(["CC", "GG"], mode="sequential")
            _assert_i1_i2(pp.join([left, right], spacer_str="-"))

        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")
            _assert_i1_i2(pp.slice_seq(base, region="r", start=1, stop=3))
            _assert_i1_i2(pp.stylize(base, region="r", style="red"))
            _assert_i1_i2(pp.remove_tags(base, "r", keep_content=True))

    def test_i1_i2_score_and_filter(self):
        with pp.Party():
            base = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            scored = base.score(fn=len, cards=["score"])
            _assert_i1_i2(scored)

            # Use always-true predicate to avoid NullSeq and keep I1 simple.
            filtered = base.filter(predicate=lambda s: True, cards=["passed"])
            _assert_i1_i2(filtered)

    def test_i1_i2_and_i3_i5_for_flip(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CC</r>TT")
            seq_pool = pp.flip(base, region="r", mode="sequential", cards=["flip"])
            df = _assert_i1_i2(seq_pool)
            card_col = f"{seq_pool.operation.name}.flip"
            assert card_col in df.columns
            assert set(df[card_col]) == {"forward", "rc"}

        with pp.Party():
            base = pp.from_seq("ACGTAA")
            rand_pool = pp.flip(base, mode="random", num_states=8, cards=["flip"])
            df1 = _assert_i1_i2(rand_pool, seed=123)
            df2 = _assert_i1_i2(rand_pool, seed=123)
            assert list(df1["seq"]) == list(df2["seq"])

    def test_i6_region_only_modification_examples(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")

            # upper on region should preserve flanks
            up = pp.upper(base, region="r")
            seq = up.generate_library(num_cycles=1)["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:3] == "AAA"
            assert clean[-3:] == "TTT"

            # score on region should compute only region content
            sc = pp.score(base, fn=lambda s: len(s), region="r", cards=["score"])
            df = sc.generate_library(num_cycles=1)
            score_col = f"{sc.operation.name}.score"
            assert df[score_col].iloc[0] == 4


class TestStep4AdversarialJoin:
    """Step 4 adversarial diagonals for high-risk op: join."""

    def test_join_diagonal_1_known_lengths_multi_parent(self):
        with pp.Party():
            left = pp.from_seqs(["AA", "TT"], mode="sequential")
            right = pp.from_seqs(["C", "G"], mode="sequential")
            joined = pp.join([left, right], spacer_str="-")
            df = _assert_i1_i2(joined)
            assert len(df) == 4
            assert joined.seq_length == 4

    def test_join_diagonal_2_variable_parent_length(self):
        with pp.Party():
            left = pp.from_seqs(["AA", "TTT"], mode="sequential")  # seq_length=None
            joined = pp.join([left, "GG"], spacer_str="")
            df = _assert_i1_i2(joined)
            assert joined.seq_length is None
            assert sorted(_clean_len(s) for s in df["seq"]) == [4, 5]

    def test_join_diagonal_3_compositional_stress(self):
        with pp.Party():
            left = pp.from_seqs(["AA", "TT"], mode="sequential")
            mid = pp.join([left, "C"])
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 4


class TestStep4AdversarialSliceSeq:
    """Step 4 adversarial diagonals for high-risk op: slice_seq."""

    def test_slice_seq_diagonal_1_named_region(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")
            out = pp.slice_seq(base, region="r", start=1, stop=3)
            df = _assert_i1_i2(out)
            assert df["seq"].iloc[0] == "CC"

    def test_slice_seq_diagonal_2_interval_at_tag_boundary(self):
        with pp.Party():
            base = pp.from_seq("AA<r>CC</r>GG")
            out = pp.slice_seq(base, region=[2, 4], start=0, stop=2)
            df = _assert_i1_i2(out)
            assert df["seq"].iloc[0] == "CC"

    def test_slice_seq_diagonal_3_variable_parent_and_composition(self):
        with pp.Party():
            base = pp.from_seqs(["ACGT", "ACGTA"], mode="sequential")
            sliced = pp.slice_seq(base, start=1, stop=3)
            assert sliced.seq_length is None
            out = pp.flip(sliced, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 4

    def test_slice_seq_keep_context_named_region_seq_length_contract(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")
            out = pp.slice_seq(base, region="r", start=0, stop=2, keep_context=True)
            # Deterministic output is 'AAACCTTT' (len=8), so seq_length should be 8.
            assert out.seq_length == 8
            downstream = out.mutagenize(num_mutations=1, mode="sequential")
            df = downstream.generate_library(num_cycles=1)
            assert len(df) == downstream.num_states


class TestStep4AdversarialStylize:
    """Step 4 adversarial diagonals for high-risk op: stylize."""

    def test_stylize_diagonal_1_named_region_contents(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>cccc</r>TTT")
            out = pp.stylize(base, region="r", style="red", which="contents")
            _assert_i1_i2(out)

    def test_stylize_diagonal_2_interval_on_tagged_sequence(self):
        with pp.Party():
            base = pp.from_seq("AA<r>CC</r>GG")
            out = pp.stylize(base, region=[1, 5], style="blue", which="all")
            _assert_i1_i2(out)

    def test_stylize_diagonal_3_compositional_stress(self):
        with pp.Party():
            base = pp.from_seqs(["acgt", "tgca"], mode="sequential")
            mid = pp.stylize(base, style="cyan", which="lower")
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 4


class TestStep4AdversarialRemoveTags:
    """Step 4 adversarial diagonals for high-risk op: remove_tags."""

    def test_remove_tags_diagonal_1_keep_content(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")
            out = pp.remove_tags(base, "r", keep_content=True)
            df = _assert_i1_i2(out)
            assert df["seq"].iloc[0] == "AAACCCCTTT"
            assert out.seq_length == 10

    def test_remove_tags_diagonal_2_drop_content(self):
        with pp.Party():
            base = pp.from_seq("AAA<r>CCCC</r>TTT")
            out = pp.remove_tags(base, "r", keep_content=False)
            df = _assert_i1_i2(out)
            assert df["seq"].iloc[0] == "AAATTT"
            assert out.seq_length == 6

    def test_remove_tags_diagonal_3_compositional_stress(self):
        with pp.Party():
            base = pp.from_seqs(
                ["AA<r>CC</r>TT", "GG<r>AA</r>CC"],
                mode="sequential",
            )
            mid = pp.remove_tags(base, "r", keep_content=False)
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 4


class TestStep4AdversarialFlip:
    """Step 4 adversarial diagonals for high-risk op: flip."""

    def test_flip_diagonal_1_sequential_cycling(self):
        with pp.Party():
            out = pp.flip("ACGTAA", mode="sequential", num_states=3, cards=["flip"])
            df = _assert_i1_i2(out)
            card_col = f"{out.operation.name}.flip"
            assert list(df[card_col]) == ["forward", "rc", "forward"]

    def test_flip_diagonal_2_random_determinism(self):
        with pp.Party():
            out = pp.flip("ACGTAA", mode="random", rc_prob=0.4, num_states=6, cards=["flip"])
            df1 = _assert_i1_i2(out, seed=77)
            df2 = _assert_i1_i2(out, seed=77)
            assert list(df1["seq"]) == list(df2["seq"])

    def test_flip_diagonal_3_region_interval_and_composition(self):
        with pp.Party():
            base = pp.from_seqs(["AAAACGTAATTT", "TTTTACGTAAGG"], mode="sequential")
            flipped = pp.flip(base, region=[3, 9], mode="sequential", num_states=2)
            scored = pp.score(flipped, fn=len, cards=["score"])
            df = scored.generate_library(num_cycles=1)
            assert len(df) == scored.num_states == 4


class TestBravoDiscrepancyVerification:
    """Repro checks for alpha-vs-bravo report discrepancies."""

    def test_stylize_preserves_pool_type_for_protein_pool(self):
        with pp.Party():
            dna = pp.from_seq("ATGAAATTT")
            protein = dna.translate(frame=1)
            styled = protein.stylize(style="red")
            assert type(styled).__name__ == "ProteinPool"

    def test_add_prefix_preserves_pool_type_for_protein_pool(self):
        with pp.Party():
            dna = pp.from_seq("ATGAAATTT")
            protein = dna.translate(frame=1)
            prefixed = protein.add_prefix("x")
            assert type(prefixed).__name__ == "ProteinPool"

    def test_remove_tags_rejects_non_bool_keep_content(self):
        with pp.Party():
            bg = pp.from_seq("ACGT<region>TTAA</region>GCGC")
            with pytest.raises(Exception):
                pp.remove_tags(bg, "region", keep_content="yes")
