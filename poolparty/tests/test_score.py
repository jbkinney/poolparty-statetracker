"""Tests for the score operation."""

import re

import pytest

import poolparty as pp
from poolparty.fixed_ops.score import ScoreOp, score


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)


# ===========================================================================
# Basic behaviour
# ===========================================================================


class TestScoreBasic:

    def test_passthrough_no_region(self):
        """Sequence passes through unchanged."""
        seq = "ACGTACGT"
        with pp.Party():
            pool = score(seq, fn=len, cards=["score"]).named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 1
        assert df["seq"].iloc[0] == seq

    def test_fixed_mode_one_state(self):
        """Score always has exactly 1 state (fixed mode)."""
        with pp.Party():
            pool = score("AAAA", fn=len).named("s")
            assert pool.num_states == 1

    def test_seq_length_preserved(self):
        """seq_length flows through."""
        with pp.Party():
            root = pp.from_seq("AACCGGTT").named("root")
            scored = root.score(fn=len).named("s")
            assert scored.seq_length == 8

    def test_fn_receives_clean_seq(self):
        """fn receives tag-stripped content."""
        captured = []
        with pp.Party():
            root = pp.from_seq("<r>ACGT</r>TTTT").named("root")
            scored = root.score(fn=lambda s: captured.append(s) or 0, cards=["score"]).named("s")
            scored.generate_library(num_cycles=1)

        assert captured == ["ACGTTTTT"]


# ===========================================================================
# Design cards
# ===========================================================================


class TestScoreDesignCards:

    def test_card_recorded(self):
        """Score value appears in the design card column."""
        with pp.Party():
            pool = score("GCGCGCGC", fn=gc_content, cards=["score"]).named("s")
            df = pool.generate_library(num_cycles=1)

        score_cols = [c for c in df.columns if c.endswith(".score")]
        assert len(score_cols) == 1
        assert df[score_cols[0]].iloc[0] == pytest.approx(1.0)

    def test_card_not_present_without_opt_in(self):
        """No card column unless cards= is specified."""
        with pp.Party():
            pool = score("GCGCGCGC", fn=gc_content).named("s")
            df = pool.generate_library(num_cycles=1)

        score_cols = [c for c in df.columns if "score" in c.lower()]
        assert len(score_cols) == 0

    def test_custom_card_key(self):
        """Custom card_key appears in the column name."""
        with pp.Party():
            pool = score("ACGT", fn=len, card_key="length", cards=["length"]).named("s")
            df = pool.generate_library(num_cycles=1)

        length_cols = [c for c in df.columns if c.endswith(".length")]
        assert len(length_cols) == 1
        assert df[length_cols[0]].iloc[0] == 4

    def test_multiple_scores_different_keys(self):
        """Two score ops with different card_keys produce distinct columns."""
        with pp.Party():
            root = pp.from_seq("GCGCAAAA").named("root")
            s1 = root.score(fn=gc_content, card_key="gc", cards=["gc"]).named("gc_score")
            s2 = s1.score(fn=len, card_key="length", cards=["length"]).named("len_score")
            df = s2.generate_library(num_cycles=1)

        gc_cols = [c for c in df.columns if c.endswith(".gc")]
        len_cols = [c for c in df.columns if c.endswith(".length")]
        assert len(gc_cols) == 1
        assert len(len_cols) == 1
        assert df[gc_cols[0]].iloc[0] == pytest.approx(0.5)
        assert df[len_cols[0]].iloc[0] == 8


# ===========================================================================
# Region — no tags in sequence
# ===========================================================================


class TestScoreRegionNoTags:

    def test_interval_region_no_tags(self):
        """Interval region [2, 6] on a plain sequence scores the sub-string."""
        seq = "AAGCGCTT"
        with pp.Party():
            pool = score(seq, fn=gc_content, region=[2, 6], cards=["score"]).named("s")
            df = pool.generate_library(num_cycles=1)

        assert df["seq"].iloc[0] == seq
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        # [2, 6) exclusive stop → positions 2,3,4,5 → "GCGC"
        expected_gc = gc_content("GCGC")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)

    def test_named_region_no_tags(self):
        """Named region on a plain (annotated) sequence scores the region content."""
        seq = "AAGCGCTT"
        with pp.Party():
            root = pp.from_seq(seq).named("root")
            tagged = root.annotate_region("mid", extent=(2, 6)).named("tagged")
            scored = tagged.score(fn=gc_content, region="mid", cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        assert strip_tags(df["seq"].iloc[0]) == seq
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        # extent=(2, 6) → "GCGC"
        expected_gc = gc_content("GCGC")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)


# ===========================================================================
# Region — sequence with one tag pair
# ===========================================================================


class TestScoreRegionOneTag:

    def test_interval_region_one_tag(self):
        """Interval region on a sequence containing one tag pair."""
        raw = "<r>ACGT</r>TTTTGGGG"
        with pp.Party():
            root = pp.from_seq(raw).named("root")
            # [0, 4) → clean positions 0,1,2,3 → "ACGT"
            scored = root.score(fn=gc_content, region=[0, 4], cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        clean = strip_tags(df["seq"].iloc[0])
        assert clean == "ACGTTTTTGGGG"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        expected_gc = gc_content("ACGT")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)

    def test_named_region_one_tag(self):
        """Named region on a sequence containing one tag pair — region fully after the tag."""
        raw = "<r>ACGT</r>TTTTGGGG"
        with pp.Party():
            root = pp.from_seq(raw).named("root")
            # extent=(4, 8) covers clean positions 4..7 → "TTTT" (fully after </r>)
            tagged = root.annotate_region("tail", extent=(4, 8)).named("tagged")
            scored = tagged.score(fn=gc_content, region="tail", cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        clean = strip_tags(df["seq"].iloc[0])
        assert clean == "ACGTTTTTGGGG"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        expected_gc = gc_content("TTTT")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)


# ===========================================================================
# Region — sequence with multiple tag pairs
# ===========================================================================


class TestScoreRegionMultipleTags:

    def test_interval_region_multi_tags(self):
        """Interval region on a sequence with multiple tags."""
        raw = "<a>AA</a><b>GG</b>CCCC<c>TT</c>"
        with pp.Party():
            root = pp.from_seq(raw).named("root")
            # [2, 6) → clean positions 2,3,4,5 → "GGCC" (from clean "AAGGCCCCTT")
            scored = root.score(fn=gc_content, region=[2, 6], cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        clean = strip_tags(df["seq"].iloc[0])
        assert clean == "AAGGCCCCTT"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        expected_gc = gc_content("GGCC")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)

    def test_named_region_multi_tags(self):
        """Named region nested inside existing tags — avoids cross-tag boundary."""
        raw = "<a>AAGCATGCTT</a><b>CCGG</b>"
        with pp.Party():
            root = pp.from_seq(raw).named("root")
            # extent=(2, 6) → clean positions 2..5 → "GCAT" (nested inside <a>)
            tagged = root.annotate_region("core", extent=(2, 6)).named("tagged")
            scored = tagged.score(fn=gc_content, region="core", cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        clean = strip_tags(df["seq"].iloc[0])
        assert clean == "AAGCATGCTTCCGG"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        expected_gc = gc_content("GCAT")
        assert df[score_col].iloc[0] == pytest.approx(expected_gc)


# ===========================================================================
# Mixin interface
# ===========================================================================


class TestScoreMixin:

    def test_mixin_method(self):
        """Pool.score() mixin works identically to factory function."""
        with pp.Party():
            root = pp.from_seq("GCGCAAAA").named("root")
            scored = root.score(fn=gc_content, cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        assert df["seq"].iloc[0] == "GCGCAAAA"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        assert df[score_col].iloc[0] == pytest.approx(0.5)

    def test_mixin_with_region(self):
        """Pool.score() mixin works with region specification."""
        with pp.Party():
            root = pp.from_seq("AAAAGCGC").named("root")
            # [4, 8) → clean positions 4,5,6,7 → "GCGC"
            scored = root.score(fn=gc_content, region=[4, 8], cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        assert df["seq"].iloc[0] == "AAAAGCGC"
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        assert df[score_col].iloc[0] == pytest.approx(1.0)


# ===========================================================================
# Chaining & compatibility
# ===========================================================================


class TestScoreChaining:

    def test_score_after_mutagenize(self):
        """Score works downstream of mutagenize."""
        with pp.Party():
            root = pp.from_seq("AAAA").named("root")
            mut = root.mutagenize(mutation_rate=0.5, num_states=3).named("mut")
            scored = mut.score(fn=gc_content, cards=["score"]).named("s")
            df = scored.generate_library(num_cycles=1)

        assert len(df) == 3
        score_col = [c for c in df.columns if c.endswith(".score")][0]
        assert all(0.0 <= v <= 1.0 for v in df[score_col])

    def test_filter_after_score(self):
        """Filter can use score results (independent chain)."""
        with pp.Party():
            root = pp.from_seqs(["GCGCGCGC", "AAAAAAAA", "CCCCCCCC"], mode="sequential")
            scored = root.score(fn=gc_content, cards=["score"]).named("s")
            filtered = scored.filter(lambda s: gc_content(s) > 0.5).named("f")
            df = filtered.generate_library(num_cycles=1, discard_null_seqs=True)

        assert len(df) == 2
        seqs = set(df["seq"])
        assert "GCGCGCGC" in seqs
        assert "CCCCCCCC" in seqs


# ===========================================================================
# Error / edge cases
# ===========================================================================


class TestScoreEdgeCases:

    def test_fn_exception_propagates(self):
        """Exceptions from fn propagate to the caller."""
        def bad_fn(s):
            raise ValueError("bad value")

        with pp.Party():
            pool = score("ACGT", fn=bad_fn, cards=["score"]).named("s")
            with pytest.raises(ValueError, match="bad value"):
                pool.generate_library(num_cycles=1)

    def test_fn_returns_string(self):
        """fn can return any type, including string."""
        with pp.Party():
            pool = score("ACGT", fn=lambda s: "high", card_key="label", cards=["label"]).named("s")
            df = pool.generate_library(num_cycles=1)

        label_col = [c for c in df.columns if c.endswith(".label")][0]
        assert df[label_col].iloc[0] == "high"

    def test_fn_returns_none(self):
        """fn can return None."""
        with pp.Party():
            pool = score("ACGT", fn=lambda s: None, cards=["score"]).named("s")
            df = pool.generate_library(num_cycles=1)

        score_col = [c for c in df.columns if c.endswith(".score")][0]
        assert df[score_col].iloc[0] is None

    def test_factory_name(self):
        """ScoreOp has the expected factory_name."""
        with pp.Party():
            pool = score("ACGT", fn=len).named("s")
            assert pool.operation.factory_name == "score"

    def test_op_type(self):
        """Factory function creates a ScoreOp."""
        with pp.Party():
            pool = score("ACGT", fn=len).named("s")
            assert isinstance(pool.operation, ScoreOp)
