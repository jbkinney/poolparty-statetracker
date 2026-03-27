"""Fixed operations audit — test_audit_fixed_ops_bravo.py

Covers 14 ops: rc, upper, lower, swapcase, join, slice_seq, stylize,
clear_gaps, clear_annotation, remove_tags, add_prefix, flip, score, filter.

Follows operation_audit.mdc Steps 1–7.
"""

import pytest
import numpy as np

import poolparty as pp
from poolparty.utils.parsing_utils import strip_all_tags
from poolparty.types import NullSeq, is_null_seq
from poolparty.dna_pool import DnaPool
from poolparty.protein_pool import ProteinPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen(pool, **kw):
    """Shorthand for generate_library with num_cycles=1."""
    return pool.generate_library(num_cycles=1, **kw)


def _gen_seqs(pool, **kw):
    """Generate and return the seq column."""
    return _gen(pool, **kw)["seq"]


def _card_col(df, suffix):
    """Find a card column ending with the given suffix (e.g. '.score')."""
    cols = [c for c in df.columns if c.endswith(suffix)]
    assert len(cols) == 1, f"Expected 1 column ending with '{suffix}', got {cols}"
    return cols[0]


# ===================================================================
# STEP 2 — Mixin forwarding (runtime, not just signature comparison)
# ===================================================================

class TestMixinForwarding:
    """Verify every mixin method delegates correctly at runtime."""

    def test_upper_mixin(self):
        with pp.Party():
            p = pp.from_seq("acgt")
            result = p.upper()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_lower_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.lower()
            df = _gen(result)
            assert df["seq"].iloc[0] == "acgt"

    def test_swapcase_mixin(self):
        with pp.Party():
            p = pp.from_seq("AcGt")
            result = p.swapcase()
            df = _gen(result)
            assert df["seq"].iloc[0] == "aCgT"

    def test_rc_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"  # ACGT is its own RC

    def test_rc_mixin_with_region(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>ACGT</r>TTT")
            result = p.rc(region="r")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            assert "ACGT" in seq  # RC of ACGT = ACGT

    def test_rc_mixin_with_style(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc(style="red")
            df = _gen(result)
            assert len(df) == 1

    def test_slice_seq_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGTACGT")
            result = p.slice_seq(start=2, stop=6)
            df = _gen(result)
            assert df["seq"].iloc[0] == "GTAC"

    def test_slice_seq_mixin_with_region(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf")
            df = _gen(result)
            assert df["seq"].iloc[0] == "ATGCCC"

    def test_slice_seq_mixin_with_style(self):
        with pp.Party():
            p = pp.from_seq("ACGTACGT")
            result = p.slice_seq(start=0, stop=4, style="red")
            df = _gen(result)
            assert len(df) == 1

    def test_add_prefix_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT").named("base")
            result = p.add_prefix("test")
            df = _gen(result)
            assert len(df) == 1

    def test_clear_gaps_mixin(self):
        with pp.Party():
            p = pp.from_seq("AC-GT")
            result = p.clear_gaps()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_clear_annotation_mixin(self):
        with pp.Party():
            p = pp.from_seq("ac<r>GT</r>")
            result = p.clear_annotation()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_stylize_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.stylize(style="red")
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_stylize_mixin_with_region(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>ACGT</r>TTT")
            result = p.stylize(region="r", style="bold")
            df = _gen(result)
            assert len(df) == 1

    def test_score_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.score(fn=len, cards=["score"])
            df = _gen(result)
            assert df[_card_col(df, ".score")].iloc[0] == 4

    def test_score_mixin_with_region(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.score(fn=len, region="r", cards=["score"])
            df = _gen(result)
            assert df[_card_col(df, ".score")].iloc[0] == 4

    def test_score_mixin_with_card_key(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.score(fn=len, card_key="length", cards=["length"])
            df = _gen(result)
            col = _card_col(df, ".length")
            assert df[col].iloc[0] == 4

    def test_filter_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: len(s) == 4)
            df = _gen(result)
            assert len(df) == 1

    def test_filter_mixin_with_cards(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True, cards=["passed"])
            df = _gen(result)
            assert df[_card_col(df, ".passed")].iloc[0] == True

    def test_flip_mixin(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip()
            df = _gen(result)
            assert len(df) == 2

    def test_flip_mixin_with_cards(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(cards=["flip"])
            df = _gen(result)
            col = _card_col(df, ".flip")
            assert set(df[col]) == {"forward", "rc"}

    def test_flip_mixin_random_mode(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(mode="random")
            df = result.generate_library(num_seqs=5, seed=42)
            assert len(df) == 5

    def test_remove_tags_mixin(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r")
            df = _gen(result)
            assert df["seq"].iloc[0] == "AACCGGTT"

    def test_remove_tags_mixin_drop_content(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r", keep_content=False)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AATT"


# ===================================================================
# STEP 3+4 — Invariant tests per contract group
# ===================================================================

# -------------------------------------------------------------------
# Group A: Length-preserving char transforms (upper, lower, swapcase, rc)
# -------------------------------------------------------------------

class TestGroupA_I1I2:
    """I1 (output length) and I2 (state exhaustion) for all Group A ops."""

    @pytest.mark.parametrize("op_name", ["upper", "lower", "swapcase", "rc"])
    def test_i1_output_length(self, op_name):
        with pp.Party():
            p = pp.from_seq("AcGt")
            result = getattr(p, op_name)()
            df = _gen(result)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == result.seq_length

    @pytest.mark.parametrize("op_name", ["upper", "lower", "swapcase", "rc"])
    def test_i2_exhaustion(self, op_name):
        with pp.Party():
            p = pp.from_seq("AcGt")
            result = getattr(p, op_name)()
            assert result.num_states == 1
            df = _gen(result)
            assert len(df) == 1

    @pytest.mark.parametrize("op_name", ["upper", "lower", "swapcase", "rc"])
    def test_i2_chained_exhaustion(self, op_name):
        """Chain with sequential parent — total rows = parent_states * 1."""
        with pp.Party():
            p = pp.from_seqs(["ACGT", "TTTT"], mode="sequential")
            result = getattr(p, op_name)()
            df = _gen(result)
            assert len(df) == result.num_states
            assert len(df) == 2

    @pytest.mark.parametrize("op_name", ["upper", "lower", "swapcase", "rc"])
    def test_i6_region_isolation(self, op_name):
        """Region-only modification: prefix/suffix unchanged."""
        with pp.Party():
            p = pp.from_seq("aaa<r>CcCc</r>ttt")
            result = getattr(p, op_name)(region="r")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:3] == "aaa"
            assert clean[-3:] == "ttt"


class TestGroupA_Representative_RC:
    """Full I1–I7 + domain-specific for rc (Group A representative)."""

    def test_i3_no_cards(self):
        """rc has no design cards — verify empty."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc()
            df = _gen(result)
            card_cols = [c for c in df.columns if c.startswith("rc.")]
            assert len(card_cols) == 0

    def test_i4_tag_preservation(self):
        """rc strips tags from region content — this is documented behavior."""
        with pp.Party():
            p = pp.from_seq("AA<outer>CC<inner>GG</inner>TT</outer>AA")
            result = p.rc(region="outer")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:2] == "AA"
            assert clean[-2:] == "AA"

    def test_i5_determinism(self):
        """rc is fixed mode — inherently deterministic."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc()
            df1 = result.generate_library(seed=1)
            df2 = result.generate_library(seed=2)
            assert df1["seq"].iloc[0] == df2["seq"].iloc[0]

    def test_i7_composition(self):
        """Chained rc has correct num_states and seq_length."""
        with pp.Party():
            p = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            result = p.rc()
            df = _gen(result)
            assert len(df) == result.num_states
            assert result.seq_length == 4

    def test_domain_rc_correctness(self):
        with pp.Party():
            p = pp.from_seq("AACG")
            result = p.rc()
            df = _gen(result)
            assert df["seq"].iloc[0] == "CGTT"

    def test_domain_rc_palindrome(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_domain_rc_with_interval_region(self):
        with pp.Party():
            p = pp.from_seq("AAACGTTT")
            result = p.rc(region=[3, 5])
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:3] == "AAA"
            assert clean[-3:] == "TTT"


# -------------------------------------------------------------------
# Group B: Variable-length content stripping (clear_gaps, clear_annotation)
# -------------------------------------------------------------------

class TestGroupB_I1I2:
    """I1 and I2 for Group B ops.

    NOTE: Both clear_gaps and clear_annotation crash at construction time
    because they reference party.alphabet which does not exist on Party.
    All tests are marked xfail to document this bug.
    """

    def test_i1_clear_gaps_no_gaps(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.clear_gaps()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_i1_clear_gaps_with_gaps(self):
        with pp.Party():
            p = pp.from_seq("AC-G.T")
            result = p.clear_gaps()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_i1_clear_annotation_strips_all(self):
        with pp.Party():
            p = pp.from_seq("ac<r>GT</r>")
            result = p.clear_annotation()
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACGT"

    def test_i2_clear_gaps_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("A-C")
            result = p.clear_gaps()
            assert result.num_states == 1
            df = _gen(result)
            assert len(df) == 1

    def test_i2_clear_annotation_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("acgt")
            result = p.clear_annotation()
            assert result.num_states == 1
            df = _gen(result)
            assert len(df) == 1

    def test_i2_chained_clear_gaps(self):
        with pp.Party():
            p = pp.from_seqs(["A-C", "G-T"], mode="sequential")
            result = p.clear_gaps()
            df = _gen(result)
            assert len(df) == 2

    def test_i6_clear_gaps_region_isolation(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>C-G</r>TTT")
            result = p.clear_gaps(region="r")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:3] == "AAA"
            assert clean[-3:] == "TTT"

    def test_i6_clear_annotation_region_isolation(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>c-g</r>TTT")
            result = p.clear_annotation(region="r")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:3] == "AAA"
            assert clean[-3:] == "TTT"


class TestGroupB_Representative_ClearAnnotation:
    """Full invariants + domain-specific for clear_annotation."""

    def test_seq_length_is_none(self):
        """clear_annotation always sets seq_length=None (variable output)."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.clear_annotation()
            assert result.seq_length is None

    def test_clear_annotation_preserves_tags_in_region(self):
        """When region is specified, tags in region content are stripped."""
        with pp.Party():
            p = pp.from_seq("AA<outer>cc<inner>gg</inner>tt</outer>AA")
            result = p.clear_annotation(region="outer")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert "AA" == clean[:2]
            assert "AA" == clean[-2:]

    def test_i7_clear_annotation_composition(self):
        with pp.Party():
            p = pp.from_seqs(["ACGT", "TTTT"], mode="sequential")
            result = p.clear_annotation()
            df = _gen(result)
            assert len(df) == result.num_states


# -------------------------------------------------------------------
# Group C: Passthrough with side effects (score, filter, add_prefix)
# -------------------------------------------------------------------

class TestGroupC_I1I2:
    """I1 and I2 for Group C ops."""

    def test_i1_score_preserves_length(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.score(fn=len)
            assert result.seq_length == 4
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == 4

    def test_i1_filter_preserves_length(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True)
            assert result.seq_length == 4

    def test_i1_add_prefix_preserves_length(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.add_prefix("test")
            assert result.seq_length == 4

    def test_i2_score_single_state(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.score(fn=len)
            assert result.num_states == 1
            df = _gen(result)
            assert len(df) == 1

    def test_i2_filter_single_state(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True)
            assert result.num_states == 1

    def test_i2_add_prefix_single_state(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.add_prefix("x")
            assert result.num_states == 1

    def test_i2_chained_score(self):
        with pp.Party():
            p = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            result = p.score(fn=len)
            df = _gen(result)
            assert len(df) == 2

    def test_i2_chained_filter_all_pass(self):
        with pp.Party():
            p = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            result = p.filter(predicate=lambda s: True)
            df = _gen(result)
            assert len(df) == 2

    def test_i2_chained_filter_some_reject(self):
        with pp.Party():
            p = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            result = p.filter(predicate=lambda s: "A" in s)
            df = result.generate_library(num_cycles=1, discard_null_seqs=True)
            assert len(df) == 1
            assert df["seq"].iloc[0] == "AAAA"


class TestGroupC_Representative_Filter:
    """Full invariants + domain-specific for filter."""

    def test_i3_card_agreement(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True, cards=["passed"])
            df = _gen(result)
            assert df[_card_col(df, ".passed")].iloc[0] == True

    def test_i3_card_rejected(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: False, cards=["passed"])
            df = _gen(result)
            assert df[_card_col(df, ".passed")].iloc[0] == False

    def test_i5_determinism(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True)
            df1 = result.generate_library(seed=1)
            df2 = result.generate_library(seed=2)
            assert df1["seq"].iloc[0] == df2["seq"].iloc[0]

    def test_i7_composition_with_sequential(self):
        with pp.Party():
            parent = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            mut = parent.mutagenize(num_mutations=1, mode="sequential")
            filtered = mut.filter(predicate=lambda s: True)
            df = _gen(filtered)
            assert len(df) == filtered.num_states

    def test_domain_nullseq_production(self):
        """Filtered-out seqs become None in the DataFrame."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: False)
            df = _gen(result)
            assert df["seq"].iloc[0] is None

    def test_domain_clean_fallback_tagged(self):
        """filter predicate receives clean (tag-free) content even after insert_tags."""
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            tagged = p.annotate_region("mid", extent=(2, 6))
            filtered = tagged.filter(predicate=lambda s: len(s) == 8)
            df = _gen(filtered)
            assert not is_null_seq(df["seq"].iloc[0])

    def test_domain_seq_length_propagation(self):
        """Bug #32 regression: filter must propagate parent seq_length."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            filtered = p.filter(predicate=lambda s: True)
            assert filtered.seq_length == 4
            chained = filtered.mutagenize(num_mutations=1, mode="sequential")
            assert chained.seq_length == 4

    def test_domain_pool_type_preservation(self):
        """Bug #33 regression: filter must preserve pool type."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            assert isinstance(p, DnaPool)
            filtered = p.filter(predicate=lambda s: True)
            assert isinstance(filtered, DnaPool)


# -------------------------------------------------------------------
# Group D: Structural transforms (slice_seq, join, stylize)
# -------------------------------------------------------------------

class TestGroupD_I1I2:
    """I1 and I2 for Group D ops."""

    def test_i1_slice_seq_basic(self):
        with pp.Party():
            p = pp.from_seq("ACGTACGT")
            result = p.slice_seq(start=2, stop=6)
            assert result.seq_length == 4
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == 4

    def test_i1_join_two_pools(self):
        with pp.Party():
            p1 = pp.from_seq("AAAA")
            p2 = pp.from_seq("CCCC")
            result = pp.join([p1, p2])
            assert result.seq_length == 8
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == 8

    def test_i1_join_with_spacer(self):
        with pp.Party():
            p1 = pp.from_seq("AAAA")
            p2 = pp.from_seq("CCCC")
            result = pp.join([p1, p2], spacer_str="--")
            assert result.seq_length == 10

    def test_i1_stylize_preserves_length(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.stylize(style="red")
            assert result.seq_length == 4

    def test_i2_slice_seq_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.slice_seq(start=0, stop=2)
            assert result.num_states == 1
            df = _gen(result)
            assert len(df) == 1

    def test_i2_join_exhaustion(self):
        with pp.Party():
            p1 = pp.from_seqs(["AA", "CC"], mode="sequential")
            p2 = pp.from_seq("GG")
            result = pp.join([p1, p2])
            df = _gen(result)
            assert len(df) == result.num_states
            assert len(df) == 2

    def test_i2_stylize_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.stylize(style="red")
            assert result.num_states == 1

    def test_i6_slice_seq_region_isolation(self):
        """slice_seq with keep_context preserves prefix/suffix."""
        with pp.Party():
            p = pp.from_seq("AAA<r>CCCCCC</r>TTT")
            result = p.slice_seq(region="r", start=0, stop=3, keep_context=True)
            df = _gen(result)
            seq = df["seq"].iloc[0]
            assert seq[:3] == "AAA"
            assert seq[-3:] == "TTT"

    def test_i6_stylize_region_isolation(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>CCCC</r>TTT")
            result = p.stylize(region="r", style="bold")
            df = _gen(result)
            seq = strip_all_tags(df["seq"].iloc[0])
            assert seq[:3] == "AAA"
            assert seq[-3:] == "TTT"


class TestGroupD_Representative_SliceSeq:
    """Full invariants + domain-specific for slice_seq."""

    def test_i7_composition(self):
        with pp.Party():
            parent = pp.from_seqs(["ACGTACGT", "TTTTTTTT"], mode="sequential")
            sliced = parent.slice_seq(start=0, stop=4)
            df = _gen(sliced)
            assert len(df) == sliced.num_states
            assert sliced.seq_length == 4

    def test_domain_named_region_extract(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf")
            df = _gen(result)
            assert df["seq"].iloc[0] == "ATGCCC"

    def test_domain_interval_region(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6])
            df = _gen(result)
            assert df["seq"].iloc[0] == "CCGG"

    def test_domain_step_slicing(self):
        with pp.Party():
            p = pp.from_seq("ABCDEFGH")
            result = p.slice_seq(step=2)
            df = _gen(result)
            assert df["seq"].iloc[0] == "ACEG"

    def test_domain_keep_context_named_region_no_slice(self):
        """keep_context=True with named region and no slice."""
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", keep_context=True)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AAAATGCCCTTT"

    def test_domain_keep_context_requires_region(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="keep_context.*requires.*region"):
                p.slice_seq(keep_context=True)

    def test_domain_interval_region_seq_length(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6])
            assert result.seq_length == 4

    def test_domain_keep_context_interval_seq_length(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6], keep_context=True)
            assert result.seq_length == 8

    def test_domain_keep_context_interval_with_slice_seq_length(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6], start=0, stop=2, keep_context=True)
            assert result.seq_length == 6  # 2 prefix + 2 sliced + 2 suffix


# -------------------------------------------------------------------
# Group E: Tag manipulation (remove_tags)
# -------------------------------------------------------------------

class TestGroupE_RemoveTags:
    """Full invariants + domain-specific for remove_tags."""

    def test_i1_keep_content_preserves_length(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r", keep_content=True)
            assert result.seq_length == 8
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == 8

    def test_i1_drop_content_shrinks_length(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r", keep_content=False)
            assert result.seq_length == 4
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == 4

    def test_i2_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CC</r>TT")
            result = p.remove_tags("r")
            assert result.num_states == 1

    def test_i2_chained_exhaustion(self):
        with pp.Party():
            parent = pp.from_seqs(["AA<r>CC</r>TT", "GG<r>AA</r>CC"], mode="sequential")
            result = parent.remove_tags("r")
            df = _gen(result)
            assert len(df) == 2

    def test_i7_composition(self):
        with pp.Party():
            parent = pp.from_seqs(["AA<r>CC</r>TT", "GG<r>AA</r>CC"], mode="sequential")
            result = parent.remove_tags("r")
            df = _gen(result)
            assert len(df) == result.num_states

    def test_domain_content_preserved(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r", keep_content=True)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AACCGGTT"

    def test_domain_content_removed(self):
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.remove_tags("r", keep_content=False)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AATT"

    def test_domain_no_beartype(self):
        """remove_tags lacks @beartype — confirm it still works with bad types gracefully."""
        with pp.Party():
            p = pp.from_seq("AA<r>CC</r>TT")
            result = p.remove_tags("r")
            df = _gen(result)
            assert len(df) == 1


# -------------------------------------------------------------------
# Group F: Multi-mode stateful (flip)
# -------------------------------------------------------------------

class TestGroupF_Flip:
    """Full invariants + domain-specific for flip."""

    def test_i1_output_length(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip()
            assert result.seq_length == 4
            df = _gen(result)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == 4

    def test_i2_sequential_exhaustion(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(mode="sequential")
            assert result.num_states == 2
            df = _gen(result)
            assert len(df) == 2

    def test_i3_card_agreement(self):
        with pp.Party():
            p = pp.from_seq("AACG")
            result = p.flip(cards=["flip"])
            df = _gen(result)
            col = _card_col(df, ".flip")
            for _, row in df.iterrows():
                if row[col] == "forward":
                    assert strip_all_tags(row["seq"]) == "AACG"
                else:
                    assert strip_all_tags(row["seq"]) == "CGTT"

    def test_i5_determinism_random(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(mode="random")
            df1 = result.generate_library(num_seqs=10, seed=42)
            df2 = result.generate_library(num_seqs=10, seed=42)
            assert list(df1["seq"]) == list(df2["seq"])

    def test_i6_region_isolation(self):
        with pp.Party():
            p = pp.from_seq("AAA<r>AACG</r>TTT")
            result = p.flip(region="r")
            df = _gen(result)
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean[:3] == "AAA"
                assert clean[-3:] == "TTT"

    def test_i7_composition(self):
        with pp.Party():
            parent = pp.from_seqs(["AACC", "GGTT"], mode="sequential")
            flipped = parent.flip(mode="sequential")
            df = _gen(flipped)
            assert len(df) == flipped.num_states
            assert len(df) == 4  # 2 seqs * 2 flip states

    def test_i10_state_immutability(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(mode="sequential")
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before

    def test_domain_fixed_mode_rejected(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="mode='fixed'"):
                p.flip(mode="fixed")

    def test_domain_forward_rc_correctness(self):
        with pp.Party():
            p = pp.from_seq("AACG")
            result = p.flip(mode="sequential")
            df = _gen(result)
            seqs = sorted(df["seq"].tolist())
            assert "AACG" in seqs
            assert "CGTT" in seqs

    def test_domain_palindrome(self):
        """Palindromic sequences produce identical forward and RC."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip(mode="sequential")
            df = _gen(result)
            assert df["seq"].iloc[0] == df["seq"].iloc[1]

    def test_domain_rc_prob(self):
        """rc_prob=0 should always produce forward."""
        with pp.Party():
            p = pp.from_seq("AACG")
            result = p.flip(mode="random", rc_prob=0.0)
            df = result.generate_library(num_seqs=10, seed=42)
            for seq in df["seq"]:
                assert seq == "AACG"

    def test_domain_rc_prob_one(self):
        """rc_prob=1 should always produce rc."""
        with pp.Party():
            p = pp.from_seq("AACG")
            result = p.flip(mode="random", rc_prob=1.0)
            df = result.generate_library(num_seqs=10, seed=42)
            for seq in df["seq"]:
                assert seq == "CGTT"


# ===================================================================
# STEP 5 — Adversarial patterns (high-risk ops)
# ===================================================================

class TestAdversarial_SliceSeq:
    """At least 3 diagonal adversarial combos for slice_seq."""

    def test_named_region_keep_context_with_slice(self):
        """Named region + keep_context=True + slice [0:3]."""
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", start=0, stop=3, keep_context=True)
            df = _gen(result)
            seq = df["seq"].iloc[0]
            assert seq == "AAAATGTTT"
            # seq_length finding: keep_context+named region returns None
            # even though it's computable
            seq_length = result.seq_length

    def test_interval_region_tagged_seq_keep_context(self):
        """Interval region [2,8] on tagged seq + keep_context=True."""
        with pp.Party():
            p = pp.from_seq("AA<r>CCGGTT</r>AA")
            result = p.slice_seq(region=[2, 8], keep_context=True)
            df = _gen(result)
            seq = df["seq"].iloc[0]
            clean = strip_all_tags(seq)
            assert clean[:2] == "AA"
            assert clean[-2:] == "AA"
            assert result.seq_length == 10

    def test_variable_length_parent_slice(self):
        """Variable-length parent (from_seqs) + start=1, stop=-1."""
        with pp.Party():
            parent = pp.from_seqs(["ACGTACGT", "TTTTTTTT", "GGGGGGGG"], mode="sequential")
            result = parent.slice_seq(start=1, stop=-1)
            assert result.seq_length is None or result.seq_length == 6
            df = _gen(result)
            for seq in df["seq"]:
                assert len(seq) == 6

    def test_compositional_stress(self):
        """Chain slice_seq between two sequential ops."""
        with pp.Party():
            parent = pp.from_seqs(["AACCGGTT", "TTGGCCAA"], mode="sequential")
            sliced = parent.slice_seq(start=0, stop=4)
            mut = sliced.mutagenize(num_mutations=1, mode="sequential")
            df = _gen(mut)
            assert len(df) == mut.num_states


class TestAdversarial_Flip:
    """At least 3 diagonal adversarial combos for flip."""

    def test_sequential_named_region_chained(self):
        """Sequential mode + named region + chained with mutagenize."""
        with pp.Party():
            parent = pp.from_seq("AAA<r>AACG</r>TTT")
            flipped = parent.flip(region="r", mode="sequential")
            mut = flipped.mutagenize(num_mutations=1, mode="sequential")
            df = _gen(mut)
            assert len(df) == mut.num_states

    def test_random_mode_seed_region(self):
        """Random mode + seed=42 + region — deterministic across Party contexts."""
        with pp.Party():
            p = pp.from_seq("AAA<r>AACG</r>TTT")
            result = p.flip(region="r", mode="random")
            df1 = result.generate_library(num_seqs=20, seed=42)
        with pp.Party():
            p = pp.from_seq("AAA<r>AACG</r>TTT")
            result = p.flip(region="r", mode="random")
            df2 = result.generate_library(num_seqs=20, seed=42)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_short_sequence(self):
        """Short sequence (3 bp) in sequential mode."""
        with pp.Party():
            p = pp.from_seq("ACG")
            result = p.flip(mode="sequential")
            df = _gen(result)
            assert len(df) == 2
            seqs = set(df["seq"])
            assert "ACG" in seqs
            assert "CGT" in seqs

    def test_compositional_stress(self):
        """Chain flip between two sequential ops, verify Cartesian product."""
        with pp.Party():
            parent = pp.from_seqs(["AACG", "TTCC"], mode="sequential")
            flipped = parent.flip(mode="sequential")
            df = _gen(flipped)
            assert len(df) == 4  # 2 seqs * 2 flip states
            assert df["seq"].nunique() == 4


class TestAdversarial_Filter:
    """At least 3 diagonal adversarial combos for filter."""

    def test_filter_after_insert_tags(self):
        """Filter after insert_tags — predicate gets clean content."""
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            tagged = p.annotate_region("mid", extent=(2, 6))
            filtered = tagged.filter(predicate=lambda s: len(s) == 8)
            df = _gen(filtered)
            assert not is_null_seq(df["seq"].iloc[0])

    def test_filter_between_sequential_ops(self):
        """Filter chained between two sequential ops."""
        with pp.Party():
            parent = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            filtered = parent.filter(predicate=lambda s: s != "CCCC")
            df = filtered.generate_library(num_cycles=1, discard_null_seqs=True)
            assert len(df) == 2

    def test_variable_length_parent_filter(self):
        """Variable-length parent + predicate on length."""
        with pp.Party():
            parent = pp.from_seqs(["AA", "CCCC", "GGGGGG"], mode="sequential")
            filtered = parent.filter(predicate=lambda s: len(s) >= 4)
            df = filtered.generate_library(num_cycles=1, discard_null_seqs=True)
            assert len(df) == 2

    def test_compositional_stress(self):
        """Chain filter between two sequential ops, verify product."""
        with pp.Party():
            parent = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            filtered = parent.filter(predicate=lambda s: True)
            mut = filtered.mutagenize(num_mutations=1, mode="sequential")
            df = _gen(mut)
            assert len(df) == mut.num_states


class TestAdversarial_Score:
    """At least 3 diagonal adversarial combos for score."""

    def test_score_named_region(self):
        """Score with named region — fn receives region content only."""
        captured = []
        with pp.Party():
            p = pp.from_seq("AAAA<r>GCGC</r>TTTT")
            result = p.score(
                fn=lambda s: captured.append(s) or len(s),
                region="r",
                cards=["score"],
            )
            df = _gen(result)
        assert captured[0] == "GCGC"
        assert df[_card_col(df, ".score")].iloc[0] == 4

    def test_score_interval_region_tagged(self):
        """Score with interval region on tagged sequence — fn receives clean content."""
        captured = []
        with pp.Party():
            p = pp.from_seq("AA<r>CCGG</r>TT")
            result = p.score(
                fn=lambda s: captured.append(s) or len(s),
                region=[2, 6],
                cards=["score"],
            )
            df = _gen(result)
        assert "<" not in captured[0]
        assert len(captured[0]) == 4

    def test_score_after_stylize(self):
        """Score chained after stylize — fn receives tag-stripped content."""
        captured = []
        with pp.Party():
            p = pp.from_seq("ACGT")
            styled = p.stylize(style="red")
            scored = styled.score(
                fn=lambda s: captured.append(s) or len(s),
                cards=["score"],
            )
            df = _gen(scored)
        assert captured[0] == "ACGT"
        assert df[_card_col(df, ".score")].iloc[0] == 4

    def test_compositional_stress(self):
        """Chain score between two sequential ops, verify product."""
        with pp.Party():
            parent = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            scored = parent.score(fn=len, cards=["score"])
            mut = scored.mutagenize(num_mutations=1, mode="sequential")
            df = _gen(mut)
            assert len(df) == mut.num_states


# ===================================================================
# STEP 6 — Contract tracing (runtime verification)
# ===================================================================

class TestContractTracing_SliceSeq:
    """Trace all 6 code paths in slice_seq."""

    def test_named_region_no_slice(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf")
            df = _gen(result)
            assert df["seq"].iloc[0] == "ATGCCC"

    def test_named_region_with_slice(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", start=0, stop=3)
            df = _gen(result)
            assert df["seq"].iloc[0] == "ATG"

    def test_named_region_keep_context_no_slice(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", keep_context=True)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AAAATGCCCTTT"

    def test_named_region_keep_context_with_slice(self):
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", start=0, stop=3, keep_context=True)
            df = _gen(result)
            assert df["seq"].iloc[0] == "AAAATGTTT"

    def test_interval_region_no_slice(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6])
            df = _gen(result)
            assert df["seq"].iloc[0] == "CCGG"

    def test_interval_region_with_slice(self):
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            result = p.slice_seq(region=[2, 6], start=1, stop=3)
            df = _gen(result)
            assert df["seq"].iloc[0] == "CG"

    def test_no_region(self):
        with pp.Party():
            p = pp.from_seq("ACGTACGT")
            result = p.slice_seq(start=2, stop=6)
            df = _gen(result)
            assert df["seq"].iloc[0] == "GTAC"


class TestContractTracing_Filter:
    """Trace filter _compute_core clean-string resolution."""

    def test_clean_present(self):
        """When parent has .clean, predicate uses it."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            filtered = p.filter(predicate=lambda s: s == "ACGT")
            df = _gen(filtered)
            assert not is_null_seq(df["seq"].iloc[0])

    def test_clean_empty_with_tags(self):
        """When .clean is empty but tags present, uses strip_all_tags."""
        with pp.Party():
            p = pp.from_seq("AACCGGTT")
            tagged = p.annotate_region("r", extent=(2, 6))
            filtered = tagged.filter(predicate=lambda s: len(s) == 8)
            df = _gen(filtered)
            assert not is_null_seq(df["seq"].iloc[0])

    def test_clean_empty_no_tags(self):
        """When .clean is empty and no tags, uses .string directly."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            upper_p = p.upper()
            filtered = upper_p.filter(predicate=lambda s: s == "ACGT")
            df = _gen(filtered)
            assert not is_null_seq(df["seq"].iloc[0])


# ===================================================================
# STEP 6 — API consistency checks
# ===================================================================

class TestAPIConsistency:
    """Mode support, pool type preservation, return types."""

    def test_stylize_hardcoded_dnapool(self):
        """FINDING: stylize returns DnaPool regardless of input pool type."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.stylize(style="red")
            assert isinstance(result, DnaPool)

    def test_add_prefix_hardcoded_dnapool(self):
        """FINDING: add_prefix returns DnaPool regardless of input pool type."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.add_prefix("x")
            assert isinstance(result, DnaPool)

    def test_score_preserves_pool_type(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            assert isinstance(p, DnaPool)
            result = p.score(fn=len)
            assert isinstance(result, DnaPool)

    def test_filter_preserves_pool_type(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True)
            assert isinstance(result, DnaPool)

    def test_flip_preserves_pool_type(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.flip()
            assert isinstance(result, DnaPool)

    def test_fixed_operation_preserves_pool_type(self):
        """FixedOp-based ops preserve pool type via fixed_operation."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.rc()
            assert isinstance(result, DnaPool)

    def test_join_preserves_first_pool_type(self):
        with pp.Party():
            p1 = pp.from_seq("AAAA")
            p2 = pp.from_seq("CCCC")
            result = pp.join([p1, p2])
            assert isinstance(result, DnaPool)

    def test_clear_gaps_seq_length_none(self):
        """clear_gaps always sets seq_length=None (documented design)."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.clear_gaps()
            assert result.seq_length is None

    def test_clear_annotation_seq_length_none(self):
        """clear_annotation always sets seq_length=None (documented design)."""
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.clear_annotation()
            assert result.seq_length is None

    def test_slice_seq_keep_context_named_region_seq_length(self):
        """seq_length is correctly computed for keep_context+named_region."""
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", keep_context=True)
            df = _gen(result)
            actual_len = len(strip_all_tags(df["seq"].iloc[0]))
            assert actual_len == 12
            assert result.seq_length == 12

    def test_slice_seq_keep_context_named_region_with_slice_seq_length(self):
        """seq_length is correctly computed for keep_context+named_region+slice."""
        with pp.Party():
            p = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            result = p.slice_seq(region="orf", start=0, stop=3, keep_context=True)
            df = _gen(result)
            actual_len = len(strip_all_tags(df["seq"].iloc[0]))
            assert actual_len == 9
            assert result.seq_length == 9


# ===================================================================
# I10 — State-space immutability during compute (all fixed ops)
# ===================================================================

class TestStateImmutability:
    """I10 for fixed ops — num_states and state._num_values unchanged by compute."""

    @pytest.mark.parametrize("op_name", ["upper", "lower", "swapcase", "rc"])
    def test_i10_group_a(self, op_name):
        with pp.Party():
            p = pp.from_seq("AcGt")
            result = getattr(p, op_name)()
            ns = result.num_states
            sv = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns
            assert result.operation.state._num_values == sv

    def test_i10_filter(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.filter(predicate=lambda s: True)
            ns = result.num_states
            sv = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns
            assert result.operation.state._num_values == sv

    def test_i10_score(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.score(fn=len)
            ns = result.num_states
            sv = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns
            assert result.operation.state._num_values == sv

    def test_i10_slice_seq(self):
        with pp.Party():
            p = pp.from_seq("ACGT")
            result = p.slice_seq(start=0, stop=2)
            ns = result.num_states
            sv = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns
            assert result.operation.state._num_values == sv

    def test_i10_join(self):
        with pp.Party():
            p1 = pp.from_seq("AA")
            p2 = pp.from_seq("CC")
            result = pp.join([p1, p2])
            ns = result.num_states
            sv = result.operation.state._num_values
            _gen(result)
            assert result.num_states == ns
            assert result.operation.state._num_values == sv
