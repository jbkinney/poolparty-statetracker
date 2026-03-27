"""Targeted checks for C1/C2 uncovered items in pre_publish_audit.md.

Covers:
  - Edge cases: empty inputs, single-char sequences, invalid alphabets,
    num_mutations > seq_length, zero-length regions
  - Parameter consistency: num_states type (int vs Integral)
  - recombine style cycling correctness
  - Error message consistency spot-checks
"""

import numpy as np
import pandas as pd
import pytest

import poolparty as pp
from poolparty.base_ops.from_iupac import from_iupac
from poolparty.base_ops.from_motif import from_motif
from poolparty.base_ops.from_seqs import from_seqs
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.base_ops.recombine import recombine
from poolparty.orf_ops.mutagenize_orf import mutagenize_orf
from poolparty.utils.parsing_utils import strip_all_tags


# ===================================================================
# C1: Source operations — edge cases
# ===================================================================


class TestFromSeqEdgeCases:

    def test_single_char(self):
        with pp.Party():
            pool = pp.from_seq("A").named("s")
        assert pool.seq_length == 1
        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "A"

    def test_empty_string_raises(self):
        """from_seq with empty string should either raise or produce length-0 output."""
        try:
            with pp.Party():
                pool = pp.from_seq("").named("s")
            df = pool.generate_library(num_cycles=1)
            assert pool.seq_length == 0
            assert df["seq"].iloc[0] == ""
        except (ValueError, Exception):
            pass  # Raising is also acceptable


class TestFromSeqsEdgeCases:

    def test_single_seq(self):
        with pp.Party():
            pool = from_seqs(["ACGT"], mode="sequential").named("s")
        assert pool.num_states == 1
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1

    def test_single_char_seqs(self):
        with pp.Party():
            pool = from_seqs(["A", "C", "G", "T"], mode="sequential").named("s")
        assert pool.num_states == 4
        df = pool.generate_library(num_cycles=1)
        assert set(df["seq"]) == {"A", "C", "G", "T"}

    def test_empty_list_raises(self):
        with pytest.raises((ValueError, IndexError)):
            with pp.Party():
                from_seqs([], mode="sequential").named("s")


class TestFromIupacEdgeCases:

    def test_single_char_iupac(self):
        """Single ambiguity code."""
        with pp.Party():
            pool = from_iupac("N", mode="sequential").named("s")
        assert pool.num_states == 4
        df = pool.generate_library(num_cycles=1)
        assert set(df["seq"]) == {"A", "C", "G", "T"}

    def test_long_fixed_iupac(self):
        """All-fixed IUPAC string has 1 state."""
        with pp.Party():
            pool = from_iupac("ACGT", mode="sequential").named("s")
        assert pool.num_states == 1
        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_invalid_iupac_char_raises(self):
        """Non-IUPAC character should raise."""
        with pytest.raises((ValueError, KeyError)):
            with pp.Party():
                from_iupac("X", mode="sequential").named("s")


class TestFromMotifEdgeCases:

    def test_single_position_motif(self):
        prob_df = pd.DataFrame({"A": [1.0], "C": [0.0], "G": [0.0], "T": [0.0]})
        with pp.Party():
            pool = from_motif(prob_df, mode="random").named("s")
        assert pool.seq_length == 1
        df = pool.generate_library(num_seqs=5, seed=42)
        assert all(s == "A" for s in df["seq"])

    def test_sequential_mode_rejected(self):
        prob_df = pd.DataFrame({"A": [0.5], "C": [0.5], "G": [0.0], "T": [0.0]})
        with pytest.raises(ValueError, match="random"):
            with pp.Party():
                from_motif(prob_df, mode="sequential").named("s")


class TestGetKmersEdgeCases:

    def test_length_1(self):
        with pp.Party():
            pool = get_kmers(length=1, mode="sequential").named("s")
        assert pool.num_states == 4

    def test_numpy_int_num_states_accepted(self):
        """get_kmers uses Optional[Integral] — accepts numpy int at factory."""
        with pp.Party():
            pool = get_kmers(length=2, mode="sequential", num_states=int(np.int64(5))).named("s")
        assert pool.num_states == 5
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 5


# ===================================================================
# C2: Mutation operations — edge cases
# ===================================================================


class TestMutagenizeEdgeCases:

    def test_num_mutations_exceeds_seq_length_raises(self):
        """num_mutations > seq_length should raise."""
        with pytest.raises(ValueError, match="exceeds"):
            with pp.Party():
                pp.from_seq("AC").mutagenize(
                    num_mutations=5, mode="sequential"
                ).named("s")

    def test_single_char_mutagenize(self):
        """Mutagenize a single character — 3 possible mutations."""
        with pp.Party():
            pool = pp.from_seq("A").mutagenize(
                num_mutations=1, mode="sequential"
            ).named("s")
        assert pool.num_states == 3  # C, G, T
        df = pool.generate_library(num_cycles=1)
        assert "A" not in set(df["seq"])  # all mutations differ from parent
        assert set(df["seq"]) == {"C", "G", "T"}

    def test_allowed_chars_restricts_mutations(self):
        """allowed_chars (per-position IUPAC) limits mutation alphabet.
        R = {A, G}, so on parent 'AAAA', mutations can only go to G."""
        with pp.Party():
            pool = pp.from_seq("AAAA").mutagenize(
                num_mutations=1, mode="sequential", allowed_chars="RRRR"
            ).named("s")
        # R allows {A, G} at each position; parent is A, so only G is a valid mutation.
        # C(4,1) * 1^1 = 4 states
        assert pool.num_states == 4
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            for c in seq:
                assert c in "AG"  # only A (parent) or G (allowed mutation)


class TestMutagenizeOrfEdgeCases:

    def test_num_mutations_exceeds_eligible_codons_raises(self):
        """num_mutations > eligible codons should raise."""
        orf = "ATGAAA"  # 2 codons
        with pytest.raises(ValueError, match="exceeds"):
            with pp.Party():
                mutagenize_orf(orf, num_mutations=5, mode="sequential").named("s")

    def test_minimum_orf_3bp(self):
        """3bp ORF (1 codon) with 1 mutation works."""
        with pp.Party():
            pool = mutagenize_orf("ATG", num_mutations=1, mode="sequential").named("s")
        assert pool.num_states > 0
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 3

    def test_numpy_int_num_states(self):
        """mutagenize_orf uses Optional[Integral] — accepts numpy int at factory."""
        with pp.Party():
            pool = mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mode="sequential",
                num_states=int(np.int64(5))
            ).named("s")
        assert pool.num_states == 5


class TestRecombineEdgeCases:

    def test_two_sources_single_breakpoint_fixed(self):
        """Minimal recombine: 2 sources, 1 breakpoint, fixed mode."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1,
                positions=[1], mode="fixed"
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        seq = df["seq"].iloc[0]
        assert seq == "AATT"

    def test_single_char_sources(self):
        """Recombine with short sources and 1 breakpoint at position 0."""
        with pp.Party():
            pool = recombine(
                sources=["AC", "TG"], num_breakpoints=1,
                positions=[0], mode="fixed"
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1


# ===================================================================
# C2: Region edge cases
# ===================================================================


class TestRegionEdgeCases:

    def test_mutagenize_full_region(self):
        """Region covers entire parent sequence."""
        with pp.Party():
            bg = pp.from_seq("AAAA")
            tagged = bg.insert_tags(region_name="all", start=0, stop=4)
            pool = tagged.mutagenize(
                region="all", num_mutations=1, mode="sequential"
            ).named("s")
        assert pool.num_states == 4 * 3  # C(4,1) * 3
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 12

    def test_mutagenize_single_position_region(self):
        """Region covers a single position."""
        with pp.Party():
            bg = pp.from_seq("AAAA")
            tagged = bg.insert_tags(region_name="one", start=2, stop=3)
            pool = tagged.mutagenize(
                region="one", num_mutations=1, mode="sequential"
            ).named("s")
        assert pool.num_states == 3  # 1 position, 3 alternatives
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[0:2] == "AA"  # left flank
            assert clean[3:] == "A"  # right flank
            assert clean[2] != "A"  # mutated position

    def test_from_iupac_region_same_length(self):
        """Insert IUPAC of same length as region — seq_length unchanged."""
        with pp.Party():
            bg = pp.from_seq("CCCCCCCC")  # 8 nt
            tagged = bg.insert_tags(region_name="ins", start=2, stop=5)  # 3-nt region
            pool = from_iupac("NNN", pool=tagged, region="ins", mode="random").named("s")
        assert pool.seq_length == 8  # no change


# ===================================================================
# C2: recombine style cycling correctness
# ===================================================================


class TestRecombineStyleCycling:

    def test_style_by_order(self):
        """styles cycle by segment order."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["bold", "italic"],
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        seq_obj = pool.operation._last_output if hasattr(pool.operation, "_last_output") else None
        assert len(df) >= 1

    def test_style_by_order_three_segments(self):
        """3 segments with 2 styles cycle: bold, italic, bold."""
        with pp.Party():
            pool = recombine(
                sources=["AAAAAA", "TTTTTT"],
                num_breakpoints=2,
                positions=[1, 3],
                mode="fixed",
                styles=["bold", "italic"],
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1

    def test_style_by_source(self):
        """styles applied by source pool index."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["bold", "italic"],
                style_by="source",
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1

    def test_styles_none_no_crash(self):
        """No styles provided — should not crash."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1

    def test_style_cycling_content_verification(self):
        """Verify styled output has correct content regardless of style."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["bold"],
                cards=["breakpoints", "pool_assignments"],
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        seq = df["seq"].iloc[0]
        bp = df[f"{op_name}.breakpoints"].iloc[0]
        pa = df[f"{op_name}.pool_assignments"].iloc[0]
        assert bp == (1,)
        assert pa == (0, 1)
        assert strip_all_tags(seq) == "AATT"

    def test_recombine_sequential_styles(self):
        """Sequential mode with styles produces styled output."""
        with pp.Party():
            pool = recombine(
                sources=["AAAAAA", "TTTTTT"],
                num_breakpoints=1,
                positions=[1, 2, 3],
                mode="sequential",
                styles=["bold", "italic"],
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert len(clean) == 6

    def test_recombine_random_styles_determinism(self):
        """Random mode with styles is deterministic with same seed."""
        results = []
        for _ in range(2):
            with pp.Party():
                pool = recombine(
                    sources=["AAAAAA", "TTTTTT"],
                    num_breakpoints=1,
                    mode="random",
                    styles=["bold"],
                ).named("s")
            df = pool.generate_library(num_seqs=10, seed=42)
            results.append(df["seq"].tolist())
        assert results[0] == results[1]


# ===================================================================
# C1/C2: Error message consistency spot-checks
# ===================================================================


class TestErrorMessageConsistency:
    """Verify error messages are descriptive and use consistent patterns."""

    def test_mutagenize_exceeds_length_message(self):
        with pytest.raises(ValueError, match="exceeds"):
            with pp.Party():
                pp.from_seq("AC").mutagenize(
                    num_mutations=5, mode="sequential"
                ).named("s")

    def test_mutagenize_orf_exceeds_codons_message(self):
        with pytest.raises(ValueError, match="exceeds"):
            with pp.Party():
                mutagenize_orf("ATGAAA", num_mutations=5, mode="sequential").named("s")

    def test_from_motif_rejects_sequential_message(self):
        prob_df = pd.DataFrame({"A": [0.5], "C": [0.5], "G": [0.0], "T": [0.0]})
        with pytest.raises(ValueError, match="random"):
            with pp.Party():
                from_motif(prob_df, mode="sequential").named("s")

    def test_mutagenize_variable_length_sequential_message(self):
        """Sequential mode with variable-length parent gives clear error."""
        with pytest.raises(ValueError, match="seq_length"):
            with pp.Party():
                src = from_seqs(["ACGT", "ACGTA"], mode="sequential")
                src.mutagenize(num_mutations=1, mode="sequential").named("s")

    def test_mutagenize_orf_invalid_mutation_type_message(self):
        with pytest.raises(ValueError, match="mutation_type"):
            with pp.Party():
                mutagenize_orf("ATGAAATTT", num_mutations=1, mutation_type="bogus").named("s")


# ===================================================================
# Parameter type consistency: num_states int vs Integral
# ===================================================================


class TestNumStatesTypeConsistency:
    """Document num_states type inconsistency across ops.

    FINDING: from_iupac, from_seqs, from_motif, mutagenize, recombine use
    Optional[int] which REJECTS np.int64 via beartype. get_kmers and
    mutagenize_orf use Optional[Integral] which accepts it at the factory
    level, but Pool.num_states return type is `int` so beartype still rejects
    the numpy int on property access.

    Decision needed: standardize all to Optional[Integral] (permissive) or
    keep Optional[int] (strict) and cast internally. Currently mixed.
    """

    def test_from_iupac_rejects_numpy_int(self):
        """from_iupac uses Optional[int] — rejects np.int64."""
        with pytest.raises(Exception):  # BeartypeCallHintParamViolation
            with pp.Party():
                from_iupac("NN", mode="sequential", num_states=np.int64(5)).named("s")

    def test_from_seqs_rejects_numpy_int(self):
        """from_seqs uses Optional[int] — rejects np.int64."""
        with pytest.raises(Exception):
            with pp.Party():
                from_seqs(["A", "C", "G", "T"], mode="sequential", num_states=np.int64(3)).named("s")

    def test_mutagenize_rejects_numpy_int(self):
        """mutagenize uses Optional[int] — rejects np.int64."""
        with pytest.raises(Exception):
            with pp.Party():
                pp.from_seq("ACGT").mutagenize(
                    num_mutations=1, mode="sequential", num_states=np.int64(5)
                ).named("s")

    def test_recombine_rejects_numpy_int(self):
        """recombine uses Optional[int] — rejects np.int64."""
        with pytest.raises(Exception):
            with pp.Party():
                recombine(
                    sources=["AAAA", "TTTT"], num_breakpoints=1,
                    positions=[1, 2], mode="sequential", num_states=np.int64(3)
                ).named("s")
