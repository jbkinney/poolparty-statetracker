"""Gap-filling tests for Source & Mutation operations audit.

Addresses testing gaps identified by comparing the merged audit report
(dev/audit/merged_source_mutation_ops.md) against the updated operation_audit.mdc rule.

Gaps filled:
  - Smoke tests for all factories
  - I1+I2 for from_seq, from_seqs, from_motif, mutagenize
  - I3 card-sequence agreement for from_seqs, from_iupac, get_kmers, mutagenize, mutagenize_orf
  - I5 determinism for from_iupac, get_kmers, mutagenize, mutagenize_orf
  - I6 region isolation for mutagenize, mutagenize_orf, recombine
  - I10 state-space immutability
  - Compositional stress test (Cartesian product + uniqueness)
  - Domain-specific invariants (IUPAC resolution, kmer validity, mutation position correctness)
  - Mixin forwarding for style, prefix, iter_order, num_states
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


# ---------------------------------------------------------------------------
# Smoke tests: every factory constructs and generates without crash
# ---------------------------------------------------------------------------


class TestSmoke:
    """Construct each factory with minimal inputs, generate 1 cycle."""

    def test_from_seq_smoke(self):
        with pp.Party():
            pool = pp.from_seq("ACGT").named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1

    def test_from_seqs_smoke(self):
        with pp.Party():
            pool = from_seqs(["ACGT", "TTTT"], mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 2

    def test_from_iupac_smoke(self):
        with pp.Party():
            pool = from_iupac("RY", mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 4

    def test_from_motif_smoke(self):
        prob_df = pd.DataFrame({"A": [0.5, 0.5], "C": [0.5, 0.5], "G": [0.0, 0.0], "T": [0.0, 0.0]})
        with pp.Party():
            pool = from_motif(prob_df, mode="random").named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_get_kmers_smoke(self):
        with pp.Party():
            pool = get_kmers(length=2, mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 16

    def test_mutagenize_smoke(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").mutagenize(
                num_mutations=1, mode="sequential"
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states

    def test_mutagenize_orf_smoke(self):
        with pp.Party():
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states

    def test_recombine_smoke(self):
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, positions=[1], mode="fixed"
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1


# ---------------------------------------------------------------------------
# I1+I2 for previously untested ops
# ---------------------------------------------------------------------------


class TestI1I2GapFill:
    """I1 (output length = seq_length) and I2 (sequential exhaustion) for ops
    that were missing these tests."""

    def test_from_seq_i1(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").named("s")
        assert pool.seq_length == 8
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 8

    def test_from_seqs_i1_i2(self):
        with pp.Party():
            pool = from_seqs(["ACGT", "TTTT", "GGGG"], mode="sequential").named("s")
        assert pool.seq_length == 4
        assert pool.num_states == 3
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 3
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 4

    def test_from_seqs_variable_length_seq_length_none(self):
        """from_seqs with variable-length inputs has seq_length=None (legitimate)."""
        with pp.Party():
            pool = from_seqs(["ACGT", "ACGTA"], mode="sequential").named("s")
        assert pool.seq_length is None

    def test_from_motif_i1(self):
        prob_df = pd.DataFrame({"A": [0.25, 0.25, 0.25], "C": [0.25, 0.25, 0.25],
                                "G": [0.25, 0.25, 0.25], "T": [0.25, 0.25, 0.25]})
        with pp.Party():
            pool = from_motif(prob_df, mode="random").named("s")
        assert pool.seq_length == 3
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 3

    def test_mutagenize_i1(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").mutagenize(
                num_mutations=1, mode="sequential"
            ).named("s")
        assert pool.seq_length == 8
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 8


# ---------------------------------------------------------------------------
# I3: Card-sequence agreement (trigger-based: all card-exposing ops)
# ---------------------------------------------------------------------------


class TestI3GapFill:
    """Card-sequence agreement for ops not previously tested."""

    def test_from_seqs_cards(self):
        seqs = ["ACGT", "TTTT", "GGGG"]
        with pp.Party():
            pool = from_seqs(seqs, seq_names=["s1", "s2", "s3"],
                             mode="sequential", cards=["seq_name", "seq_index"]).named("fs")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        for _, row in df.iterrows():
            idx = row[f"{op_name}.seq_index"]
            name = row[f"{op_name}.seq_name"]
            assert row["seq"] == seqs[idx]
            assert name == ["s1", "s2", "s3"][idx]

    def test_from_iupac_cards(self):
        with pp.Party():
            pool = from_iupac("RY", mode="sequential", cards=["iupac_state"]).named("iu")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        assert f"{op_name}.iupac_state" in df.columns
        assert len(df) == 4

    def test_get_kmers_cards(self):
        with pp.Party():
            pool = get_kmers(length=2, mode="sequential", cards=["kmer", "kmer_index"]).named("km")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        for _, row in df.iterrows():
            assert row["seq"] == row[f"{op_name}.kmer"]
            assert isinstance(row[f"{op_name}.kmer_index"], (int, np.integer))

    def test_mutagenize_cards(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").mutagenize(
                num_mutations=1, mode="sequential",
                cards=["positions", "wt_chars", "mut_chars"]
            ).named("mut")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        parent = "AAAA"
        for _, row in df.iterrows():
            positions = row[f"{op_name}.positions"]
            wt_chars = row[f"{op_name}.wt_chars"]
            mut_chars = row[f"{op_name}.mut_chars"]
            seq = row["seq"]
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert parent[pos] == wt
                assert strip_all_tags(seq)[pos] == mut
                assert wt != mut

    def test_mutagenize_orf_cards(self):
        orf = "ATGAAATTT"
        with pp.Party():
            pool = mutagenize_orf(
                orf, num_mutations=1, mode="sequential",
                cards=["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"]
            ).named("mo")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        for _, row in df.iterrows():
            wt_codons = row[f"{op_name}.wt_codons"]
            mut_codons = row[f"{op_name}.mut_codons"]
            for wt, mut in zip(wt_codons, mut_codons):
                assert len(wt) == 3
                assert len(mut) == 3
                assert wt != mut


# ---------------------------------------------------------------------------
# I5: Determinism (trigger-based: all random-mode ops)
# ---------------------------------------------------------------------------


class TestI5GapFill:
    """Determinism for ops not previously tested."""

    def test_from_iupac_determinism(self):
        results = []
        for _ in range(2):
            with pp.Party():
                pool = from_iupac("NNNN", mode="random").named("s")
            df = pool.generate_library(num_seqs=20, seed=42)
            results.append(df["seq"].tolist())
        assert results[0] == results[1]

    def test_get_kmers_determinism(self):
        results = []
        for _ in range(2):
            with pp.Party():
                pool = get_kmers(length=3, mode="random").named("s")
            df = pool.generate_library(num_seqs=20, seed=42)
            results.append(df["seq"].tolist())
        assert results[0] == results[1]

    def test_mutagenize_determinism(self):
        results = []
        for _ in range(2):
            with pp.Party():
                pool = pp.from_seq("ACGTACGT").mutagenize(
                    num_mutations=1, mode="random"
                ).named("s")
            df = pool.generate_library(num_seqs=20, seed=42)
            results.append(df["seq"].tolist())
        assert results[0] == results[1]

    def test_mutagenize_orf_determinism(self):
        results = []
        for _ in range(2):
            with pp.Party():
                pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="random").named("s")
            df = pool.generate_library(num_seqs=20, seed=42)
            results.append(df["seq"].tolist())
        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# I6: Region isolation (trigger-based: ops with region not previously tested)
# ---------------------------------------------------------------------------


class TestI6GapFill:
    """Region isolation for ops not previously tested."""

    def test_mutagenize_region_flanking(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTTAA")
            tagged = bg.insert_tags(region_name="zone", start=4, stop=6)
            pool = tagged.mutagenize(
                region="zone", num_mutations=1, mode="random"
            ).named("s")
        df = pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:4] == "AACC", f"Left flank changed: {clean}"
            assert clean[6:] == "TTAA", f"Right flank changed: {clean}"

    def test_mutagenize_orf_region_flanking(self):
        with pp.Party():
            bg = pp.from_seq("GGG<orf>ATGAAA</orf>CCC")
            pool = bg.mutagenize_orf("orf", num_mutations=1, mode="random", frame=1).named("s")
        df = pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:3] == "GGG", f"Left flank changed: {clean}"
            assert clean[9:] == "CCC", f"Right flank changed: {clean}"

    def test_recombine_region_flanking(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTTAA")
            tagged = bg.insert_tags(region_name="mid", start=4, stop=6)
            pool = tagged.recombine(
                region="mid", sources=["AA", "TT"],
                num_breakpoints=1, positions=[0], mode="fixed"
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:4] == "AACC", f"Left flank changed: {clean}"
            assert clean[6:] == "TTAA", f"Right flank changed: {clean}"


# ---------------------------------------------------------------------------
# I10: State-space immutability during compute
# ---------------------------------------------------------------------------


class TestI10StateImmutability:
    """Verify num_states and state._num_values don't change during compute."""

    def test_mutagenize_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGT").mutagenize(
                num_mutations=1, mode="sequential"
            ).named("s")
        ns_before = pool.num_states
        sv_before = pool.operation.state._num_values
        pool.generate_library(num_cycles=1)
        assert pool.num_states == ns_before
        assert pool.operation.state._num_values == sv_before

    def test_from_iupac_state_immutability(self):
        with pp.Party():
            pool = from_iupac("RY", mode="sequential").named("s")
        ns_before = pool.num_states
        sv_before = pool.operation.state._num_values
        pool.generate_library(num_cycles=1)
        assert pool.num_states == ns_before
        assert pool.operation.state._num_values == sv_before

    def test_mutagenize_orf_state_immutability(self):
        with pp.Party():
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential").named("s")
        ns_before = pool.num_states
        sv_before = pool.operation.state._num_values
        pool.generate_library(num_cycles=1)
        assert pool.num_states == ns_before
        assert pool.operation.state._num_values == sv_before


# ---------------------------------------------------------------------------
# Compositional stress test: full Cartesian product + uniqueness
# ---------------------------------------------------------------------------


class TestCompositionalStress:
    """Chain audited op between two sequential ops, verify Cartesian product."""

    def test_from_iupac_in_chain(self):
        with pp.Party():
            src = from_seqs(["AAAA", "TTTT"], mode="sequential")
            tagged = src.insert_tags(region_name="ins", start=1, stop=3)
            mid = from_iupac("NN", pool=tagged, region="ins", mode="sequential").named("mid")
        N = mid.operation.state._num_values
        df = mid.generate_library(num_cycles=1)
        assert len(df) == 2 * N

    def test_mutagenize_in_chain(self):
        with pp.Party():
            src = from_seqs(["AAAA", "CCCC"], mode="sequential")
            mut = src.mutagenize(num_mutations=1, mode="sequential").named("mut")
        src_states = 2
        mut_states = mut.operation.state._num_values
        df = mut.generate_library(num_cycles=1)
        assert len(df) == src_states * mut_states

    def test_three_level_chain(self):
        with pp.Party():
            src = from_seqs(["ACGT", "TTTT"], mode="sequential")
            mid = src.mutagenize(num_mutations=1, mode="sequential")
            end = mid.upper().named("end")
        df = end.generate_library(num_cycles=1)
        assert len(df) == end.num_states
        for seq in df["seq"]:
            assert seq == seq.upper()


# ---------------------------------------------------------------------------
# Domain-specific invariants
# ---------------------------------------------------------------------------


class TestDomainSpecific:
    """Domain-specific invariants for source and mutation ops."""

    def test_iupac_resolves_correct_bases(self):
        """R should resolve to A or G only."""
        with pp.Party():
            pool = from_iupac("R", mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        assert set(df["seq"]) == {"A", "G"}

    def test_iupac_boundary_single_base(self):
        """Single fixed base IUPAC code has exactly 1 state."""
        with pp.Party():
            pool = from_iupac("A", mode="sequential").named("s")
        assert pool.num_states == 1
        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "A"

    def test_kmer_correct_length_and_alphabet(self):
        """All kmers have correct length and only contain ACGT."""
        with pp.Party():
            pool = get_kmers(length=3, mode="sequential").named("s")
        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 3
            assert all(c in "ACGT" for c in seq)

    def test_kmer_boundary_length_1(self):
        """k=1 produces exactly 4 kmers."""
        with pp.Party():
            pool = get_kmers(length=1, mode="sequential").named("s")
        assert pool.num_states == 4
        df = pool.generate_library(num_cycles=1)
        assert set(df["seq"]) == {"A", "C", "G", "T"}

    def test_mutagenize_positions_within_region(self):
        """All mutation positions must be within the region bounds.
        Positions in cards are region-relative (0-based within region content)."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            tagged = bg.insert_tags(region_name="zone", start=3, stop=7)
            pool = tagged.mutagenize(
                region="zone", num_mutations=1, mode="sequential",
                cards=["positions"]
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        region_len = 4
        for _, row in df.iterrows():
            for pos in row[f"{op_name}.positions"]:
                assert 0 <= pos < region_len, f"Position {pos} outside region [0,{region_len})"

    def test_mutagenize_orf_codon_aligned(self):
        """mutagenize_orf mutations occur at codon-aligned positions."""
        orf = "ATGAAATTTGGG"
        with pp.Party():
            pool = mutagenize_orf(
                orf, num_mutations=1, mode="sequential",
                cards=["codon_positions"]
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        op_name = pool.operation.name
        for _, row in df.iterrows():
            seq = strip_all_tags(row["seq"])
            codon_positions = row[f"{op_name}.codon_positions"]
            for cp in codon_positions:
                codon_start = cp * 3
                assert codon_start + 3 <= len(seq), f"Codon position {cp} out of bounds"


# ---------------------------------------------------------------------------
# Mixin forwarding: test style, prefix, iter_order, num_states
# ---------------------------------------------------------------------------


class TestMixinForwarding:
    """Verify mixin methods forward all factory parameters without TypeError."""

    def test_mutagenize_mixin_style(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").mutagenize(
                num_mutations=1, mode="random", style="bold"
            ).named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_mutagenize_mixin_prefix(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").mutagenize(
                num_mutations=1, mode="random", prefix="mut"
            ).named("s")
        assert "mut" in pool.operation.name

    def test_mutagenize_mixin_iter_order(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT").mutagenize(
                num_mutations=1, mode="random", iter_order=5.0
            ).named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_mutagenize_mixin_num_states(self):
        with pp.Party():
            pool = pp.from_seq("ACGT").mutagenize(
                num_mutations=1, mode="sequential", num_states=5
            ).named("s")
        assert pool.num_states == 5
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 5

    def test_recombine_mixin_style(self):
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            tagged = bg.insert_tags(region_name="mid", start=3, stop=7)
            pool = tagged.recombine(
                region="mid", sources=["CCCC", "GGGG"],
                num_breakpoints=1, positions=[1], mode="fixed",
                styles=["italic"]
            ).named("s")
        df = pool.generate_library(num_cycles=1)
        assert len(df) >= 1

    def test_mutagenize_orf_mixin_style(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1, mode="random", style="bold"
            ).named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_mutagenize_orf_mixin_num_states(self):
        """mutagenize_orf respects user num_states override (BUG #48 fixed)."""
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1, mode="sequential", num_states=3
            ).named("s")
        assert pool.num_states == 3
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 3

    def test_insert_from_iupac_mixin_style(self):
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            tagged = bg.insert_tags(region_name="ins", start=3, stop=6)
            pool = tagged.insert_from_iupac(
                "NN", region="ins", mode="random", style="bold"
            ).named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_insert_kmers_mixin_num_states(self):
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            tagged = bg.insert_tags(region_name="ins", start=3, stop=6)
            pool = tagged.insert_kmers(
                length=3, region="ins", mode="sequential", num_states=5
            ).named("s")
        assert pool.num_states == 5
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 5
