"""Invariant tests for Source & Mutation operations audit.

Tests gaps identified in the audit (dev/audit/source_mutation_ops.md):
  I1 - Output length matches seq_length
  I2 - State exhaustion in sequential mode
  I3 - Card-sequence agreement
  I4 - Region tag preservation
  I5 - Determinism (recombine)
  I6 - Region-constrained op only modifies the region
  I7 - Composition (chaining ops)
  Critical bugs - mutagenize_orf named-region crash, interval-tag-crossing
"""

import pytest

from math import comb

import poolparty as pp
from poolparty.base_ops.from_iupac import from_iupac
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.base_ops.recombine import recombine
from poolparty.orf_ops.mutagenize_orf import mutagenize_orf
from poolparty.utils.parsing_utils import strip_all_tags


# ---------------------------------------------------------------------------
# I1: Output length matches seq_length
# ---------------------------------------------------------------------------


class TestI1OutputLengthMatchesSeqLength:
    """Verify that every output row's molecular length equals pool.seq_length."""

    def test_from_iupac_region_seq_length(self):
        """from_iupac as insert-into-region reports correct combined seq_length (BUG #43 fixed)."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")  # 10 nt
            tagged = bg.insert_tags(region_name="ins", start=3, stop=6)  # 3-nt region
            pool = from_iupac("NN", pool=tagged, region="ins", mode="sequential").named("iupac")

        # bg(10) - region(3) + insert(2) = 9
        assert pool.seq_length == 9

    def test_from_iupac_region_actual_output_length(self):
        """from_iupac with region produces output of correct actual length."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")  # 10 nt
            tagged = bg.insert_tags(region_name="ins", start=3, stop=6)  # 3-nt region
            pool = from_iupac("NN", pool=tagged, region="ins", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 9

    def test_get_kmers_region_seq_length(self):
        """get_kmers as insert-into-region produces correct seq_length."""
        with pp.Party():
            bg = pp.from_seq("CCCCCCCCCC")  # 10 nt
            tagged = bg.insert_tags(region_name="slot", start=4, stop=7)  # 3-nt region
            pool = get_kmers(length=2, pool=tagged, region="slot", mode="sequential").named("kmer")

        expected_len = pool.seq_length
        assert expected_len is not None
        # bg(10) - region(3) + kmer(2) = 9
        assert expected_len == 9

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == expected_len

    def test_recombine_region_seq_length(self):
        """recombine into a region reports correct combined seq_length (BUG #44 fixed)."""
        with pp.Party():
            bg = pp.from_seq("GGGGGGGG")  # 8 nt
            tagged = bg.insert_tags(region_name="mid", start=2, stop=6)  # 4-nt region
            pool = tagged.recombine(
                region="mid",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        # region is 4 nt, sources are 4 nt => same total length = 8
        assert pool.seq_length == 8

    def test_recombine_region_actual_output_length(self):
        """recombine into a region produces output of correct actual length."""
        with pp.Party():
            bg = pp.from_seq("GGGGGGGG")  # 8 nt
            tagged = bg.insert_tags(region_name="mid", start=2, stop=6)  # 4-nt region
            pool = tagged.recombine(
                region="mid",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        df = pool.generate_library()
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == 8


# ---------------------------------------------------------------------------
# I2: State exhaustion -- sequential mode produces correct row count
# ---------------------------------------------------------------------------


class TestI2StateExhaustion:
    """Verify that sequential mode exhaustion produces exactly num_states rows."""

    def test_get_kmers_sequential_exhaustion(self):
        """get_kmers sequential mode produces exactly 4^k rows on full cycle."""
        with pp.Party():
            pool = get_kmers(length=2, mode="sequential").named("kmer")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) == 16  # 4^2

    def test_from_iupac_cycling(self):
        """from_iupac with num_states > natural cycles correctly."""
        with pp.Party():
            # RY has 2*2 = 4 natural states
            pool = from_iupac("RY", mode="sequential", num_states=10).named("iupac")

        assert pool.num_states == 10

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 10

        # The 4 natural sequences should appear, cycling
        natural = {"AC", "AT", "GC", "GT"}
        observed = set(df["seq"])
        assert observed == natural


# ---------------------------------------------------------------------------
# I4: Region tag preservation
# ---------------------------------------------------------------------------


class TestI4RegionTagPreservation:
    """Verify that region tags survive mutation/recombination."""

    def test_mutagenize_region_tag_preservation(self):
        """mutagenize with named region preserves tags in output."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")  # 10 nt
            tagged = bg.insert_tags(region_name="mut_zone", start=3, stop=7)
            pool = tagged.mutagenize(
                region="mut_zone", num_mutations=1, mode="random"
            ).named("mutant")

        df = pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert "<mut_zone>" in seq
            assert "</mut_zone>" in seq

    def test_recombine_tagged_sources(self):
        """recombine into a tagged region preserves region tags."""
        with pp.Party():
            bg = pp.from_seq("NNNNNNNN")
            tagged = bg.insert_tags(region_name="middle", start=2, stop=6)
            pool = tagged.recombine(
                region="middle",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        df = pool.generate_library()
        seq = df["seq"].iloc[0]
        assert "<middle>" in seq
        assert "</middle>" in seq


# ---------------------------------------------------------------------------
# I5: Determinism (recombine -- not previously tested)
# ---------------------------------------------------------------------------


class TestI5DeterminismRecombine:
    """Verify that recombine in fixed/sequential mode is deterministic."""

    def test_determinism_recombine(self):
        """Same seed produces identical recombine results."""
        results = []
        for _ in range(2):
            with pp.Party():
                pool = recombine(
                    sources=["AAAAAAAA", "TTTTTTTT"],
                    num_breakpoints=2,
                    mode="random",
                ).named("recombined")
            df = pool.generate_library(num_seqs=20, seed=99)
            results.append(list(df["seq"]))

        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# I6: Region-constrained op only modifies the region
# ---------------------------------------------------------------------------


class TestI6FlankingPreservation:
    """Verify that region-constrained ops leave flanking sequence intact."""

    def test_get_kmers_region_flanking(self):
        """get_kmers with region only changes the region, flanks are preserved."""
        with pp.Party():
            bg = pp.from_seq("AACCGGTTAA")  # 10 nt
            tagged = bg.insert_tags(region_name="slot", start=4, stop=6)  # 2-nt region
            pool = get_kmers(length=2, pool=tagged, region="slot", mode="sequential").named("kmer")

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            # Flanks: positions 0-3 = "AACC", positions 6-9 = "TTAA"
            assert clean[:4] == "AACC", f"Left flank changed: {clean}"
            assert clean[6:] == "TTAA", f"Right flank changed: {clean}"

    def test_from_iupac_region_flanking(self):
        """from_iupac with region only changes the region, flanks are preserved."""
        with pp.Party():
            bg = pp.from_seq("TTGGCCAATT")  # 10 nt
            tagged = bg.insert_tags(region_name="ins", start=3, stop=7)  # 4-nt region
            pool = from_iupac("NNNN", pool=tagged, region="ins", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:3] == "TTG", f"Left flank changed: {clean}"
            assert clean[7:] == "ATT", f"Right flank changed: {clean}"


# ---------------------------------------------------------------------------
# I7: Composition -- chained ops propagate seq_length correctly
# ---------------------------------------------------------------------------


class TestI7Composition:
    """Verify that chaining ops produces correct seq_length and valid output."""

    def test_recombine_then_mutagenize(self):
        """recombine output can be consumed by mutagenize with correct seq_length."""
        with pp.Party():
            pool = recombine(
                sources=["AAAAAAAA", "TTTTTTTT"],
                num_breakpoints=1,
                positions=[3],
                mode="fixed",
            )
            mutated = pool.mutagenize(num_mutations=1, mode="random").named("mutant")

        expected_len = mutated.seq_length
        assert expected_len == 8

        df = mutated.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert len(strip_all_tags(seq)) == expected_len

    def test_mutagenize_orf_then_downstream(self):
        """mutagenize_orf output can be consumed by a downstream fixed op."""
        orf = "ATGAAATTTGGG"  # 12 nt, 4 codons
        with pp.Party():
            pool = mutagenize_orf(orf, num_mutations=1, mode="random")
            upper_pool = pool.upper().named("uppered")

        expected_len = upper_pool.seq_length
        assert expected_len == 12

        df = upper_pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert len(clean) == expected_len
            assert clean == clean.upper()


# ---------------------------------------------------------------------------
# Critical bug: mutagenize_orf sequential + named-region crash
# ---------------------------------------------------------------------------


class TestMutagenizeOrfNamedRegionCrash:
    """BUG #45: mutagenize_orf crashes with IndexError in sequential mode
    when region is a named string.

    Root cause: At init time, num_codons is computed from the full parent
    seq_length (placeholder orf_start=0, orf_end=parent_len). At compute
    time, actual ORF bounds are resolved from tags, yielding fewer codons.
    The sequential cache has position indices valid for init-time geometry
    but out of range for compute-time geometry.
    """

    def test_mutagenize_orf_sequential_named_region_embedded_tags(self):
        """Embedded ORF tags + sequential mode works after BUG #45 fix."""
        with pp.Party():
            pool = (
                pp.from_seq("GGG<orf>ATGAAA</orf>CCC")
                .mutagenize_orf("orf", num_mutations=1, mode="sequential", frame=1)
            ).named("test")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) > 0
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert len(clean) == 12

    def test_mutagenize_orf_sequential_annotate_orf(self):
        """annotate_orf + sequential mode works after BUG #45 fix."""
        with pp.Party():
            bg = pp.from_seq("GGGATGAAACCC")
            orf_pool = bg.annotate_orf("myorf", extent=(3, 9), frame=1)
            pool = orf_pool.mutagenize_orf(
                "myorf", num_mutations=1, mode="sequential"
            ).named("test")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) > 0
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert len(clean) == 12

    def test_mutagenize_orf_sequential_interval_region_works(self):
        """Interval region + sequential mode works correctly (control case)."""
        with pp.Party():
            pool = pp.from_seq("GGGATGAAACCC").mutagenize_orf(
                region=[3, 9], num_mutations=1, mode="sequential", frame=1
            ).named("test")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) > 0

    def test_mutagenize_orf_random_named_region_works(self):
        """Named region + random mode works correctly (control case)."""
        with pp.Party():
            pool = (
                pp.from_seq("GGG<orf>ATGAAA</orf>CCC")
                .mutagenize_orf("orf", num_mutations=1, mode="random", frame=1)
            ).named("test")
        df = pool.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert len(clean) == 12


# ---------------------------------------------------------------------------
# Critical bug: interval crossing tag boundaries (known issue #19)
# ---------------------------------------------------------------------------


class TestIntervalCrossingTagBoundary:
    """Interval-based region operations that cross named-tag boundaries
    can produce malformed XML output (opening tag destroyed, closing tag
    remains). This is known issue #19.
    """

    @pytest.mark.xfail(
        reason="Known issue #19: interval crossing tag boundary produces "
        "malformed output (opening tag destroyed, closing tag remains).",
        strict=True,
    )
    def test_from_iupac_interval_crossing_tag(self):
        """from_iupac with interval crossing a named tag produces valid XML."""
        with pp.Party():
            bg = pp.from_seq("AAAA<region>CCCC</region>GGGG")
            pool = from_iupac(
                "NNNN", pool=bg, region=[2, 6], mode="sequential"
            ).named("iupac")

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # Every opening tag should have a matching closing tag
            has_open = "<region>" in seq
            has_close = "</region>" in seq
            assert has_open == has_close, (
                f"Malformed tags: open={has_open}, close={has_close} in {seq!r}"
            )

    @pytest.mark.xfail(
        reason="Known issue #19: interval crossing tag boundary produces "
        "malformed output (opening tag destroyed, closing tag remains).",
        strict=True,
    )
    def test_get_kmers_interval_crossing_tag(self):
        """get_kmers with interval crossing a named tag produces valid XML."""
        with pp.Party():
            bg = pp.from_seq("AAAA<region>CCCC</region>GGGG")
            pool = get_kmers(
                length=4, pool=bg, region=[2, 6], mode="sequential"
            ).named("kmer")

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            has_open = "<region>" in seq
            has_close = "</region>" in seq
            assert has_open == has_close, (
                f"Malformed tags: open={has_open}, close={has_close} in {seq!r}"
            )


# ---------------------------------------------------------------------------
# Additional invariant: mutagenize region-constrained state count
# ---------------------------------------------------------------------------


class TestMutagenizeRegionStateCount:
    """Verify mutagenize sequential state counts with region constraint."""

    def test_named_region_state_count(self):
        """Named region length 4, num_mutations=2: C(4,2)*3^2 = 54 states."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            tagged = bg.insert_tags(region_name="zone", start=3, stop=7)
            pool = tagged.mutagenize(
                region="zone", num_mutations=2, mode="sequential"
            ).named("mut")

        assert pool.num_states == comb(4, 2) * 3**2  # 54
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 54

    def test_interval_region_state_count(self):
        """Interval region length 4, num_mutations=2: same 54 states."""
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            pool = bg.mutagenize(
                region=[3, 7], num_mutations=2, mode="sequential"
            ).named("mut")

        assert pool.num_states == comb(4, 2) * 3**2  # 54
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 54


# ---------------------------------------------------------------------------
# I3: Card-sequence agreement (recombine cards via factory)
# ---------------------------------------------------------------------------


class TestI3CardSequenceAgreement:
    """Verify design card data is coherent with output sequences."""

    def test_recombine_cards_coherent(self):
        """recombine breakpoints and pool_assignments match output structure."""
        with pp.Party():
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                cards=["breakpoints", "pool_assignments"],
            ).named("rec")

        op_name = pool.operation.name
        df = pool.generate_library()
        bp = df[f"{op_name}.breakpoints"].iloc[0]
        pa = df[f"{op_name}.pool_assignments"].iloc[0]
        seq = df["seq"].iloc[0]

        assert bp == (1,)
        assert pa == (0, 1)
        # First 2 chars from pool 0 (AAAA), last 2 from pool 1 (TTTT)
        assert seq[:2] == "AA"
        assert seq[2:] == "TT"
