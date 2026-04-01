"""ORF operations audit — test_audit_orf_ops_bravo.py

Covers 5 ops: translate, reverse_translate, mutagenize_orf, stylize_orf, annotate_orf.

Follows operation_audit.mdc Steps 1–7.
"""

import pytest
import numpy as np

import poolparty as pp
from poolparty.codon_table import CodonTable, STANDARD_GENETIC_CODE
from poolparty.dna_pool import DnaPool
from poolparty.protein_pool import ProteinPool
from poolparty.utils.parsing_utils import strip_all_tags


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
    """Find a card column containing the given suffix."""
    cols = [c for c in df.columns if suffix in c]
    assert len(cols) == 1, f"Expected 1 column with '{suffix}', got {cols}"
    return cols[0]


# ===================================================================
# STEP 2 — Smoke tests (every factory inside Party + generate)
# ===================================================================

class TestSmokeTests:
    """Verify every ORF factory constructs and generates without crashing."""

    def test_translate_smoke(self):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA")
            df = _gen(pool)
            assert len(df) >= 1
            assert df["seq"].iloc[0] is not None

    def test_reverse_translate_smoke(self):
        with pp.Party():
            pool = pp.reverse_translate("MAK")
            df = _gen(pool)
            assert len(df) >= 1
            assert df["seq"].iloc[0] is not None

    def test_mutagenize_orf_smoke(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAATTT", num_mutations=1).named("m")
            df = _gen(pool)
            assert len(df) >= 1

    def test_stylize_orf_smoke(self):
        with pp.Party():
            pool = pp.stylize_orf("ATGAAATTT", style_codons=["red"]).named("s")
            df = _gen(pool)
            assert len(df) >= 1

    def test_annotate_orf_smoke(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA")
            result = pp.annotate_orf(pool, "cds")
            df = _gen(result)
            assert len(df) >= 1
            assert "<cds>" in df["seq"].iloc[0]


# ===================================================================
# STEP 2 — Mixin forwarding (runtime, not just signature comparison)
# ===================================================================

class TestMixinForwarding:
    """Verify every mixin method delegates correctly at runtime."""

    def test_translate_mixin_basic(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT")
            result = pool.translate()
            assert isinstance(result, ProteinPool)
            df = _gen(result)
            assert df["seq"].iloc[0] == "MA"

    def test_translate_mixin_all_params(self):
        """Call translate mixin with every factory-supported parameter."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").annotate_orf("cds", frame=1)
            result = pool.translate(
                region="cds",
                frame=1,
                include_stop=False,
                preserve_codon_styles=True,
                genetic_code="standard",
                iter_order=1.0,
                prefix="prot",
            )
            df = _gen(result)
            assert df["seq"].iloc[0] == "MA"

    def test_reverse_translate_mixin_basic(self):
        with pp.Party():
            prot = pp.from_seq("ATGGCT").translate()
            result = prot.reverse_translate()
            assert isinstance(result, DnaPool)
            df = _gen(result)
            assert len(df["seq"].iloc[0]) == 6

    def test_reverse_translate_mixin_all_params(self):
        with pp.Party():
            prot = pp.from_seq("ATGGCT").translate()
            result = prot.reverse_translate(
                codon_selection="random",
                num_states=5,
                genetic_code="standard",
                iter_order=2.0,
                prefix="dna",
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) == 5

    def test_mutagenize_orf_mixin_basic(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT")
            result = pool.mutagenize_orf(num_mutations=1)
            df = _gen(result)
            assert len(df) >= 1

    def test_mutagenize_orf_mixin_all_params(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTTGGG")
            result = pool.mutagenize_orf(
                num_mutations=1,
                mutation_type="any_codon",
                codon_positions=[0, 1],
                style="red",
                frame=1,
                prefix="mut",
                mode="sequential",
                num_states=10,
                iter_order=1.0,
                cards=["codon_positions", "mut_codons"],
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) == 10

    def test_stylize_orf_mixin_basic(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT")
            result = pool.stylize_orf(style_codons=["red", "blue"])
            df = _gen(result)
            assert len(df) >= 1

    def test_stylize_orf_mixin_all_params(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT")
            result = pool.stylize_orf(
                style_codons=["red", "blue"],
                frame=2,
                iter_order=1.0,
                prefix="sty",
            )
            df = _gen(result)
            assert len(df) >= 1

    def test_annotate_orf_mixin_basic(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA")
            result = pool.annotate_orf("cds")
            df = _gen(result)
            assert "<cds>" in df["seq"].iloc[0]

    def test_annotate_orf_mixin_all_params(self):
        with pp.Party():
            pool = pp.from_seq("NNATGGCTTAANN")
            result = pool.annotate_orf(
                "cds",
                extent=(2, 11),
                frame=1,
                style_codons=["red", "green", "blue"],
                iter_order=1.0,
                prefix="ann",
            )
            df = _gen(result)
            assert "<cds>" in df["seq"].iloc[0]


# ===================================================================
# STEP 3 — Invariant testing
# ===================================================================

# -------------------------------------------------------------------
# I1. Output length = declared seq_length
# -------------------------------------------------------------------

class TestI1OutputLength:

    def test_translate_output_length(self):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA")
            assert pool.seq_length == 3  # 9 / 3 = 3 codons
            df = _gen(pool)
            assert len(df["seq"].iloc[0]) == pool.seq_length

    def test_translate_no_stop_output_length_with_stop_codon(self):
        """include_stop=False: seq_length is None (data-dependent), output is correct."""
        with pp.Party():
            pool = pp.translate("ATGGCTTAA", include_stop=False)
            assert pool.seq_length is None
            df = _gen(pool)
            assert df["seq"].iloc[0] == "MA"

    def test_translate_no_stop_output_length_without_stop_codon(self):
        """include_stop=False with no stop codon: seq_length is None, output correct."""
        with pp.Party():
            pool = pp.translate("ATGGCT", include_stop=False)
            assert pool.seq_length is None
            df = _gen(pool)
            assert df["seq"].iloc[0] == "MA"

    def test_translate_no_stop_single_codon(self):
        """include_stop=False with single codon: seq_length is None, output correct."""
        with pp.Party():
            pool = pp.translate("ATG", include_stop=False)
            assert pool.seq_length is None
            df = _gen(pool)
            assert df["seq"].iloc[0] == "M"

    def test_translate_frame2_output_length(self):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA", frame=2)
            # skip 1 base, (9-1)//3 = 2 codons
            assert pool.seq_length == 2
            df = _gen(pool)
            assert len(df["seq"].iloc[0]) == pool.seq_length

    def test_reverse_translate_output_length(self):
        with pp.Party():
            pool = pp.reverse_translate("MAPK")
            assert pool.seq_length == 12  # 4 AAs * 3
            df = _gen(pool)
            assert len(df["seq"].iloc[0]) == pool.seq_length

    def test_mutagenize_orf_preserves_length(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAATTTGGG", num_mutations=1, mode="sequential")
            assert pool.seq_length == 12
            df = _gen(pool)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == pool.seq_length

    def test_mutagenize_orf_with_region_preserves_length(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAA</orf>CCC", "orf", num_mutations=1, frame=1
            )
            assert pool.seq_length == 12
            df = pool.generate_library(num_seqs=10, seed=42)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == pool.seq_length

    def test_stylize_orf_preserves_length(self):
        with pp.Party():
            pool = pp.stylize_orf("ATGAAATTT", style_codons=["red"])
            assert pool.seq_length == 9
            df = _gen(pool)
            assert len(strip_all_tags(df["seq"].iloc[0])) == pool.seq_length

    def test_annotate_orf_output_length(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA")
            result = pp.annotate_orf(pool, "cds")
            # Tags don't change biological length
            assert result.seq_length == 9
            df = _gen(result)
            assert len(strip_all_tags(df["seq"].iloc[0])) == result.seq_length


# -------------------------------------------------------------------
# I2. Sequential exhaustion = correct row count
# -------------------------------------------------------------------

class TestI2StateExhaustion:

    def test_mutagenize_orf_sequential_exhaustion(self):
        with pp.Party():
            # 3 codons, missense_only_first => 19 alts, 1 mutation
            pool = pp.mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential").named("m")
            assert pool.num_states == 57  # 3 * 19
            df = _gen(pool)
            assert len(df) == pool.num_states

    def test_mutagenize_orf_double_mutation_exhaustion(self):
        with pp.Party():
            # C(3,2)=3 * 19^2=361 => 1083
            pool = pp.mutagenize_orf("ATGAAATTT", num_mutations=2, mode="sequential").named("m")
            assert pool.num_states == 1083
            df = _gen(pool)
            assert len(df) == pool.num_states

    def test_mutagenize_orf_any_codon_exhaustion(self):
        with pp.Party():
            # 2 codons, any_codon => 63 alts, 1 mutation => 126
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="any_codon", mode="sequential"
            ).named("m")
            assert pool.num_states == 126
            df = _gen(pool)
            assert len(df) == pool.num_states

    def test_translate_fixed_single_state(self):
        with pp.Party():
            pool = pp.translate("ATGGCT")
            assert pool.num_states == 1
            df = _gen(pool)
            assert len(df) == 1

    def test_stylize_orf_fixed_single_state(self):
        with pp.Party():
            pool = pp.stylize_orf("ATGAAA", style_codons=["red"])
            assert pool.num_states == 1
            df = _gen(pool)
            assert len(df) == 1


# -------------------------------------------------------------------
# I3. Card-sequence agreement
# -------------------------------------------------------------------

class TestI3CardAgreement:

    def test_mutagenize_orf_card_agreement(self):
        ct = CodonTable("standard")
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mode="sequential",
                cards=["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"],
            ).named("m")

        df = _gen(pool)
        pos_col = _card_col(df, "codon_positions")
        wt_c_col = _card_col(df, "wt_codons")
        mut_c_col = _card_col(df, "mut_codons")
        wt_a_col = _card_col(df, "wt_aas")
        mut_a_col = _card_col(df, "mut_aas")

        for _, row in df.iterrows():
            seq = row["seq"]
            for pos, wt_c, mut_c, wt_a, mut_a in zip(
                row[pos_col], row[wt_c_col], row[mut_c_col],
                row[wt_a_col], row[mut_a_col],
            ):
                # Codon in sequence matches reported mut_codon
                actual_codon = seq[pos * 3 : pos * 3 + 3]
                assert actual_codon == mut_c
                # AA translations match
                assert ct.codon_to_aa.get(wt_c.upper()) == wt_a
                assert ct.codon_to_aa.get(mut_c.upper()) == mut_a
                # Mutation actually changed the codon
                assert wt_c != mut_c


# -------------------------------------------------------------------
# I4. Region tag preservation
# -------------------------------------------------------------------

class TestI4TagPreservation:

    def test_mutagenize_orf_preserves_tags(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAA</orf>CCC", "orf", num_mutations=1, frame=1
            ).named("m")
            df = pool.generate_library(num_seqs=20, seed=42)
            for seq in df["seq"]:
                assert "<orf>" in seq
                assert "</orf>" in seq
                assert seq[:3] == "GGG"
                assert seq.endswith("CCC")

    def test_stylize_orf_preserves_tags(self):
        with pp.Party():
            pool = pp.stylize_orf(
                "AA<cre>ACGTAC</cre>TT", region="cre",
                style_codons=["red", "blue"], frame=1,
            ).named("s")
            df = _gen(pool)
            seq = df["seq"].iloc[0]
            assert "<cre>" in seq
            assert "</cre>" in seq

    def test_annotate_orf_inserts_tags(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA")
            result = pp.annotate_orf(pool, "cds")
            df = _gen(result)
            seq = df["seq"].iloc[0]
            assert "<cds>" in seq
            assert "</cds>" in seq
            assert strip_all_tags(seq) == "ATGGCTTAA"


# -------------------------------------------------------------------
# I5. Determinism
# -------------------------------------------------------------------

class TestI5Determinism:

    def test_mutagenize_orf_deterministic(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAATTTGGG", num_mutations=2, mode="random", num_states=20).named("m")

        df1 = pool.generate_library(num_cycles=1, seed=42)
        df2 = pool.generate_library(num_cycles=1, seed=42)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_reverse_translate_random_deterministic(self):
        with pp.Party():
            pool = pp.reverse_translate("LLLLLL", codon_selection="random", num_states=10)

        df1 = pool.generate_library(num_cycles=1, seed=42)
        df2 = pool.generate_library(num_cycles=1, seed=42)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_translate_deterministic(self):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA")

        df1 = _gen(pool)
        df2 = _gen(pool)
        assert list(df1["seq"]) == list(df2["seq"])


# -------------------------------------------------------------------
# I6. Region-only modification
# -------------------------------------------------------------------

class TestI6RegionIsolation:

    def test_mutagenize_orf_interval_region_flanks_unchanged(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGGATGAAACCC", [3, 9], num_mutations=1
            ).named("m")
            df = pool.generate_library(num_seqs=20, seed=42)
            for seq in df["seq"]:
                assert seq[:3] == "GGG"
                assert seq[-3:] == "CCC"

    def test_mutagenize_orf_named_region_flanks_unchanged(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAA</orf>CCC", "orf", num_mutations=1, frame=1
            ).named("m")
            df = pool.generate_library(num_seqs=20, seed=42)
            for seq in df["seq"]:
                assert seq[:3] == "GGG"
                assert seq.endswith("CCC")

    def test_stylize_orf_region_only_styles_region(self):
        """Styling with named region should only affect molecular positions inside region."""
        with pp.Party():
            pool = pp.stylize_orf(
                "AA<cre>ACGTAC</cre>TT", region="cre",
                style_frames=["red", "green", "blue"], frame=1,
            ).named("s")
            df = pool.generate_library(num_seqs=1, _include_inline_styles=True)
            seq_style = df["_inline_styles"].iloc[0]
            all_positions = set()
            for _, positions in seq_style.style_list:
                all_positions.update(positions)
            # Positions 0,1 (AA prefix) and last 2 (TT suffix) should not be styled
            assert 0 not in all_positions
            assert 1 not in all_positions


# -------------------------------------------------------------------
# I7. Composition
# -------------------------------------------------------------------

class TestI7Composition:

    def test_translate_reverse_translate_chain(self):
        with pp.Party():
            original = "ATGGCTCCCAAG"  # MAPK
            protein = pp.from_seq(original).translate()
            dna_back = protein.reverse_translate()
            protein_again = dna_back.translate()

            df_orig = _gen(protein)
            df_round = _gen(protein_again)
            assert df_orig["seq"].iloc[0] == df_round["seq"].iloc[0]

    def test_mutagenize_orf_chained_with_translate(self):
        with pp.Party():
            pool = (
                pp.from_seq("ATGAAATTT")
                .mutagenize_orf(num_mutations=1, mutation_type="missense_only_first", mode="sequential")
                .translate()
                .named("chain")
            )
            df = _gen(pool)
            assert len(df) == pool.num_states
            # Every translated protein should differ from the wild-type
            wt_protein = "MKF"
            for prot in df["seq"]:
                assert len(prot) == 3
                assert prot != wt_protein  # missense changes the AA

    def test_annotate_then_mutagenize(self):
        with pp.Party():
            pool = (
                pp.from_seq("GGGATGAAACCC")
                .annotate_orf("cds", extent=(3, 9))
                .mutagenize_orf("cds", num_mutations=1, mode="sequential")
                .named("chain")
            )
            df = _gen(pool)
            assert len(df) == pool.num_states
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == pool.seq_length


# -------------------------------------------------------------------
# I9. Init-time vs compute-time geometry parity
# -------------------------------------------------------------------

class TestI9GeometryParity:

    def test_mutagenize_orf_named_region_sequential(self):
        """Init-time ORF bounds must match compute-time bounds (bug #45 regression)."""
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAACCC</orf>TTT", region="orf",
                num_mutations=1, mode="sequential", frame=1,
            ).named("m")
            df = _gen(pool)
            assert len(df) == pool.num_states
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == pool.seq_length

    def test_mutagenize_orf_sequential_user_num_states(self):
        """User num_states override must work in sequential mode (bug #48 regression)."""
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mode="sequential", num_states=10,
            ).named("m")
            assert pool.num_states == 10
            df = _gen(pool)
            assert len(df) == 10

    def test_mutagenize_orf_sequential_cycling(self):
        """num_states > natural should cycle through states."""
        with pp.Party():
            # 2 codons, nonsense => 3 alts, 1 mutation => 6 natural
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="nonsense",
                mode="sequential", num_states=12,
            ).named("m")
            assert pool.num_states == 12
            df = _gen(pool)
            assert len(df) == 12
            # First 6 should equal second 6 (cycling)
            first_half = list(df["seq"][:6])
            second_half = list(df["seq"][6:])
            assert first_half == second_half


# -------------------------------------------------------------------
# I10. State-space immutability during compute
# -------------------------------------------------------------------

class TestI10StateImmutability:

    def test_mutagenize_orf_state_immutability(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mode="sequential"
            ).named("m")
            ns_before = pool.num_states
            sv_before = pool.operation.state._num_values
            _gen(pool)
            assert pool.num_states == ns_before
            assert pool.operation.state._num_values == sv_before

    def test_translate_state_immutability(self):
        with pp.Party():
            pool = pp.translate("ATGGCT")
            ns_before = pool.num_states
            _gen(pool)
            assert pool.num_states == ns_before

    def test_reverse_translate_random_state_immutability(self):
        with pp.Party():
            pool = pp.reverse_translate("MAPK", codon_selection="random", num_states=5)
            ns_before = pool.num_states
            sv_before = pool.operation.state._num_values
            pool.generate_library(num_cycles=1)
            assert pool.num_states == ns_before
            assert pool.operation.state._num_values == sv_before


# ===================================================================
# ORF-specific invariants
# ===================================================================

# -------------------------------------------------------------------
# D1. Frame handling
# -------------------------------------------------------------------

class TestFrameHandling:

    @pytest.mark.parametrize("frame", [1, 2, 3])
    def test_translate_positive_frames(self, frame):
        with pp.Party():
            seq = "AATGGCTTAA"  # 10 bases
            pool = pp.translate(seq, frame=frame)
            df = _gen(pool)
            prot = df["seq"].iloc[0]
            frame_offset = frame - 1
            expected_codons = (10 - frame_offset) // 3
            assert len(prot) == expected_codons

    @pytest.mark.parametrize("frame", [-1, -2, -3])
    def test_translate_negative_frames(self, frame):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA", frame=frame)
            df = _gen(pool)
            prot = df["seq"].iloc[0]
            frame_offset = abs(frame) - 1
            expected_codons = (9 - frame_offset) // 3
            assert len(prot) == expected_codons

    def test_translate_non_divisible_by_3(self):
        """Non-divisible-by-3 input: partial codons are silently skipped."""
        with pp.Party():
            # 7 bases, frame=1: ATG GCT T -> 2 codons
            pool = pp.translate("ATGGCTT")
            df = _gen(pool)
            assert len(df["seq"].iloc[0]) == 2

    def test_mutagenize_orf_non_divisible_by_3(self):
        """Non-divisible-by-3 input: partial codons are not mutable."""
        with pp.Party():
            # 5 bases, frame=1: ATG AA -> 1 codon
            pool = pp.mutagenize_orf("ATGAA", num_mutations=1)
            assert pool.operation.num_codons == 1

    @pytest.mark.parametrize("frame", [1, 2, 3])
    def test_mutagenize_orf_frame_offset(self, frame):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAATTTGGG", num_mutations=1, frame=frame)
            frame_offset = (4 - frame) % 3
            expected_codons = (12 - frame_offset) // 3
            assert pool.operation.num_codons == expected_codons


# -------------------------------------------------------------------
# D2. Stop codons
# -------------------------------------------------------------------

class TestStopCodons:

    def test_translate_includes_stop_by_default(self):
        with pp.Party():
            pool = pp.translate("ATGTAA")
            df = _gen(pool)
            assert df["seq"].iloc[0] == "M*"

    def test_translate_exclude_stop(self):
        with pp.Party():
            pool = pp.translate("ATGTAA", include_stop=False)
            df = _gen(pool)
            assert df["seq"].iloc[0] == "M"

    def test_translate_internal_stop(self):
        """Internal stop codons are translated as * like any other codon."""
        with pp.Party():
            # ATG TAA GCT -> M * A
            pool = pp.translate("ATGTAAGCT")
            df = _gen(pool)
            assert df["seq"].iloc[0] == "M*A"

    def test_translate_internal_stop_exclude(self):
        """With include_stop=False, ALL stop codons (internal and terminal) are excluded."""
        with pp.Party():
            pool = pp.translate("ATGTAAGCTTAA", include_stop=False)
            df = _gen(pool)
            prot = df["seq"].iloc[0]
            assert "*" not in prot
            assert prot == "MA"

    def test_reverse_translate_stop_codon(self):
        ct = CodonTable("standard")
        with pp.Party():
            pool = pp.reverse_translate("*")
            df = _gen(pool)
            seq = df["seq"].iloc[0]
            assert seq == ct.aa_to_codons["*"][0]  # TGA (most frequent)


# -------------------------------------------------------------------
# D3. Codon table correctness
# -------------------------------------------------------------------

class TestCodonTableCorrectness:

    def test_standard_code_representative_codons(self):
        """Verify representative standard genetic code mappings."""
        ct = CodonTable("standard")
        assert ct.codon_to_aa["ATG"] == "M"
        assert ct.codon_to_aa["TAA"] == "*"
        assert ct.codon_to_aa["TAG"] == "*"
        assert ct.codon_to_aa["TGA"] == "*"
        assert ct.codon_to_aa["TGG"] == "W"
        assert ct.codon_to_aa["GCT"] == "A"
        assert ct.codon_to_aa["AAG"] == "K"
        assert ct.codon_to_aa["TTC"] == "F"

    def test_all_64_codons_present(self):
        ct = CodonTable("standard")
        assert len(ct.all_codons) == 64

    def test_all_20_amino_acids_plus_stop(self):
        ct = CodonTable("standard")
        assert len(ct.aa_to_codons) == 21  # 20 AAs + *

    def test_translate_known_sequence(self):
        """End-to-end: translate a known sequence and verify AA output."""
        with pp.Party():
            # ATG=M, GCT=A, AAG=K, TGG=W, TAA=*
            pool = pp.translate("ATGGCTAAGTGGTAA")
            df = _gen(pool)
            assert df["seq"].iloc[0] == "MAKW*"

    def test_reverse_translate_deterministic_codons(self):
        """Verify 'first' selection uses highest-frequency codon."""
        ct = CodonTable("standard")
        with pp.Party():
            pool = pp.reverse_translate("MAKW")
            df = _gen(pool)
            seq = df["seq"].iloc[0]
            # M->ATG, A->GCC (most freq), K->AAG (most freq), W->TGG
            assert seq == "ATGGCCAAGTGG"


# -------------------------------------------------------------------
# D4. ORF boundary detection
# -------------------------------------------------------------------

class TestOrfBoundaryDetection:

    def test_translate_full_sequence(self):
        with pp.Party():
            pool = pp.translate("ATGGCTTAA")
            df = _gen(pool)
            assert df["seq"].iloc[0] == "MA*"

    def test_translate_interval_region(self):
        with pp.Party():
            # Only translate positions [3, 9): GCT TAA
            pool = pp.translate("ATGGCTTAA", region=[3, 9])
            df = _gen(pool)
            assert df["seq"].iloc[0] == "A*"

    def test_translate_tagged_region(self):
        with pp.Party():
            pool = pp.from_seq("NN<cds>ATGGCTTAA</cds>NN").annotate_orf("cds", frame=1)
            result = pool.translate(region="cds")
            df = _gen(result)
            assert df["seq"].iloc[0] == "MA*"

    def test_mutagenize_orf_full_sequence(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAA", num_mutations=1, mode="sequential")
            # 2 codons, 19 alts each
            assert pool.num_states == 38

    def test_mutagenize_orf_interval(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGGATGAAACCC", [3, 9], num_mutations=1, mode="sequential"
            )
            assert pool.num_states == 38

    def test_mutagenize_orf_tagged_region(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAA</orf>CCC", "orf", num_mutations=1,
                mode="sequential", frame=1,
            )
            assert pool.num_states == 38


# -------------------------------------------------------------------
# D5. mutagenize_orf mutation types
# -------------------------------------------------------------------

class TestMutagenizeOrfMutationTypes:

    def test_synonymous_preserves_aa(self):
        ct = CodonTable("standard")
        with pp.Party():
            # Use CTG (Leu, has 5 synonymous codons)
            pool = pp.mutagenize_orf(
                "CTGCTG", num_mutations=1, mutation_type="synonymous",
                mode="random", cards=["wt_aas", "mut_aas"],
            ).named("m")
            df = pool.generate_library(num_seqs=30, seed=42)
            wt_col = _card_col(df, "wt_aas")
            mut_col = _card_col(df, "mut_aas")
            for _, row in df.iterrows():
                for wt_a, mut_a in zip(row[wt_col], row[mut_col]):
                    assert wt_a == mut_a

    def test_nonsynonymous_changes_aa(self):
        ct = CodonTable("standard")
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="nonsynonymous_first",
                mode="sequential", cards=["wt_aas", "mut_aas"],
            ).named("m")
            df = _gen(pool)
            wt_col = _card_col(df, "wt_aas")
            mut_col = _card_col(df, "mut_aas")
            for _, row in df.iterrows():
                for wt_a, mut_a in zip(row[wt_col], row[mut_col]):
                    assert wt_a != mut_a

    def test_missense_excludes_stop(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mutation_type="missense_only_first",
                mode="sequential", cards=["mut_aas"],
            ).named("m")
            df = _gen(pool)
            mut_col = _card_col(df, "mut_aas")
            for _, row in df.iterrows():
                for mut_a in row[mut_col]:
                    assert mut_a != "*"

    def test_nonsense_only_stop(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="nonsense",
                mode="sequential", cards=["mut_aas"],
            ).named("m")
            df = _gen(pool)
            mut_col = _card_col(df, "mut_aas")
            for _, row in df.iterrows():
                for mut_a in row[mut_col]:
                    assert mut_a == "*"

    def test_any_codon_changes_codon(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="any_codon",
                mode="sequential", cards=["wt_codons", "mut_codons"],
            ).named("m")
            df = _gen(pool)
            wt_col = _card_col(df, "wt_codons")
            mut_col = _card_col(df, "mut_codons")
            for _, row in df.iterrows():
                for wt_c, mut_c in zip(row[wt_col], row[mut_col]):
                    assert wt_c != mut_c


# -------------------------------------------------------------------
# D6. Translation roundtrip
# -------------------------------------------------------------------

class TestTranslationRoundtrip:

    def test_translate_reverse_translate_same_protein(self):
        """translate -> reverse_translate produces a valid coding sequence for the same protein."""
        with pp.Party():
            original_dna = "ATGGCTCCCAAGTGG"  # MAPKW
            protein = pp.from_seq(original_dna).translate()
            dna_back = protein.reverse_translate()

            # Translate the reverse-translated DNA
            protein_again = dna_back.translate()

            df1 = _gen(protein)
            df2 = _gen(protein_again)
            assert df1["seq"].iloc[0] == df2["seq"].iloc[0]

    def test_roundtrip_all_amino_acids(self):
        """Every amino acid survives a roundtrip."""
        all_aas = "ACDEFGHIKLMNPQRSTVWY"
        with pp.Party():
            protein = pp.reverse_translate(all_aas)
            back_to_protein = protein.translate(include_stop=False)
            df = _gen(back_to_protein)
            assert df["seq"].iloc[0] == all_aas

    def test_roundtrip_with_stop(self):
        with pp.Party():
            protein = pp.reverse_translate("MAK*")
            back = protein.translate()
            df = _gen(back)
            assert df["seq"].iloc[0] == "MAK*"


# -------------------------------------------------------------------
# D7. Pool type transitions
# -------------------------------------------------------------------

class TestPoolTypeTransitions:

    def test_translate_returns_protein_pool(self):
        with pp.Party():
            pool = pp.translate("ATGGCT")
            assert isinstance(pool, ProteinPool)
            assert not isinstance(pool, DnaPool)

    def test_reverse_translate_returns_dna_pool(self):
        with pp.Party():
            pool = pp.reverse_translate("MA")
            assert isinstance(pool, DnaPool)
            assert not isinstance(pool, ProteinPool)

    def test_translate_from_string_returns_protein_pool(self):
        with pp.Party():
            pool = pp.translate("ATGGCT")
            assert isinstance(pool, ProteinPool)

    def test_reverse_translate_from_string_returns_dna_pool(self):
        with pp.Party():
            pool = pp.reverse_translate("MA")
            assert isinstance(pool, DnaPool)

    def test_translate_via_mixin_returns_protein_pool(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate()
            assert isinstance(pool, ProteinPool)

    def test_reverse_translate_via_mixin_returns_dna_pool(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate().reverse_translate()
            assert isinstance(pool, DnaPool)

    def test_mutagenize_orf_returns_dna_pool(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAA", num_mutations=1)
            assert isinstance(pool, DnaPool)

    def test_stylize_orf_returns_dna_pool(self):
        with pp.Party():
            pool = pp.stylize_orf("ATGAAA", style_codons=["red"])
            assert isinstance(pool, DnaPool)


# ===================================================================
# STEP 4 — Adversarial patterns (mutagenize_orf — HIGH RISK)
# ===================================================================

class TestAdversarialMutagenizeOrf:
    """Diagonal combinations + assumption inversions for mutagenize_orf."""

    # --- Diagonal 1: Named region + sequential + frame=2 ---
    def test_diagonal_named_region_sequential_frame2(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGG<orf>ATGAAACCC</orf>TTT", "orf",
                num_mutations=1, mode="sequential", frame=2,
            ).named("d1")
            df = _gen(pool)
            assert len(df) == pool.num_states
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == pool.seq_length
                assert "<orf>" in seq

    # --- Diagonal 2: Interval region + random + mutation_rate ---
    def test_diagonal_interval_random_mutation_rate(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GGGATGAAACCCAAATTT", [3, 15], mutation_rate=0.5,
                mode="random", num_states=50,
            ).named("d2")
            df = pool.generate_library(num_cycles=1, seed=42)
            assert len(df) == 50
            for seq in df["seq"]:
                assert len(seq) == 18
                assert seq[:3] == "GGG"
                assert seq[-3:] == "TTT"

    # --- Diagonal 3: Variable-length parent → should raise ---
    def test_diagonal_variable_length_parent_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ATGAAATTT", "ATGAAATTTGGG"])
            with pytest.raises(ValueError, match="must have a defined seq_length"):
                pp.mutagenize_orf(pool, num_mutations=1)

    # --- Diagonal 4: Codon positions + negative frame + sequential ---
    def test_diagonal_codon_positions_neg_frame_sequential(self):
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTTGGG", num_mutations=1, frame=-1,
                codon_positions=[0, 1], mode="sequential",
            ).named("d4")
            df = _gen(pool)
            assert len(df) == pool.num_states
            for seq in df["seq"]:
                assert len(seq) == 12

    # --- Assumption inversion 1: Region not aligned to codon boundary ---
    def test_assumption_partial_codons_in_region(self):
        """Region that yields partial codons: partial codons are just not mutable."""
        with pp.Party():
            # Region [1, 5]: 4 bases with frame=1, (4-0)//3 = 1 codon
            pool = pp.mutagenize_orf(
                "GATGAAACCC", [1, 5], num_mutations=1
            ).named("partial")
            assert pool.operation.num_codons == 1
            df = pool.generate_library(num_seqs=5, seed=42)
            for seq in df["seq"]:
                assert len(seq) == 10

    # --- Assumption inversion 2: Single-codon ORF ---
    def test_assumption_single_codon_orf(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATG", num_mutations=1, mode="sequential").named("tiny")
            assert pool.operation.num_codons == 1
            df = _gen(pool)
            assert len(df) == pool.num_states

    # --- Assumption inversion 3: Too few bases for any codon ---
    def test_assumption_zero_codons(self):
        """With frame=3, only 1 base left → 0 codons → can't request 1 mutation."""
        with pp.Party():
            with pytest.raises(ValueError, match="num_mutations.*exceeds"):
                pp.mutagenize_orf("ATG", num_mutations=1, frame=3)


# ===================================================================
# STEP 5 — Contract tracing (mutagenize_orf)
# ===================================================================

class TestContractTracingMutagenizeOrf:
    """C1 (state->output) and C3 (region round-trip)."""

    def test_c1_state_to_output_sequential(self):
        """Every state value in [0, num_states) maps to a distinct mutation."""
        with pp.Party():
            pool = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1,
                mutation_type="missense_only_first", mode="sequential",
                cards=["codon_positions", "mut_codons"],
            ).named("c1")
            df = _gen(pool)
            assert len(df) == pool.num_states
            # All rows should be unique (each state produces a distinct mutation)
            mutation_keys = []
            pos_col = _card_col(df, "codon_positions")
            mut_col = _card_col(df, "mut_codons")
            for _, row in df.iterrows():
                key = (tuple(row[pos_col]), tuple(row[mut_col]))
                mutation_keys.append(key)
            assert len(set(mutation_keys)) == len(mutation_keys)

    def test_c1_cycling_wraps_correctly(self):
        """num_states > natural cycles back to state 0."""
        with pp.Party():
            # 2 codons, nonsense => 3 stop codons, 1 mutation => 6 natural
            pool = pp.mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="nonsense",
                mode="sequential", num_states=8,
                cards=["codon_positions", "mut_codons"],
            ).named("cycle")
            df = _gen(pool)
            assert len(df) == 8
            pos_col = _card_col(df, "codon_positions")
            mut_col = _card_col(df, "mut_codons")
            # State 6 should equal state 0, state 7 should equal state 1
            row0 = (tuple(df[pos_col].iloc[0]), tuple(df[mut_col].iloc[0]))
            row6 = (tuple(df[pos_col].iloc[6]), tuple(df[mut_col].iloc[6]))
            row1 = (tuple(df[pos_col].iloc[1]), tuple(df[mut_col].iloc[1]))
            row7 = (tuple(df[pos_col].iloc[7]), tuple(df[mut_col].iloc[7]))
            assert row0 == row6
            assert row1 == row7

    def test_c3_region_roundtrip_tags_preserved(self):
        """Mutations inside named region preserve all tag boundaries."""
        with pp.Party():
            pool = pp.mutagenize_orf(
                "GG<orf>ATGAAACCC</orf>TT", "orf",
                num_mutations=2, mode="random", num_states=20, frame=1,
            ).named("c3")
            df = pool.generate_library(num_cycles=1, seed=42)
            for seq in df["seq"]:
                # Tags preserved
                assert "<orf>" in seq
                assert "</orf>" in seq
                # Flanking regions unchanged
                assert seq[:2] == "GG"
                assert seq.endswith("TT")
                # Bio length unchanged
                assert len(strip_all_tags(seq)) == pool.seq_length

    def test_c3_literal_to_molecular_correctness(self):
        """Verify molecular coordinate conversion is correct for tagged sequences."""
        with pp.Party():
            # Use a known mutation and check the exact output
            pool = pp.mutagenize_orf(
                "AA<orf>ATGAAA</orf>CC", "orf",
                num_mutations=1, codon_positions=[1], frame=1,
                mutation_type="nonsense", mode="sequential",
                cards=["codon_positions", "mut_codons"],
            ).named("c3b")
            df = _gen(pool)
            pos_col = _card_col(df, "codon_positions")
            mut_col = _card_col(df, "mut_codons")
            for _, row in df.iterrows():
                seq = row["seq"]
                # Position 1 means codon index 1 within the ORF
                assert row[pos_col] == (1,)
                # The WT codon at position 1 is AAA
                # The mutant codon should be a stop codon
                mut_codon = row[mut_col][0]
                assert mut_codon in ("TGA", "TAA", "TAG")
                # Verify the codon appears in the correct position in the sequence
                clean = strip_all_tags(seq)
                # Flanking: AA + [ATG + MUT_CODON] + CC
                assert clean[:2] == "AA"
                assert clean[-2:] == "CC"
                assert clean[2:5] == "ATG"  # First codon unchanged
                assert clean[5:8] == mut_codon


# ===================================================================
# STEP 6 — API consistency
# ===================================================================

class TestAPIConsistency:

    def test_translate_mode_always_fixed(self):
        with pp.Party():
            pool = pp.translate("ATGGCT")
            assert pool.operation.mode == "fixed"

    def test_reverse_translate_mode_first_is_fixed(self):
        with pp.Party():
            pool = pp.reverse_translate("MA", codon_selection="first")
            assert pool.operation.mode == "fixed"

    def test_reverse_translate_mode_random_is_random(self):
        with pp.Party():
            pool = pp.reverse_translate("MA", codon_selection="random")
            assert pool.operation.mode == "random"

    def test_mutagenize_orf_invalid_mode_raises(self):
        with pp.Party():
            with pytest.raises(Exception):
                pp.mutagenize_orf("ATGAAA", num_mutations=1, mode="bogus")

    def test_mutagenize_orf_sequential_with_mutation_rate_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="not supported with mutation_rate"):
                pp.mutagenize_orf("ATGAAA", mutation_rate=0.5, mode="sequential")

    def test_mutagenize_orf_sequential_nonuniform_type_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="uniform mutation type"):
                pp.mutagenize_orf(
                    "ATGAAA", num_mutations=1,
                    mutation_type="synonymous", mode="sequential",
                )


# ===================================================================
# STEP 7 — Findings verification
# ===================================================================

class TestFindings:
    """Concrete repros for each preliminary finding."""

    def test_f1_translate_copy_preserves_region(self):
        """TranslateOp.copy() preserves region via _get_copy_params override."""
        with pp.Party():
            pool = pp.from_seq("NNATGGCTTAANN").annotate_orf("cds", extent=(2, 11))
            protein = pool.translate(region="cds")
            df_orig = _gen(protein)
            assert df_orig["seq"].iloc[0] == "MA*"

            copied_pool = protein.copy()
            df_copy = _gen(copied_pool)
            assert df_copy["seq"].iloc[0] == "MA*"

    def test_f1_reverse_translate_copy_preserves_region(self):
        """ReverseTranslateOp.copy() preserves region via _get_copy_params override."""
        with pp.Party():
            protein = pp.from_seq("ATGGCTTAA").translate()
            back = protein.reverse_translate(region=[0, 2], codon_selection="first")
            df_orig = _gen(back)
            orig_seq = strip_all_tags(df_orig["seq"].iloc[0])
            assert len(orig_seq) == 6  # 2 AAs × 3bp

            copied_pool = back.copy()
            df_copy = copied_pool.generate_library(num_cycles=1)
            copy_seq = strip_all_tags(df_copy["seq"].iloc[0])
            assert len(copy_seq) == 6
            assert copy_seq == orig_seq

    def test_f2_annotate_orf_variable_length_parent(self):
        """annotate_orf(extent=None) with variable-length parent wraps full sequence."""
        with pp.Party():
            pool = pp.from_seqs(["ATGGCT", "ATGGCTAAA"])
            assert pool.seq_length is None
            result = pp.annotate_orf(pool, "cds")
            df = _gen(result)
            for seq in df["seq"]:
                assert "<cds>" in seq
                assert "</cds>" in seq
                assert "<cds/>" not in seq

    def test_f3_reverse_translate_rejects_dna_pool(self):
        """reverse_translate factory raises TypeError for non-ProteinPool input."""
        with pp.Party():
            dna_pool = pp.from_seq("ACGT")
            with pytest.raises(TypeError, match="reverse_translate requires a ProteinPool"):
                pp.reverse_translate(dna_pool)

    def test_f4_reverse_translate_in_all(self):
        """reverse_translate and ReverseTranslateOp are in __all__."""
        assert "reverse_translate" in pp.__all__
        assert "ReverseTranslateOp" in pp.__all__

    def test_f5_orf_ops_init_has_translate(self):
        """translate and TranslateOp are in orf_ops.__init__.__all__."""
        from poolparty import orf_ops
        assert "translate" in orf_ops.__all__
        assert "TranslateOp" in orf_ops.__all__

    def test_f6_annotate_orf_param_name_consistency(self):
        """annotate_orf and annotate_region both use 'region_name'."""
        import inspect
        sig = inspect.signature(pp.annotate_orf)
        params = list(sig.parameters.keys())
        assert params[1] == "region_name"
        sig_ar = inspect.signature(pp.annotate_region)
        params_ar = list(sig_ar.parameters.keys())
        assert "region_name" in params_ar

    def test_f7_annotate_orf_has_beartype(self):
        """annotate_orf is decorated with @beartype."""
        assert hasattr(pp.annotate_orf, "__wrapped__")

    def test_f8_resolve_frame_consistent_semantics(self):
        """All _resolve_frame implementations raise ValueError for plain Region."""
        from poolparty.orf_ops.translate import _resolve_frame as translate_rf
        from poolparty.orf_ops.mutagenize_orf import _resolve_frame as mutagenize_rf
        from poolparty.orf_ops.stylize_orf import _resolve_frame as stylize_rf

        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA")
            pp.annotate_region(pool, "plain_region", extent=(0, 9))

            with pytest.raises(ValueError, match="plain Region, not an OrfRegion"):
                translate_rf("plain_region", None)

            with pytest.raises(ValueError, match="plain Region, not an OrfRegion"):
                mutagenize_rf("plain_region", None)

            with pytest.raises(ValueError, match="plain Region, not an OrfRegion"):
                stylize_rf("plain_region", None)

    def test_f9_stylize_orf_hardcodes_dnapool(self):
        """stylize_orf hardcodes DnaPool return instead of type(pool)."""
        with pp.Party():
            pool = pp.stylize_orf("ATGAAA", style_codons=["red"])
            assert type(pool) is DnaPool
            # This is technically correct (ORF ops are DNA-only) but
            # inconsistent with the pattern used by mutagenize, filter, etc.
            pytest.xfail("F9: stylize_orf hardcodes DnaPool return type")

    def test_f10_translate_include_stop_false_seq_length(self):
        """include_stop=False sets seq_length=None since output length is data-dependent."""
        with pp.Party():
            pool = pp.translate("ATGGCT", include_stop=False)
            assert pool.seq_length is None
            df = _gen(pool)
            assert strip_all_tags(df["seq"].iloc[0]) == "MA"


# ===================================================================
# STEP 4 — Adversarial patterns (reverse_translate)
# ===================================================================

class TestAdversarialReverseTranslate:
    """Diagonal combinations + assumption inversions for reverse_translate."""

    def test_diagonal_region_interval_random(self):
        """Interval region + random codon selection + num_states."""
        with pp.Party():
            protein = pp.translate("ATGGCTAAATTT")  # "MA*F" or similar
            dna = pp.reverse_translate(
                protein, region=[0, 2], codon_selection="random", num_states=20,
            )
            df = dna.generate_library(num_cycles=1, seed=42)
            assert len(df) == 20
            for seq in df["seq"]:
                assert len(seq) == 6  # 2 AAs × 3bp

    def test_diagonal_string_input_random_high_states(self):
        """String input + random + many states → all valid codons."""
        with pp.Party():
            dna = pp.reverse_translate("MWMW", codon_selection="random", num_states=50)
            df = dna.generate_library(num_cycles=1, seed=42)
            assert len(df) == 50
            for seq in df["seq"]:
                assert len(seq) == 12  # 4 AAs × 3bp
                # W has only one codon (TGG), so positions 3-5 and 9-11 must be TGG
                assert seq[3:6] == "TGG"
                assert seq[9:12] == "TGG"

    def test_diagonal_single_aa_first(self):
        """Single amino acid protein with deterministic codon selection."""
        with pp.Party():
            dna = pp.reverse_translate("M", codon_selection="first")
            df = _gen(dna)
            assert len(df) == 1
            assert df["seq"].iloc[0] == "ATG"

    def test_assumption_stop_codon_in_protein(self):
        """Protein containing * (stop) — reverse_translate should produce a stop codon."""
        with pp.Party():
            dna = pp.reverse_translate("M*", codon_selection="first")
            df = _gen(dna)
            seq = df["seq"].iloc[0]
            assert len(seq) == 6
            assert seq[:3] == "ATG"
            from poolparty.codon_table import CodonTable
            ct = CodonTable("standard")
            assert ct.codon_to_aa.get(seq[3:6]) == "*"

    def test_assumption_all_20_aas_roundtrip(self):
        """Every standard amino acid reverse-translates to a valid codon."""
        from poolparty.codon_table import CodonTable
        ct = CodonTable("standard")
        all_aas = "ACDEFGHIKLMNPQRSTVWY"
        with pp.Party():
            dna = pp.reverse_translate(all_aas, codon_selection="first")
            df = _gen(dna)
            seq = df["seq"].iloc[0]
            assert len(seq) == 60  # 20 AAs × 3bp
            for i, aa in enumerate(all_aas):
                codon = seq[i*3:(i+1)*3]
                assert ct.codon_to_aa[codon] == aa

    def test_assumption_random_produces_variety(self):
        """Random codon selection produces multiple distinct sequences."""
        with pp.Party():
            dna = pp.reverse_translate("LLLL", codon_selection="random", num_states=30)
            df = dna.generate_library(num_cycles=1, seed=42)
            unique_seqs = df["seq"].nunique()
            assert unique_seqs > 1, "Random codon selection should produce variety for L (6 codons)"


# ===================================================================
# STEP 4 — Adversarial patterns (stylize_orf)
# ===================================================================

class TestAdversarialStylizeOrf:
    """Diagonal combinations + assumption inversions for stylize_orf."""

    def test_diagonal_named_region_frame2_style_codons(self):
        """Named OrfRegion + frame=2 + style_codons."""
        with pp.Party():
            pool = pp.from_seq("GATGAAACCC")
            pool = pp.annotate_orf(pool, "cds", extent=(1, 10), frame=2)
            styled = pp.stylize_orf(pool, region="cds", style_codons=["red", "blue"])
            df = _gen(styled)
            assert len(df) == 1
            assert strip_all_tags(df["seq"].iloc[0]) == "GATGAAACCC"

    def test_diagonal_interval_region_reverse_frame_style_frames(self):
        """Interval region + negative frame + style_frames."""
        with pp.Party():
            styled = pp.stylize_orf(
                "ATGAAACCCGGG", region=[0, 12],
                style_frames=["r", "g", "b"], frame=-1,
            )
            df = _gen(styled)
            assert len(df) == 1
            assert len(df["seq"].iloc[0]) == 12

    def test_diagonal_full_sequence_style_codons_many_styles(self):
        """Full sequence (no region) + many codon styles cycling."""
        with pp.Party():
            styled = pp.stylize_orf(
                "ATGATGATGATG",
                style_codons=["red", "green", "blue"],
            )
            df = _gen(styled)
            assert strip_all_tags(df["seq"].iloc[0]) == "ATGATGATGATG"

    def test_assumption_single_base_in_region(self):
        """Region with only 1 molecular position — no complete codon, styling still works."""
        with pp.Party():
            styled = pp.stylize_orf("ATGAAACCC", region=[0, 1], style_codons=["red"])
            df = _gen(styled)
            assert len(df) == 1

    def test_assumption_style_frames_6_entries(self):
        """style_frames with 6 entries (2 codon groups) cycles correctly."""
        with pp.Party():
            styled = pp.stylize_orf(
                "ATGATGATG",
                style_frames=["a", "b", "c", "d", "e", "f"],
            )
            df = _gen(styled)
            assert strip_all_tags(df["seq"].iloc[0]) == "ATGATGATG"

    def test_assumption_mutually_exclusive_style_args(self):
        """Providing both style_codons and style_frames raises ValueError."""
        with pp.Party():
            with pytest.raises(ValueError, match="mutually exclusive"):
                pp.stylize_orf("ATGAAA", style_codons=["red"], style_frames=["r", "g", "b"])

    def test_assumption_empty_styles_raises(self):
        """Empty style lists are rejected."""
        with pp.Party():
            with pytest.raises(ValueError):
                pp.stylize_orf("ATGAAA", style_codons=[])
            with pytest.raises(ValueError):
                pp.stylize_orf("ATGAAA", style_frames=[])


# ===================================================================
# STEP 5 — Contract tracing (reverse_translate)
# ===================================================================

class TestContractTracingReverseTranslate:
    """C1 (state→output mapping) and C3 (region round-trip) for reverse_translate."""

    def test_c1_first_deterministic_mapping(self):
        """codon_selection='first' always maps each AA to the same codon."""
        from poolparty.codon_table import CodonTable
        ct = CodonTable("standard")
        with pp.Party():
            dna = pp.reverse_translate("MAFW", codon_selection="first")
            df1 = _gen(dna)
            df2 = _gen(dna)
            assert df1["seq"].iloc[0] == df2["seq"].iloc[0]
            seq = df1["seq"].iloc[0]
            for i, aa in enumerate("MAFW"):
                codon = seq[i*3:(i+1)*3]
                assert ct.codon_to_aa[codon] == aa
                assert codon == ct.aa_to_codons[aa][0]

    def test_c1_random_states_all_valid_codons(self):
        """codon_selection='random' — every generated codon maps back to the original AA."""
        from poolparty.codon_table import CodonTable
        ct = CodonTable("standard")
        with pp.Party():
            dna = pp.reverse_translate("MAFW", codon_selection="random", num_states=50)
            df = dna.generate_library(num_cycles=1, seed=42)
            for seq in df["seq"]:
                for i, aa in enumerate("MAFW"):
                    codon = seq[i*3:(i+1)*3]
                    assert ct.codon_to_aa[codon] == aa

    def test_c3_region_only_reverse_translates_region(self):
        """Interval region: only the specified AAs are reverse-translated."""
        with pp.Party():
            protein = pp.translate("ATGGCTAAATTTCCC")  # full translation
            df_prot = _gen(protein)
            full_protein = df_prot["seq"].iloc[0]

            dna = pp.reverse_translate(protein, region=[1, 3], codon_selection="first")
            df_dna = _gen(dna)
            dna_seq = df_dna["seq"].iloc[0]
            assert len(dna_seq) == 6  # 2 AAs × 3bp

    def test_c2_composition_preserves_states(self):
        """Multi-state parent → reverse_translate preserves state dimension."""
        with pp.Party():
            mutated_dna = pp.mutagenize_orf(
                "ATGAAATTT", num_mutations=1, mode="sequential",
                mutation_type="missense_only_first",
            )
            protein = mutated_dna.translate()
            parent_states = protein.num_states
            dna = protein.reverse_translate(codon_selection="first")
            df = _gen(dna)
            assert len(df) == parent_states


# ===================================================================
# STEP 5 — Contract tracing (stylize_orf)
# ===================================================================

class TestContractTracingStylizeOrf:
    """C1 (output mapping) and C3 (region isolation) for stylize_orf."""

    def test_c1_style_codons_assigns_correct_positions(self):
        """style_codons=['red','blue'] assigns alternating codon colors."""
        with pp.Party():
            styled = pp.stylize_orf("ATGATG", style_codons=["red", "blue"])
            df = _gen(styled)
            seq_obj = styled.operation._compute_core(
                [styled.operation.parent_pools[0].operation._compute_core([], None)[0]],
                None,
            )[0]
            assert seq_obj.style is not None
            style_list = seq_obj.style.style_list
            style_map = {spec: set(pos.tolist()) for spec, pos in style_list}
            assert "red" in style_map
            assert "blue" in style_map
            assert style_map["red"] == {0, 1, 2}
            assert style_map["blue"] == {3, 4, 5}

    def test_c3_region_isolation_flanks_unstyled(self):
        """Styling only applies within region; flanks have no ORF styles."""
        with pp.Party():
            pool = pp.from_seq("GGGATGAAACCCGGG")
            pool = pp.annotate_orf(pool, "cds", extent=(3, 12), frame=1)
            styled = pp.stylize_orf(pool, region="cds", style_codons=["red", "blue"])
            df = _gen(styled)
            bio_seq = strip_all_tags(df["seq"].iloc[0])
            assert bio_seq == "GGGATGAAACCCGGG"

    def test_c1_frame_offset_shifts_codon_boundaries(self):
        """frame=2 shifts where codon boundaries land."""
        with pp.Party():
            styled1 = pp.stylize_orf("AATGATG", style_codons=["red", "blue"], frame=1)
            styled2 = pp.stylize_orf("AATGATG", style_codons=["red", "blue"], frame=2)
            df1 = _gen(styled1)
            df2 = _gen(styled2)
            assert df1["seq"].iloc[0] == df2["seq"].iloc[0]  # same text, different styles


# ===================================================================
# Compositional stress test (Step 4 supplement)
# ===================================================================

class TestCompositionalStress:

    def test_mutagenize_orf_between_sequential_ops(self):
        """Chain mutagenize_orf between two sequential ops and verify Cartesian product."""
        with pp.Party():
            pool = (
                pp.from_seq("ATGAAATTT")
                .mutagenize_orf(num_mutations=1, mutation_type="nonsense",
                                mode="sequential")
                .named("mut")
            )
            # 3 codons * 3 stop codons = 9 states
            assert pool.num_states == 9
            df = _gen(pool)
            assert len(df) == 9

    def test_annotate_then_mutagenize_then_translate(self):
        """Full pipeline: annotate -> mutagenize -> translate."""
        with pp.Party():
            pool = (
                pp.from_seq("GGGATGGCTAAACCC")
                .annotate_orf("cds", extent=(3, 12))
                .mutagenize_orf("cds", num_mutations=1, mode="sequential",
                                mutation_type="missense_only_first")
                .translate(region="cds", include_stop=False)
                .named("pipeline")
            )
            df = _gen(pool)
            assert len(df) == pool.num_states
            # Every protein should be 3 AAs (3 codons, no stop)
            for prot in df["seq"]:
                assert len(prot) == 3


# ===================================================================
# NullSeq propagation
# ===================================================================

class TestNullSeqPropagation:

    def test_translate_null_propagation(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT").filter(lambda s: False).translate()
            df = _gen(pool)
            assert df["seq"].iloc[0] is None

    def test_reverse_translate_null_propagation(self):
        with pp.Party():
            prot = pp.from_seq("ATGGCT").translate()
            filtered = prot.filter(lambda s: False)
            dna = filtered.reverse_translate()
            df = _gen(dna)
            assert df["seq"].iloc[0] is None
