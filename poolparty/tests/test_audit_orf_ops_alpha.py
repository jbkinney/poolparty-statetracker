"""Audit tests for ORF operations (C8).

Follows `.cursor/rules/operation_audit.mdc` Steps 1-7:
- Step 2: smoke + runtime forwarding parity
- Step 3: I1-I10 + ORF-specific invariants
- Step 4: adversarial diagonals for high-risk ops
- Step 5: contract tracing for highest-risk ops

Scope:
- translate
- reverse_translate
- mutagenize_orf
- stylize_orf
- annotate_orf
"""

from __future__ import annotations

import pytest

import poolparty as pp
from poolparty.codon_table import CodonTable
from poolparty.utils.parsing_utils import strip_all_tags


def _clean_len(seq: str) -> int:
    return len(strip_all_tags(seq))


def _card_col(df, suffix: str) -> str:
    matches = [c for c in df.columns if c.split(".")[-1] == suffix]
    assert matches, f"missing card column for suffix '{suffix}' in columns: {list(df.columns)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Step 2: smoke tests and runtime forwarding parity
# ---------------------------------------------------------------------------


class TestStep2Smoke:
    """Smoke every ORF factory inside Party and generate minimal output."""

    def test_translate_smoke(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "MA*"

    def test_reverse_translate_smoke(self):
        with pp.Party():
            pool = pp.reverse_translate("MA", codon_selection="first")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert _clean_len(df["seq"].iloc[0]) == 6

    def test_mutagenize_orf_smoke(self):
        with pp.Party():
            pool = pp.mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) > 0

    def test_stylize_orf_smoke(self):
        with pp.Party():
            pool = pp.stylize_orf("ATGAAATTT", style_codons=["red", "blue"])
        df = pool.generate_library(num_cycles=1, _include_inline_styles=True)
        assert len(df) == 1
        assert df["_inline_styles"].iloc[0] is not None

    def test_annotate_orf_smoke(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").annotate_orf("orf", frame=1)
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert "<orf>" in df["seq"].iloc[0]
        assert "</orf>" in df["seq"].iloc[0]


class TestStep2RuntimeForwarding:
    """Forwarding checks: runtime calls must succeed without TypeError."""

    def test_dna_mixin_translate_runtime_forwarding(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate(
                region=[0, 9],
                frame=1,
                include_stop=False,
                preserve_codon_styles=False,
                genetic_code="standard",
                iter_order=1.0,
                prefix="tr",
            )
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1

    def test_dna_mixin_mutagenize_orf_runtime_forwarding(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                region=[0, 9],
                num_mutations=1,
                mutation_type="nonsynonymous_first",
                codon_positions=[0, 2],
                style="bold",
                frame=1,
                prefix="mo",
                mode="sequential",
                num_states=2,
                iter_order=2.0,
                cards=["codon_positions", "wt_codons", "mut_codons"],
            )
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 2

    def test_dna_mixin_stylize_orf_runtime_forwarding(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").stylize_orf(
                region=[0, 9],
                style_frames=["red", "green", "blue"],
                frame=2,
                iter_order=1.0,
                prefix="sty",
            )
        df = pool.generate_library(num_cycles=1, _include_inline_styles=True)
        assert len(df) == 1
        assert df["_inline_styles"].iloc[0] is not None

    def test_dna_mixin_annotate_orf_runtime_forwarding(self):
        with pp.Party():
            pool = pp.from_seq("NNNATGAAATTTNNN").annotate_orf(
                "orf",
                extent=(3, 12),
                frame=1,
                style="cyan",
                iter_order=1.0,
                prefix="ann",
            )
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert "<orf>" in df["seq"].iloc[0]

    def test_protein_mixin_reverse_translate_runtime_forwarding(self):
        with pp.Party():
            protein = pp.from_seq("ATGGCTAAA").translate()
            dna = protein.reverse_translate(
                region=[0, 2],
                codon_selection="random",
                num_states=3,
                genetic_code="standard",
                iter_order=1.0,
                prefix="rt",
            )
        df = dna.generate_library(num_cycles=1, seed=42)
        assert len(df) == 3
        for seq in df["seq"]:
            assert _clean_len(seq) == 6


# ---------------------------------------------------------------------------
# Pre-known fixed bugs: confirm and skip from novel findings
# ---------------------------------------------------------------------------


class TestKnownFixedBugs:
    """Confirm #41, #45, #48 are fixed."""

    def test_bug_41_mutagenize_orf_mixin_cards_forwarded(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1,
                mode="sequential",
                cards=["codon_positions"],
            )
        df = pool.generate_library(num_cycles=1)
        card_col = _card_col(df, "codon_positions")
        assert card_col in df.columns

    def test_bug_45_sequential_named_region_geometry_parity(self):
        with pp.Party():
            pool = pp.from_seq("GGG<orf>ATGAAA</orf>CCC").mutagenize_orf(
                region="orf",
                num_mutations=1,
                mode="sequential",
                frame=1,
            )
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        assert len(df) > 0

    def test_bug_48_sequential_num_states_override_respected(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1,
                mode="sequential",
                num_states=3,
            )
        assert pool.num_states == 3
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Step 3: I1-I10 + ORF-specific invariants
# ---------------------------------------------------------------------------


class TestStep3CoreInvariants:
    """Core invariants with ORF-focused contracts."""

    def test_i1_output_length_matches_seq_length_all_ops(self):
        with pp.Party():
            translated = pp.from_seq("ATGGCTTAA").translate()
            back = pp.reverse_translate("MAK")
            mut = pp.from_seq("ATGAAATTT").mutagenize_orf(num_mutations=1, mode="sequential")
            sty = pp.from_seq("ATGAAATTT").stylize_orf(style_codons=["red", "blue"])
            ann = pp.from_seq("ATGAAATTT").annotate_orf("orf", frame=1)

        for pool in [translated, back, mut, sty, ann]:
            assert pool.seq_length is not None
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert _clean_len(seq) == pool.seq_length

    @pytest.mark.xfail(
        strict=True,
        reason="translate(include_stop=False) underreports seq_length when sequence has no stop codon",
    )
    def test_i1_translate_include_stop_false_reports_correct_seq_length(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate(include_stop=False)
        df = pool.generate_library(num_cycles=1)
        assert pool.seq_length == _clean_len(df["seq"].iloc[0]) == 2

    def test_i2_state_exhaustion_row_count_matches_num_states(self):
        with pp.Party():
            mut = pp.from_seq("ATGAAATTT").mutagenize_orf(num_mutations=1, mode="sequential")
            trans = pp.from_seq("ATGGCTTAA").translate()
            rev = pp.reverse_translate("MA", codon_selection="random", num_states=4)
            sty = pp.from_seq("ATGAAATTT").stylize_orf(style_codons=["red", "blue"])
            ann = pp.from_seq("ATGAAATTT").annotate_orf("orf", frame=1)

        for pool, kwargs in [
            (mut, {"num_cycles": 1}),
            (trans, {"num_cycles": 1}),
            (rev, {"num_cycles": 1, "seed": 42}),
            (sty, {"num_cycles": 1}),
            (ann, {"num_cycles": 1}),
        ]:
            df = pool.generate_library(**kwargs)
            assert len(df) == pool.num_states

    def test_i3_card_sequence_agreement_mutagenize_orf(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1,
                mode="sequential",
                cards=["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"],
            )
        df = pool.generate_library(num_cycles=1)
        codon_col = _card_col(df, "codon_positions")
        wt_c_col = _card_col(df, "wt_codons")
        mut_c_col = _card_col(df, "mut_codons")
        wt_a_col = _card_col(df, "wt_aas")
        mut_a_col = _card_col(df, "mut_aas")
        codon_table = CodonTable("standard")

        for _, row in df.iterrows():
            seq = strip_all_tags(row["seq"])
            for cp, wt_c, mut_c, wt_a, mut_a in zip(
                row[codon_col], row[wt_c_col], row[mut_c_col], row[wt_a_col], row[mut_a_col]
            ):
                start = cp * 3
                assert seq[start : start + 3] == mut_c
                assert wt_c != mut_c
                assert codon_table.codon_to_aa[wt_c] == wt_a
                assert codon_table.codon_to_aa[mut_c] == mut_a

    def test_i4_region_tags_preserved_when_expected(self):
        with pp.Party():
            mut = pp.from_seq("GGG<orf>ATGAAA</orf>CCC").mutagenize_orf(
                region="orf", num_mutations=1, mode="random", frame=1
            )
            sty = pp.from_seq("GGG<orf>ATGAAA</orf>CCC").stylize_orf(
                region="orf", style_codons=["red", "blue"], frame=1
            )

        df_mut = mut.generate_library(num_seqs=10, seed=42)
        for seq in df_mut["seq"]:
            assert "<orf>" in seq and "</orf>" in seq

        df_sty = sty.generate_library(num_cycles=1)
        assert "<orf>" in df_sty["seq"].iloc[0]
        assert "</orf>" in df_sty["seq"].iloc[0]

    def test_i5_random_determinism(self):
        results = []
        for _ in range(2):
            with pp.Party():
                m = pp.from_seq("ATGAAATTT").mutagenize_orf(num_mutations=1, mode="random")
                r = pp.reverse_translate("LLLL", codon_selection="random", num_states=10)
            df_m = m.generate_library(num_seqs=20, seed=123)
            df_r = r.generate_library(num_cycles=1, seed=123)
            results.append((df_m["seq"].tolist(), df_r["seq"].tolist()))
        assert results[0] == results[1]

    def test_i6_region_only_modification(self):
        with pp.Party():
            mut = pp.from_seq("GGG<orf>ATGAAA</orf>CCC").mutagenize_orf(
                region="orf", num_mutations=1, mode="random", frame=1
            )
            trans = pp.from_seq("NNN<orf2>ATGGCTTAA</orf2>NNN").translate(region="orf2", frame=1)

        df_mut = mut.generate_library(num_seqs=20, seed=42)
        for seq in df_mut["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:3] == "GGG"
            assert clean[9:] == "CCC"

        df_trans = trans.generate_library(num_cycles=1)
        assert df_trans["seq"].iloc[0] == "MA*"

    def test_i7_composition_state_count_and_seq_length(self):
        with pp.Party():
            src = pp.from_seqs(["ATGAAATTT", "ATGCCCTTT"], mode="sequential")
            mid = src.mutagenize_orf(num_mutations=1, mode="sequential", num_states=3)
            out = mid.translate()

        df = out.generate_library(num_cycles=1)
        assert len(df) == out.num_states
        assert out.num_states == 2 * 3
        assert out.seq_length == 3

    def test_i8_length_algebra_for_orf_transforms(self):
        with pp.Party():
            dna = pp.from_seq("ATGGCTTAA")
            protein = dna.translate(frame=1, include_stop=True)
            back = protein.reverse_translate(codon_selection="first")
        assert protein.seq_length == 3
        assert back.seq_length == 9
        df_back = back.generate_library(num_cycles=1)
        assert _clean_len(df_back["seq"].iloc[0]) == 9

    def test_i9_init_compute_geometry_parity_sequential_named_region(self):
        with pp.Party():
            tagged = pp.from_seq("GGGATGAAACCC").annotate_orf("orf", extent=(3, 9), frame=1)
            pool = tagged.mutagenize_orf(region="orf", num_mutations=1, mode="sequential")
        df = pool.generate_library(num_cycles=1)
        assert len(df) == pool.num_states
        for seq in df["seq"]:
            assert _clean_len(seq) == pool.seq_length

    def test_i10_state_space_immutability_during_compute(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(num_mutations=1, mode="sequential")
        ns_before = pool.num_states
        sv_before = pool.operation.state._num_values
        pool.generate_library(num_cycles=1)
        assert pool.num_states == ns_before
        assert pool.operation.state._num_values == sv_before


class TestOrfSpecificInvariants:
    """Additional ORF-specific invariants required by the audit task."""

    def test_frame_handling_offsets_and_non_divisible_by_three(self):
        with pp.Party():
            f1 = pp.from_seq("ATGGCTTAA").translate(frame=1)
            f2 = pp.from_seq("ATGGCTTAA").translate(frame=2)
            f3 = pp.from_seq("ATGGCTTAA").translate(frame=3)
            short = pp.from_seq("ATGAA").translate(frame=1)

        assert f1.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"
        assert f2.generate_library(num_cycles=1)["seq"].iloc[0] == "WL"
        assert f3.generate_library(num_cycles=1)["seq"].iloc[0] == "GL"
        # Non-divisible-by-3 input is handled gracefully (partial codon ignored).
        assert short.generate_library(num_cycles=1)["seq"].iloc[0] == "M"

    def test_stop_codons_internal_and_terminal_behavior(self):
        with pp.Party():
            seq = "ATGTAAGCTTAA"  # M * A *
            with_stop = pp.from_seq(seq).translate(include_stop=True)
            without_stop = pp.from_seq(seq).translate(include_stop=False)

        assert with_stop.generate_library(num_cycles=1)["seq"].iloc[0] == "M*A*"
        # include_stop=False removes stop codons in output stream.
        assert without_stop.generate_library(num_cycles=1)["seq"].iloc[0] == "MA"

    def test_codon_table_correctness_representative_codons(self):
        codon_table = CodonTable("standard")
        assert codon_table.codon_to_aa["ATG"] == "M"
        assert codon_table.codon_to_aa["TAA"] == "*"
        assert codon_table.codon_to_aa["TGA"] == "*"
        assert codon_table.codon_to_aa["GCT"] == "A"

    def test_orf_boundary_detection_tag_vs_interval_vs_full(self):
        with pp.Party():
            tagged = pp.from_seq("NNN<orf>ATGGCTTAA</orf>NNN").translate(region="orf", frame=1)
            interval = pp.from_seq("NNNATGGCTTAANNN").translate(region=[3, 12], frame=1)
            whole = pp.from_seq("ATGGCTTAA").translate(frame=1)
        assert tagged.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"
        assert interval.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"
        assert whole.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"

    def test_mutagenize_orf_mutation_types_sub_syn_non_syn(self):
        with pp.Party():
            sub = pp.mutagenize_orf(
                "CTGCTG",
                num_mutations=1,
                mutation_type="any_codon",
                mode="random",
                cards=["wt_codons", "mut_codons"],
            )
            syn = pp.mutagenize_orf(
                "CTGCTG",
                num_mutations=1,
                mutation_type="synonymous",
                mode="random",
                cards=["wt_aas", "mut_aas"],
            )
            nonsyn = pp.mutagenize_orf(
                "ATGAAATTT",
                num_mutations=1,
                mutation_type="missense_only_first",
                mode="sequential",
                cards=["wt_aas", "mut_aas"],
            )

        df_sub = sub.generate_library(num_seqs=20, seed=42)
        wt_codons_col = _card_col(df_sub, "wt_codons")
        mut_codons_col = _card_col(df_sub, "mut_codons")
        for _, row in df_sub.iterrows():
            for wt, mut in zip(row[wt_codons_col], row[mut_codons_col]):
                assert wt != mut

        df_syn = syn.generate_library(num_seqs=20, seed=42)
        wt_aas_col = _card_col(df_syn, "wt_aas")
        mut_aas_col = _card_col(df_syn, "mut_aas")
        for _, row in df_syn.iterrows():
            for wt, mut in zip(row[wt_aas_col], row[mut_aas_col]):
                assert wt == mut

        df_nonsyn = nonsyn.generate_library(num_cycles=1)
        wt_aas_col = _card_col(df_nonsyn, "wt_aas")
        mut_aas_col = _card_col(df_nonsyn, "mut_aas")
        for _, row in df_nonsyn.iterrows():
            for wt, mut in zip(row[wt_aas_col], row[mut_aas_col]):
                assert wt != mut
                assert mut != "*"

    def test_translation_roundtrip_valid_coding_for_same_protein(self):
        with pp.Party():
            protein = pp.from_seq("ATGGCTCCCAAG").translate()
            back_dna = protein.reverse_translate(codon_selection="random", num_states=8)
            translated_again = back_dna.translate()

        p0 = protein.generate_library(num_cycles=1)["seq"].iloc[0]
        p_df = translated_again.generate_library(num_cycles=1, seed=42)
        assert len(p_df) == 8
        assert set(p_df["seq"]) == {p0}

    def test_pool_type_transitions_translate_and_reverse_translate(self):
        with pp.Party():
            protein_pool = pp.from_seq("ATGGCTTAA").translate()
            dna_pool = protein_pool.reverse_translate(codon_selection="first")
        assert isinstance(protein_pool, pp.ProteinPool)
        assert isinstance(dna_pool, pp.DnaPool)
        assert protein_pool.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"
        assert dna_pool.generate_library(num_cycles=1)["seq"].iloc[0].startswith("ATG")


# ---------------------------------------------------------------------------
# Step 4 + 5: high-risk adversarial patterns and contract tracing
# ---------------------------------------------------------------------------


class TestHighRiskMutagenizeOrf:
    """High-risk op: mutagenize_orf adversarial + contract tracing."""

    def test_adversarial_diagonal_1_named_region_sequential_clipped_states(self):
        with pp.Party():
            pool = pp.from_seq("GGG<orf>ATGAAATTT</orf>CCC").mutagenize_orf(
                region="orf",
                num_mutations=1,
                mode="sequential",
                frame=1,
                num_states=5,
            )
        assert pool.num_states == 5
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 5
        for seq in df["seq"]:
            clean = strip_all_tags(seq)
            assert clean[:3] == "GGG"
            assert clean[12:] == "CCC"

    def test_adversarial_diagonal_2_reverse_frame_interval_random(self):
        with pp.Party():
            pool = pp.from_seq("CCCATGAAATTTGGG").mutagenize_orf(
                region=[3, 12],
                num_mutations=1,
                frame=-1,
                mode="random",
                num_states=12,
                cards=["codon_positions"],
            )
        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) == 12
        assert _card_col(df, "codon_positions") in df.columns

    def test_adversarial_diagonal_3_assumption_inversion_unknown_geometry_raises(self):
        with pp.Party():
            parent = pp.from_seqs(["ATGAAA", "ATGAAATTT"], mode="sequential")
            with pytest.raises(ValueError, match="parent_pool must have a defined seq_length"):
                parent.mutagenize_orf(num_mutations=1, mode="sequential")

    def test_contract_trace_c1_state_to_output_mapping(self):
        with pp.Party():
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(
                num_mutations=1,
                mode="sequential",
                cards=["codon_positions", "mut_codons"],
            )
        df = pool.generate_library(num_cycles=1)
        pos_col = _card_col(df, "codon_positions")
        mut_col = _card_col(df, "mut_codons")
        for _, row in df.iterrows():
            seq = row["seq"]
            for cp, mut_c in zip(row[pos_col], row[mut_col]):
                start = cp * 3
                assert seq[start : start + 3] == mut_c

    def test_contract_trace_c2_state_composition_cartesian(self):
        with pp.Party():
            src = pp.from_seqs(["ATGAAATTT", "ATGCCCTTT"], mode="sequential")
            mut = src.mutagenize_orf(num_mutations=1, mode="sequential", num_states=4)
        df = mut.generate_library(num_cycles=1)
        assert mut.num_states == 2 * 4
        assert len(df) == 8

    def test_contract_trace_c3_region_round_trip_and_tag_preservation(self):
        with pp.Party():
            pool = pp.from_seq("GGG<orf>ATGAAATTT</orf>CCC").mutagenize_orf(
                region="orf",
                num_mutations=1,
                mode="random",
                frame=1,
            )
        df = pool.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert "<orf>" in seq and "</orf>" in seq
            clean = strip_all_tags(seq)
            assert clean[:3] == "GGG"
            assert clean[12:] == "CCC"


class TestHighRiskTranslate:
    """High-risk op: translate adversarial + contract tracing."""

    def test_adversarial_diagonal_1_tagged_region_frame_shift(self):
        with pp.Party():
            pool = pp.from_seq("NNN<orf>ATGGCTTAA</orf>NNN").translate(region="orf", frame=2)
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "WL"

    def test_adversarial_diagonal_2_reverse_frame_with_interval(self):
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate(region=[0, 6], frame=-1)
        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "SH"

    def test_adversarial_diagonal_3_assumption_inversion_iupac_ambiguity_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="cannot handle IUPAC ambiguity codes"):
                pp.from_seq("ATGNNNTAA").translate(frame=1).generate_library(num_cycles=1)

    def test_contract_trace_c1_frame_to_output_mapping(self):
        with pp.Party():
            f1 = pp.from_seq("ATGGCTTAA").translate(frame=1)
            f2 = pp.from_seq("ATGGCTTAA").translate(frame=2)
            f3 = pp.from_seq("ATGGCTTAA").translate(frame=3)
        assert f1.generate_library(num_cycles=1)["seq"].iloc[0] == "MA*"
        assert f2.generate_library(num_cycles=1)["seq"].iloc[0] == "WL"
        assert f3.generate_library(num_cycles=1)["seq"].iloc[0] == "GL"

    def test_contract_trace_c2_state_composition_passthrough(self):
        with pp.Party():
            src = pp.from_seqs(["ATGGCTTAA", "ATGAAATGA"], mode="sequential")
            translated = src.translate()
        df = translated.generate_library(num_cycles=1)
        assert translated.num_states == 2
        assert len(df) == 2
        assert set(df["seq"]) == {"MA*", "MK*"}

    def test_contract_trace_c3_region_roundtrip(self):
        with pp.Party():
            pool = pp.from_seq("AAA<orf>ATGGCTTAA</orf>TTT").translate(region="orf", frame=1)
        df = pool.generate_library(num_cycles=1)
        # Only ORF content is translated; flanks/tags do not leak into protein output.
        assert df["seq"].iloc[0] == "MA*"
