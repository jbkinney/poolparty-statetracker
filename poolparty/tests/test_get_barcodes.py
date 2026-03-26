"""Tests for get_barcodes operation."""

import pytest

import poolparty as pp
from poolparty.base_ops.get_barcodes import (
    GetBarcodesOp,
    _check_gc_content,
    _check_homopolymer,
    _edit_distance,
    _hamming_distance,
    get_barcodes,
)


# --- Helper function tests ---


class TestDistanceFunctions:
    def test_hamming_identical(self):
        assert _hamming_distance("ACGT", "ACGT") == 0

    def test_hamming_all_different(self):
        assert _hamming_distance("AAAA", "TTTT") == 4

    def test_hamming_partial(self):
        assert _hamming_distance("ACGT", "ACTT") == 1

    def test_edit_distance_identical(self):
        assert _edit_distance("ACGT", "ACGT") == 0

    def test_edit_distance_insertion(self):
        assert _edit_distance("ACG", "ACGT") == 1

    def test_edit_distance_deletion(self):
        assert _edit_distance("ACGT", "ACG") == 1

    def test_edit_distance_substitution(self):
        assert _edit_distance("ACGT", "ACTT") == 1

    def test_edit_distance_empty(self):
        assert _edit_distance("", "ACGT") == 4
        assert _edit_distance("ACGT", "") == 4

    def test_edit_distance_different_lengths(self):
        assert _edit_distance("ACGT", "ACGTTT") == 2


class TestHomopolymerCheck:
    def test_no_homopolymer(self):
        assert _check_homopolymer("ACGT", 2) is True

    def test_homopolymer_at_limit(self):
        assert _check_homopolymer("AACGT", 2) is True

    def test_homopolymer_exceeds(self):
        assert _check_homopolymer("AAACGT", 2) is False

    def test_all_same(self):
        assert _check_homopolymer("AAAA", 3) is False

    def test_empty(self):
        assert _check_homopolymer("", 1) is True


class TestGCCheck:
    def test_within_range(self):
        assert _check_gc_content("ACGT", 0.4, 0.6) is True

    def test_below_range(self):
        assert _check_gc_content("AATT", 0.4, 0.6) is False

    def test_above_range(self):
        assert _check_gc_content("CCGG", 0.0, 0.4) is False

    def test_at_boundary(self):
        assert _check_gc_content("ACGT", 0.5, 0.5) is True


# --- GetBarcodesOp tests ---


class TestGetBarcodesBasic:
    def test_creates_correct_num_barcodes(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=10, length=8, seed=42)
            assert pool.num_states == 10

    def test_barcode_length(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=5, length=12, seed=42)
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert len(seq) == 12

    def test_severed_dag(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=5, length=8, seed=42)
            assert pool.parents == []

    def test_all_unique(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=50, length=8, seed=42)
            df = pool.generate_library(num_cycles=1)
            assert df["seq"].nunique() == 50

    def test_only_dna_bases(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=20, length=8, seed=42)
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert all(c in "ACGT" for c in seq)


class TestGetBarcodesConstraints:
    def test_min_edit_distance(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=20, length=8, min_edit_distance=3, seed=42
            )
            df = pool.generate_library(num_cycles=1)
            seqs = list(df["seq"])
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    assert _edit_distance(seqs[i], seqs[j]) >= 3

    def test_min_hamming_distance(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=20, length=8, min_hamming_distance=3, seed=42
            )
            df = pool.generate_library(num_cycles=1)
            seqs = list(df["seq"])
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    assert _hamming_distance(seqs[i], seqs[j]) >= 3

    def test_gc_range(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=20, length=10, gc_range=(0.3, 0.7), seed=42
            )
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                gc = sum(1 for c in seq if c in "GC") / len(seq)
                assert 0.3 <= gc <= 0.7

    def test_max_homopolymer(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=20, length=10, max_homopolymer=2, seed=42
            )
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert _check_homopolymer(seq, 2)

    def test_avoid_sequences(self):
        avoid = ["ACGTACGT"]
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=10,
                length=8,
                avoid_sequences=avoid,
                avoid_min_distance=3,
                seed=42,
            )
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert _edit_distance(seq, "ACGTACGT") >= 3

    def test_combined_constraints(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=15,
                length=10,
                min_edit_distance=3,
                gc_range=(0.3, 0.7),
                max_homopolymer=3,
                seed=42,
            )
            df = pool.generate_library(num_cycles=1)
            seqs = list(df["seq"])
            assert len(seqs) == 15
            for seq in seqs:
                gc = sum(1 for c in seq if c in "GC") / len(seq)
                assert 0.3 <= gc <= 0.7
                assert _check_homopolymer(seq, 3)
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    assert _edit_distance(seqs[i], seqs[j]) >= 3


class TestGetBarcodesVariableLength:
    def test_variable_length(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=10, length=[6, 8, 10], seed=42
            )
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                stripped = seq.rstrip("-")
                assert len(stripped) in [6, 8, 10]
                assert len(seq) == 10  # padded to max

    def test_variable_length_proportions(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=30,
                length=[6, 10],
                length_proportions=[0.5, 0.5],
                seed=42,
            )
            df = pool.generate_library(num_cycles=1)
            seqs = list(df["seq"])
            short_count = sum(1 for s in seqs if s.rstrip("-") != s)
            long_count = sum(1 for s in seqs if s.rstrip("-") == s)
            assert short_count == 15
            assert long_count == 15

    def test_padding_left(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=5, length=[4, 8], padding_side="left", seed=42
            )
            df = pool.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert len(seq) == 8
                if seq.lstrip("-") != seq:
                    assert seq[0] == "-"


class TestGetBarcodesValidation:
    def test_invalid_num_barcodes_zero(self):
        with pp.Party():
            with pytest.raises(ValueError, match="positive integer"):
                pp.get_barcodes(num_barcodes=0, length=8)

    def test_invalid_num_barcodes_negative(self):
        with pp.Party():
            with pytest.raises(ValueError, match="positive integer"):
                pp.get_barcodes(num_barcodes=-1, length=8)

    def test_invalid_length_zero(self):
        with pp.Party():
            with pytest.raises(ValueError, match="positive integers"):
                pp.get_barcodes(num_barcodes=5, length=0)

    def test_hamming_with_variable_length(self):
        with pp.Party():
            with pytest.raises(ValueError, match="min_hamming_distance cannot be used"):
                pp.get_barcodes(
                    num_barcodes=5, length=[6, 8], min_hamming_distance=3
                )

    def test_avoid_without_distance(self):
        with pp.Party():
            with pytest.raises(ValueError, match="avoid_min_distance is required"):
                pp.get_barcodes(
                    num_barcodes=5, length=8, avoid_sequences=["ACGT"]
                )

    def test_gc_range_invalid(self):
        with pp.Party():
            with pytest.raises(ValueError, match="gc_range"):
                pp.get_barcodes(num_barcodes=5, length=8, gc_range=(0.8, 0.2))

    def test_length_proportions_mismatch(self):
        with pp.Party():
            with pytest.raises(ValueError, match="length_proportions length"):
                pp.get_barcodes(
                    num_barcodes=5,
                    length=[6, 8],
                    length_proportions=[0.5, 0.3, 0.2],
                )

    def test_too_many_barcodes_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="Could only generate"):
                pp.get_barcodes(
                    num_barcodes=1000,
                    length=3,
                    min_edit_distance=3,
                    seed=42,
                    max_attempts=5000,
                )

    def test_negative_length_in_list(self):
        with pp.Party():
            with pytest.raises(ValueError, match="positive integers"):
                pp.get_barcodes(num_barcodes=5, length=[8, -1])


class TestGetBarcodesDeterminism:
    def test_seed_deterministic(self):
        with pp.Party():
            pool1 = pp.get_barcodes(num_barcodes=10, length=8, seed=42)
            df1 = pool1.generate_library(num_cycles=1)
        with pp.Party():
            pool2 = pp.get_barcodes(num_barcodes=10, length=8, seed=42)
            df2 = pool2.generate_library(num_cycles=1)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_different_seeds_different_results(self):
        with pp.Party():
            pool1 = pp.get_barcodes(num_barcodes=10, length=8, seed=42)
            df1 = pool1.generate_library(num_cycles=1)
        with pp.Party():
            pool2 = pp.get_barcodes(num_barcodes=10, length=8, seed=99)
            df2 = pool2.generate_library(num_cycles=1)
        assert list(df1["seq"]) != list(df2["seq"])


class TestGetBarcodesDesignCards:
    def test_design_cards_present(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=5, length=8, seed=42,
                cards=["barcode_index", "barcode"],
            )
            df = pool.generate_library(num_cycles=1)
            bc_index_cols = [c for c in df.columns if "barcode_index" in c]
            bc_cols = [c for c in df.columns if "barcode" in c and "barcode_index" not in c]
            assert len(bc_index_cols) > 0
            assert len(bc_cols) > 0

    def test_barcode_card_matches_seq(self):
        with pp.Party():
            pool = pp.get_barcodes(
                num_barcodes=5, length=8, seed=42,
                cards=["barcode"],
            )
            df = pool.generate_library(num_cycles=1)
            bc_cols = [c for c in df.columns if "barcode" in c]
            assert len(bc_cols) > 0
            for _, row in df.iterrows():
                assert row["seq"] == row[bc_cols[0]]

    def test_design_card_keys_defined(self):
        with pp.Party():
            pool = pp.get_barcodes(num_barcodes=3, length=8, seed=42)
            assert "barcode_index" in pool.operation.design_card_keys
            assert "barcode" in pool.operation.design_card_keys


# --- replace_region(sync=True) tests ---


class TestReplaceRegionSync:
    def test_sync_one_to_one_pairing(self):
        """sync=True gives 1:1 pairing, not Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(
                ["AAA<bc/>TTT", "CCC<bc/>GGG", "GGG<bc/>CCC"],
                mode="sequential",
            )
            content = pp.from_seqs(["XX", "YY", "ZZ"], mode="sequential")
            result = bg.replace_region(content, "bc", sync=True)
            df = result.generate_library(num_cycles=1)

        assert len(df) == 3
        seqs = list(df["seq"])
        assert "AAAXXTTT" in seqs
        assert "CCCYYGGG" in seqs
        assert "GGGZZCCC" in seqs

    def test_sync_false_cartesian(self):
        """Default sync=False gives Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(
                ["AAA<bc/>TTT", "CCC<bc/>GGG"],
                mode="sequential",
            )
            content = pp.from_seqs(["XX", "YY"], mode="sequential")
            result = bg.replace_region(content, "bc")

        assert result.num_states == 4

    def test_sync_with_get_barcodes(self):
        """Integration: sync + get_barcodes for MPRA-like workflow."""
        with pp.Party():
            bg = pp.from_seqs(
                ["AAAA<bc/>TTTT", "CCCC<bc/>GGGG", "GGGG<bc/>CCCC"],
                mode="sequential",
            )
            barcodes = pp.get_barcodes(
                num_barcodes=3, length=4, min_edit_distance=2, seed=42
            )
            result = bg.replace_region(barcodes, "bc", sync=True)
            df = result.generate_library(num_cycles=1)

        assert len(df) == 3
        bc_seqs = [row["seq"][4:8] for _, row in df.iterrows()]
        assert len(set(bc_seqs)) == 3


# --- replace_region(keep_tags=True) tests ---


class TestReplaceRegionKeepTags:
    def test_keep_tags_preserves_region(self):
        """keep_tags=True wraps content in original region tags."""
        with pp.Party():
            bg = pp.from_seq("AAA<bc/>TTT")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "bc", keep_tags=True)
            df = result.generate_library(num_seqs=1)

        seq = df["seq"].iloc[0]
        assert "<bc>" in seq
        assert "</bc>" in seq
        assert "XXXX" in seq

    def test_keep_tags_false_removes_tags(self):
        """Default keep_tags=False removes tags."""
        with pp.Party():
            bg = pp.from_seq("AAA<bc/>TTT")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "bc", keep_tags=False)
            df = result.generate_library(num_seqs=1)

        seq = df["seq"].iloc[0]
        assert "<bc>" not in seq
        assert "AAAXXXXTTT" == seq

    def test_keep_tags_with_existing_content(self):
        """keep_tags with region that had existing content."""
        with pp.Party():
            bg = pp.from_seq("PRE<cre>OLD</cre>SUF")
            content = pp.from_seq("NEWCONTENT")
            result = pp.replace_region(bg, content, "cre", keep_tags=True)
            df = result.generate_library(num_seqs=1)

        seq = df["seq"].iloc[0]
        assert seq == "PRE<cre>NEWCONTENT</cre>SUF"

    def test_keep_tags_region_still_tracked(self):
        """With keep_tags=True, the region should remain in the pool's region set."""
        with pp.Party():
            bg = pp.from_seq("AAA<bc/>TTT")
            content = pp.from_seq("XXXX")
            result = bg.replace_region(content, "bc", keep_tags=True)
            assert result.has_region("bc")

    def test_no_keep_tags_region_untracked(self):
        """Without keep_tags, the region should be removed from the pool."""
        with pp.Party():
            bg = pp.from_seq("AAA<bc/>TTT")
            content = pp.from_seq("XXXX")
            result = bg.replace_region(content, "bc")
            assert not result.has_region("bc")


    def test_keep_tags_preserves_content_style(self):
        """keep_tags=True should not drop the style from the content pool."""
        with pp.Party():
            bg = pp.from_seq("AAA<bc/>TTT")
            content = pp.get_barcodes(num_barcodes=1, length=4, style="bold", seed=42)
            result = bg.replace_region(content, "bc", keep_tags=True)
            df = result.generate_library(num_seqs=1)

        seq = df["seq"].iloc[0]
        assert "<bc>" in seq
        # Verify the output Seq object carries style on the barcode positions
        output_seq = result.operation.compute(
            [p.operation.compute([], None)[0] for p in result.parents]
        )[0]
        assert output_seq.style is not None
        assert bool(output_seq.style)  # non-empty style list


class TestReplaceRegionSyncAndKeepTags:
    def test_sync_and_keep_tags_together(self):
        """Both sync=True and keep_tags=True work in combination."""
        with pp.Party():
            bg = pp.from_seqs(
                ["AAA<bc/>TTT", "CCC<bc/>GGG"],
                mode="sequential",
            )
            barcodes = pp.get_barcodes(num_barcodes=2, length=4, seed=42)
            result = bg.replace_region(
                barcodes, "bc", sync=True, keep_tags=True
            )
            df = result.generate_library(num_cycles=1)

        assert len(df) == 2
        for seq in df["seq"]:
            assert "<bc>" in seq
            assert "</bc>" in seq
