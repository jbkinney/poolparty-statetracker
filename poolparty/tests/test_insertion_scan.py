"""Tests for the insertion_scan wrapper function."""

import pytest

import poolparty as pp
from poolparty.scan_ops import insertion_scan


class TestInsertionScanBasics:
    """Test basic insertion_scan functionality."""

    def test_returns_pool(self):
        """Test that insertion_scan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins)
            assert hasattr(result, "operation")

    def test_sequential_mode_default(self):
        """Test insertion_scan defaults to sequential mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            ins = pp.from_seqs(["TTT"], mode="sequential")  # 3 chars
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        # Default: start=0, end=10, step_size=1 => 11 positions
        df = result.generate_library(num_cycles=1)
        assert len(df) == 11

    def test_output_length_is_sum(self):
        """Test that output length equals bg_length + ins_length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            result = insertion_scan(bg, ins).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 13  # 10 + 3

    def test_insert_appears_in_output(self):
        """Test that insert sequence appears in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            ins = pp.from_seqs(["TTT"], mode="sequential")
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "TTT" in seq

    def test_background_fully_preserved(self):
        """Test that entire background sequence is preserved."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            ins = pp.from_seqs(["TTT"], mode="sequential")
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # Remove insert and check background is intact
            without_insert = seq.replace("TTT", "")
            assert without_insert == "AAAAAAAAAA"


class TestInsertionScanStringInputs:
    """Test insertion_scan with string inputs."""

    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = insertion_scan("AAAAAAAAAA", pp.from_seqs(["TTT"])).named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 13

    def test_ins_pool_as_string(self):
        """Test insert as string."""
        with pp.Party() as party:
            result = insertion_scan(pp.from_seqs(["AAAAAAAAAA"]), "TTT").named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 13

    def test_both_as_strings(self):
        """Test both background and insert as strings."""
        with pp.Party() as party:
            result = insertion_scan("AAAAAAAAAA", "TTT").named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 13


class TestInsertionScanSlicePositions:
    """Test positions parameter with slice syntax."""

    def test_slice_start(self):
        """Test slice with start offset."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(3, None) on valid range [0, 10] gives positions 3, 4, ..., 10
            result = insertion_scan(bg, ins, positions=slice(3, None), mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 8

        # All inserts should start at position 3 or later
        for seq in df["seq"]:
            idx = seq.index("TTT")
            assert idx >= 3

    def test_slice_stop(self):
        """Test slice with stop limit."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(None, 5) on valid range [0, 10] gives positions 0, 1, 2, 3, 4
            result = insertion_scan(bg, ins, positions=slice(None, 5), mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 5

        # All inserts should start at position 4 or earlier
        for seq in df["seq"]:
            idx = seq.index("TTT")
            assert idx <= 4

    def test_slice_step(self):
        """Test slice with step."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(None, None, 2) on valid range [0, 10] gives positions 0, 2, 4, 6, 8, 10
            result = insertion_scan(
                bg, ins, positions=slice(None, None, 2), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 6

    def test_slice_combined(self):
        """Test slice with start, stop, and step."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(2, 9, 2) on valid range [0, 10] gives positions 2, 4, 6, 8
            result = insertion_scan(bg, ins, positions=slice(2, 9, 2), mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 4


class TestInsertionScanModes:
    """Test different modes."""

    def test_random_mode(self):
        """Test random mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins, mode="random").named("result")

        df = result.generate_library(num_seqs=50, seed=42)
        assert len(df) == 50

        # Should have variability in insert positions
        positions = [seq.index("TTT") for seq in df["seq"]]
        assert len(set(positions)) > 1

    def test_random_mode_with_num_states(self):
        """Test random mode with explicit num_states."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins, mode="random", num_states=5).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 13


class TestInsertionScanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test pool_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins).named("my_result")

        assert result.name == "my_result"

    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins).named("my_result")

        # Operation name is set via Pool.named()
        assert result.name == "my_result"


class TestInsertionScanValidation:
    """Test input validation."""

    def test_position_exceeds_maximum(self):
        """Test error when position exceeds maximum allowed value."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # max_position = 10 (can insert at any position including after last char)
            # Position validation happens at construction in sequential mode
            with pytest.raises(ValueError, match="out of range"):
                insertion_scan(bg, ins, positions=[11], mode="sequential")


class TestInsertionScanWithMultipleSeqs:
    """Test insertion_scan with pools containing multiple sequences."""

    def test_multiple_backgrounds(self):
        """Test with multiple background sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA", "CCCCCCCCCC"], mode="sequential")
            ins = pp.from_seqs(["TTT"], mode="sequential")
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 11 positions = 22 sequences
        assert len(df) == 22

        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 13

    def test_multiple_inserts(self):
        """Test with multiple insert sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT", "GGG"], mode="sequential")
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 11 positions * 2 inserts = 22 sequences
        assert len(df) == 22

        for seq in df["seq"]:
            assert "TTT" in seq or "GGG" in seq
            assert len(seq) == 13

    def test_multiple_backgrounds_and_inserts(self):
        """Test with multiple backgrounds and inserts."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA", "CCCCCCCCCC"], mode="sequential")
            ins = pp.from_seqs(["TTT", "GGG"], mode="sequential")
            result = insertion_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 11 positions * 2 inserts = 44 sequences
        assert len(df) == 44


class TestInsertionScanEdgeCases:
    """Test edge cases."""

    def test_insert_at_start(self):
        """Test insertion at position 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = insertion_scan(bg, ins, positions=[0], mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "TTTAAAAAAAAAA"  # Insert at start, nothing removed

    def test_insert_at_end(self):
        """Test insertion at end position."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # max_position = 10
            result = insertion_scan(bg, ins, positions=[10], mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "AAAAAAAAAATTT"  # Insert at end

    def test_insert_in_middle(self):
        """Test insertion in middle of sequence."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            result = insertion_scan(bg, ins, positions=[5], mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "AAAAATTTAAAAA"  # 5 A's + TTT + 5 A's


class TestInsertionVsReplacement:
    """Test difference between insertion_scan and replacement_scan."""

    def test_length_difference(self):
        """Test that insertion adds length while replacement preserves it."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars

            insert_result = insertion_scan(bg, ins).named("insert")
            replace_result = pp.replacement_scan(bg, ins).named("replace")

        insert_df = insert_result.generate_library(num_seqs=1)
        replace_df = replace_result.generate_library(num_seqs=1)

        # Insertion adds length
        assert len(insert_df["seq"].iloc[0]) == 13  # 10 + 3
        # Replacement preserves length
        assert len(replace_df["seq"].iloc[0]) == 10

    def test_background_preservation(self):
        """Test that insertion preserves full background, replacement does not."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars

            insert_result = insertion_scan(bg, ins, positions=[5]).named("insert")
            replace_result = pp.replacement_scan(bg, ins, positions=[5]).named("replace")

        insert_df = insert_result.generate_library(num_seqs=1)
        replace_df = replace_result.generate_library(num_seqs=1)

        # Insertion: full background preserved
        insert_without_ttt = insert_df["seq"].iloc[0].replace("TTT", "")
        assert insert_without_ttt == "AAAAAAAAAA"

        # Replacement: some background removed
        replace_without_ttt = replace_df["seq"].iloc[0].replace("TTT", "")
        assert replace_without_ttt == "AAAAAAA"  # 7 A's (3 replaced)
