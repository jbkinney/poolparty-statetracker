"""Tests for the replacement_scan wrapper function."""

import pytest

import poolparty as pp
from poolparty.scan_ops import replacement_scan


class TestReplacementScanBasics:
    """Test basic replacement_scan functionality."""

    def test_returns_pool(self):
        """Test that replacement_scan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = replacement_scan(bg, ins)
            assert hasattr(result, "operation")

    def test_sequential_mode_default(self):
        """Test replacement_scan defaults to sequential mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            ins = pp.from_seqs(["TTT"], mode="sequential")  # 3 chars
            result = replacement_scan(bg, ins, mode="sequential").named("result")

        # Default: start=0, end=7, step_size=1 => 8 positions
        df = result.generate_library(num_cycles=1)
        assert len(df) == 8

    def test_preserves_total_length(self):
        """Test that output length equals background length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            result = replacement_scan(bg, ins).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 10

    def test_insert_appears_in_output(self):
        """Test that insert sequence appears in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            ins = pp.from_seqs(["TTT"], mode="sequential")
            result = replacement_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "TTT" in seq


class TestReplacementScanStringInputs:
    """Test replacement_scan with string inputs."""

    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = replacement_scan("AAAAAAAAAA", pp.from_seqs(["TTT"])).named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 10

    def test_ins_pool_as_string(self):
        """Test insert as string."""
        with pp.Party() as party:
            result = replacement_scan(pp.from_seqs(["AAAAAAAAAA"]), "TTT").named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 10

    def test_both_as_strings(self):
        """Test both background and insert as strings."""
        with pp.Party() as party:
            result = replacement_scan("AAAAAAAAAA", "TTT").named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 10


class TestReplacementScanSlicePositions:
    """Test positions parameter with slice syntax."""

    def test_slice_start(self):
        """Test slice with start offset."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(3, None) on valid range [0, 7] gives positions 3, 4, 5, 6, 7
            result = replacement_scan(bg, ins, positions=slice(3, None), mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 5

        # All inserts should start at position 3 or later
        for seq in df["seq"]:
            idx = seq.index("TTT")
            assert idx >= 3

    def test_slice_stop(self):
        """Test slice with stop limit."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(None, 5) on valid range [0, 7] gives positions 0, 1, 2, 3, 4
            result = replacement_scan(bg, ins, positions=slice(None, 5), mode="sequential").named(
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
            # slice(None, None, 2) on valid range [0, 7] gives positions 0, 2, 4, 6
            result = replacement_scan(
                bg, ins, positions=slice(None, None, 2), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 4

    def test_slice_combined(self):
        """Test slice with start, stop, and step."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # slice(2, 7, 2) on valid range [0, 7] gives positions 2, 4, 6
            result = replacement_scan(bg, ins, positions=slice(2, 7, 2), mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 3


class TestReplacementScanModes:
    """Test different modes."""

    def test_random_mode(self):
        """Test random mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = replacement_scan(bg, ins, mode="random").named("result")

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
            result = replacement_scan(bg, ins, mode="random", num_states=5).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 10

            # Should NOT have dots
            assert "." not in seq


class TestReplacementScanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test pool_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = replacement_scan(bg, ins).named("my_result")

        assert result.name == "my_result"

    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = replacement_scan(bg, ins).named("my_result")

        # Operation name is set via Pool.named()
        assert result.name == "my_result"


class TestReplacementScanValidation:
    """Test input validation."""

    def test_position_exceeds_maximum(self):
        """Test error when position exceeds maximum allowed value."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # max_position = 10 - 3 = 7
            # Position validation happens at construction in sequential mode
            with pytest.raises(ValueError, match="out of range"):
                replacement_scan(bg, ins, positions=[8], mode="sequential")


class TestReplacementScanWithMultipleSeqs:
    """Test replacement_scan with pools containing multiple sequences."""

    def test_multiple_backgrounds(self):
        """Test with multiple background sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA", "CCCCCCCCCC"], mode="sequential")
            ins = pp.from_seqs(["TTT"], mode="sequential")
            result = replacement_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 8 positions = 16 sequences
        assert len(df) == 16

        for seq in df["seq"]:
            assert "TTT" in seq
            assert len(seq) == 10

    def test_multiple_inserts(self):
        """Test with multiple insert sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT", "GGG"], mode="sequential")
            result = replacement_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 8 positions * 2 inserts = 16 sequences
        assert len(df) == 16

        for seq in df["seq"]:
            assert "TTT" in seq or "GGG" in seq
            assert len(seq) == 10

    def test_multiple_backgrounds_and_inserts(self):
        """Test with multiple backgrounds and inserts."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA", "CCCCCCCCCC"], mode="sequential")
            ins = pp.from_seqs(["TTT", "GGG"], mode="sequential")
            result = replacement_scan(bg, ins, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 8 positions * 2 inserts = 32 sequences
        assert len(df) == 32


class TestReplacementScanEdgeCases:
    """Test edge cases."""

    def test_start_at_zero(self):
        """Test replacement at position 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            ins = pp.from_seqs(["TTT"])
            result = replacement_scan(bg, ins, positions=[0], mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "TTTAAAAAAA"

    def test_replace_at_end(self):
        """Test replacement at maximum position."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTT"])  # 3 chars
            # max_position = 7
            result = replacement_scan(bg, ins, positions=[7], mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "AAAAAAATTT"

    def test_replace_same_length_as_background(self):
        """Test when insert length equals background length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seqs(["TTTTTTTTTT"])  # 10 chars
            # max_end = 10 - 10 = 0
            result = replacement_scan(bg, ins).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        # Only position 0, so insert replaces everything
        assert df["seq"].iloc[0] == "TTTTTTTTTT"
