"""Tests for the deletion_scan wrapper function."""

import pytest

import poolparty as pp
from poolparty.scan_ops import deletion_scan


class TestDeletionScanBasics:
    """Test basic deletion_scan functionality."""

    def test_returns_pool(self):
        """Test that deletion_scan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3)
            assert hasattr(result, "operation")

    def test_sequential_mode_default(self):
        """Test deletion_scan defaults to sequential mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = deletion_scan(bg, deletion_length=3, mode="sequential").named("result")

        # Default: start=0, end=7, step_size=1 => 8 positions
        df = result.generate_library(num_cycles=1)
        assert len(df) == 8

    def test_preserves_total_length_with_marker(self):
        """Test that output length equals background length when marker is used."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = deletion_scan(
                bg, deletion_length=3, deletion_marker="-", mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 10

    def test_reduces_length_without_marker(self):
        """Test that output length is reduced when no marker is used."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = deletion_scan(
                bg, deletion_length=3, deletion_marker=None, mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 7  # 10 - 3

    def test_marker_appears_in_output(self):
        """Test that deletion marker appears in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = deletion_scan(
                bg, deletion_length=3, deletion_marker="-", mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "---" in seq


class TestDeletionScanStringInputs:
    """Test deletion_scan with string inputs."""

    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = deletion_scan("AAAAAAAAAA", deletion_length=3).named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert "---" in seq
            assert len(seq) == 10


class TestDeletionScanSlicePositions:
    """Test positions parameter with slice syntax."""

    def test_slice_start(self):
        """Test slice with start offset."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            # slice(3, None) on valid range [0, 7] gives positions 3, 4, 5, 6, 7
            result = deletion_scan(
                bg, deletion_length=3, positions=slice(3, None), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 5

        # All deletions should start at position 3 or later
        for seq in df["seq"]:
            idx = seq.index("---")
            assert idx >= 3

    def test_slice_stop(self):
        """Test slice with stop limit."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            # slice(None, 5) on valid range [0, 7] gives positions 0, 1, 2, 3, 4
            result = deletion_scan(
                bg, deletion_length=3, positions=slice(None, 5), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 5

        # All deletions should start at position 4 or earlier
        for seq in df["seq"]:
            idx = seq.index("---")
            assert idx <= 4

    def test_slice_step(self):
        """Test slice with step."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            # slice(None, None, 2) on valid range [0, 7] gives positions 0, 2, 4, 6
            result = deletion_scan(
                bg, deletion_length=3, positions=slice(None, None, 2), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 4

    def test_slice_combined(self):
        """Test slice with start, stop, and step."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            # slice(2, 7, 2) on valid range [0, 7] gives positions 2, 4, 6
            result = deletion_scan(
                bg, deletion_length=3, positions=slice(2, 7, 2), mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 3


class TestDeletionScanModes:
    """Test different modes."""

    def test_random_mode(self):
        """Test random mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3, mode="random").named("result")

        df = result.generate_library(num_seqs=50, seed=42)
        assert len(df) == 50

        # Should have variability in deletion positions
        positions = [seq.index("---") for seq in df["seq"]]
        assert len(set(positions)) > 1

    def test_random_mode_with_num_states(self):
        """Test random mode with explicit num_states."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3, mode="random", num_states=5).named(
                "result"
            )

        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

        for seq in df["seq"]:
            assert "---" in seq
            assert len(seq) == 10


class TestDeletionScanMarkerOptions:
    """Test deletion_marker parameter."""

    def test_default_marker_is_dash(self):
        """Test that default marker is '-'."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3).named("result")

        df = result.generate_library(num_seqs=1)
        assert "---" in df["seq"].iloc[0]

    def test_custom_marker(self):
        """Test custom deletion marker."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3, deletion_marker="X").named("result")

        df = result.generate_library(num_seqs=1)
        assert "XXX" in df["seq"].iloc[0]

    def test_multi_char_marker(self):
        """Test multi-character deletion marker."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=2, deletion_marker="[]").named("result")

        df = result.generate_library(num_seqs=1)
        # deletion_marker * deletion_length = '[]' * 2 = '[][]'
        assert "[][]" in df["seq"].iloc[0]

    def test_none_marker_removes_segment(self):
        """Test that None marker removes segment without replacement."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = deletion_scan(
                bg, deletion_length=3, deletion_marker=None, mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # Should only have A's, no markers
            assert set(seq) == {"A"}
            assert len(seq) == 7  # 10 - 3


class TestDeletionScanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test pool naming via .named()."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3).named("my_result")

        assert result.name == "my_result"

    def test_op_name(self):
        """Test operation naming via .named()."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = deletion_scan(bg, deletion_length=3).named("my_result")

        # Operation name is set via Pool.named()
        assert result.name == "my_result"


class TestDeletionScanValidation:
    """Test input validation."""

    def test_deletion_length_must_be_positive(self):
        """Test error when deletion_length <= 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])

            with pytest.raises(ValueError, match="del_length must be > 0"):
                deletion_scan(bg, deletion_length=0)

            with pytest.raises(ValueError, match="del_length must be > 0"):
                deletion_scan(bg, deletion_length=-1)

    def test_deletion_length_must_be_less_than_bg_length(self):
        """Test error when deletion_length >= bg_length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars

            with pytest.raises(ValueError, match="del_length .* must be < pool.seq_length"):
                deletion_scan(bg, deletion_length=10)

            with pytest.raises(ValueError, match="del_length .* must be < pool.seq_length"):
                deletion_scan(bg, deletion_length=15)

    def test_position_exceeds_maximum(self):
        """Test error when position exceeds maximum allowed value."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            # max_position = 10 - 3 = 7
            # Error raised during generation (positions validated by marker_scan)
            result = deletion_scan(bg, deletion_length=3, positions=[8])

            with pytest.raises(ValueError, match="out of range"):
                result.generate_library(num_seqs=1)


class TestDeletionScanWithMultipleSeqs:
    """Test deletion_scan with pools containing multiple sequences."""

    def test_multiple_backgrounds(self):
        """Test with multiple background sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA", "CCCCCCCCCC"], mode="sequential")
            result = deletion_scan(bg, deletion_length=3, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 8 positions = 16 sequences
        assert len(df) == 16

        for seq in df["seq"]:
            assert "---" in seq
            assert len(seq) == 10


class TestDeletionScanEdgeCases:
    """Test edge cases."""

    def test_delete_at_start(self):
        """Test deletion at position 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = deletion_scan(bg, deletion_length=3, positions=[0], mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "---AAAAAAA"

    def test_delete_at_end(self):
        """Test deletion at maximum position."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            # max_position = 7
            result = deletion_scan(bg, deletion_length=3, positions=[7], mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "AAAAAAA---"

    def test_delete_single_char(self):
        """Test deleting single character."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = deletion_scan(bg, deletion_length=1, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 10 - 1 = 9 positions
        assert len(df) == 10

        for seq in df["seq"]:
            assert "-" in seq
            assert seq.count("-") == 1

    def test_delete_almost_entire_sequence(self):
        """Test deleting almost the entire sequence."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = deletion_scan(bg, deletion_length=9, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # 10 - 9 = 1 position
        assert len(df) == 2  # positions 0 and 1

        for seq in df["seq"]:
            assert seq.count("-") == 9
            assert seq.count("A") == 1


class TestDeletionVsReplacement:
    """Test equivalence between deletion_scan with marker and replacement_scan."""

    def test_equivalent_to_replacement_scan(self):
        """Test that deletion with marker is equivalent to replacement_scan."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")

            # deletion_scan with marker
            del_result = deletion_scan(
                bg, deletion_length=3, deletion_marker="-", mode="sequential"
            ).named("del")

            # equivalent replacement_scan
            rep_result = pp.replacement_scan(bg, "---", mode="sequential").named("rep")

        del_df = del_result.generate_library(num_cycles=1)
        rep_df = rep_result.generate_library(num_cycles=1)

        # Should produce identical sequences
        assert set(del_df["seq"]) == set(rep_df["seq"])
