"""Tests for the deletion_multiscan function."""

import pytest

import poolparty as pp
from poolparty.multiscan_ops import deletion_multiscan


class TestDeletionMultiscanBasics:
    """Test basic deletion_multiscan functionality."""

    def test_returns_pool(self):
        """Test that deletion_multiscan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=2)
            assert hasattr(result, "operation")

    def test_preserves_total_length_with_marker(self):
        """Test that output length equals background length when marker is used."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, deletion_marker="-"
            ).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert len(seq) == 18

    def test_reduces_length_without_marker(self):
        """Test that output length is reduced when no marker is used."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, deletion_marker=None
            ).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert len(seq) == 12  # 18 - (3 * 2)

    def test_markers_appear_in_output(self):
        """Test that deletion markers appear in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, deletion_marker="-"
            ).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Should have two separate '---' regions
            assert seq.count("-") == 6  # 3 * 2 deletions


class TestDeletionMultiscanStringInputs:
    """Test deletion_multiscan with string inputs."""

    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = deletion_multiscan(
                "AAAAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2
            ).named("result")

        df = result.generate_library(num_seqs=3, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 6
            assert len(seq) == 18


class TestDeletionMultiscanModes:
    """Test different modes."""

    def test_random_mode(self):
        """Test random mode (default)."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, mode="random"
            ).named("result")

        df = result.generate_library(num_seqs=50, seed=42)
        assert len(df) == 50

        # All sequences should have exactly 6 deletion chars
        for seq in df["seq"]:
            assert seq.count("-") == 6

    def test_random_mode_with_num_states(self):
        """Test random mode with explicit num_states."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, mode="random", num_states=5
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

        for seq in df["seq"]:
            assert seq.count("-") == 6
            assert len(seq) == 18

    def test_sequential_mode(self):
        """Test that sequential mode works."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert seq.count("-") == 6


class TestDeletionMultiscanMarkerOptions:
    """Test deletion_marker parameter."""

    def test_default_marker_is_dash(self):
        """Test that default marker is '-'."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=2).named("result")

        df = result.generate_library(num_seqs=1, seed=42)
        assert "-" in df["seq"].iloc[0]

    def test_custom_marker(self):
        """Test custom deletion marker."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, deletion_marker="X"
            ).named("result")

        df = result.generate_library(num_seqs=1, seed=42)
        assert "XXX" in df["seq"].iloc[0]
        assert df["seq"].iloc[0].count("X") == 6

    def test_none_marker_removes_segment(self):
        """Test that None marker removes segment without replacement."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, deletion_marker=None
            ).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Should only have A's, no markers
            assert set(seq) == {"A"}
            assert len(seq) == 12  # 18 - (3 * 2)


class TestDeletionMultiscanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test pool naming via .named()."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=2).named("my_result")

        assert result.name == "my_result"


class TestDeletionMultiscanValidation:
    """Test input validation."""

    def test_deletion_length_must_be_positive(self):
        """Test error when deletion_length <= 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])

            with pytest.raises(ValueError, match="deletion_length must be > 0"):
                deletion_multiscan(bg, deletion_length=0, num_deletions=2)

            with pytest.raises(ValueError, match="deletion_length must be > 0"):
                deletion_multiscan(bg, deletion_length=-1, num_deletions=2)

    def test_deletion_length_must_be_less_than_bg_length(self):
        """Test error when deletion_length >= bg_length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars

            with pytest.raises(ValueError, match="deletion_length .* must be < pool.seq_length"):
                deletion_multiscan(bg, deletion_length=18, num_deletions=1)

            with pytest.raises(ValueError, match="deletion_length .* must be < pool.seq_length"):
                deletion_multiscan(bg, deletion_length=20, num_deletions=1)

    def test_num_deletions_must_be_positive(self):
        """Test error when num_deletions < 1."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])

            with pytest.raises(ValueError, match="num_deletions must be >= 1"):
                deletion_multiscan(bg, deletion_length=3, num_deletions=0)

    def test_cannot_fit_deletions(self):
        """Test error when deletions cannot fit without overlapping."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars

            # 5 deletions of length 3 = 15 chars, but only 10 available
            with pytest.raises(ValueError, match="Cannot fit .* non-overlapping deletions"):
                deletion_multiscan(bg, deletion_length=3, num_deletions=5)


class TestDeletionMultiscanNumDeletions:
    """Test different numbers of deletions."""

    def test_single_deletion(self):
        """Test with single deletion (equivalent to deletion_scan)."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=1).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 3
            assert len(seq) == 18

    def test_three_deletions(self):
        """Test with three deletions."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=3).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 9  # 3 * 3
            assert len(seq) == 18

    def test_max_deletions(self):
        """Test with high number of deletions."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAAAAAAAA"])  # 24 chars
            # Use 4 deletions of length 3 = 12 chars (leaves room for spacing)
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=4).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 12  # 3 * 4
            assert len(seq) == 24


class TestDeletionMultiscanNonOverlapping:
    """Test that deletions do not overlap."""

    def test_deletions_are_non_overlapping(self):
        """Test that multiple deletions produce correct total deletion count."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            result = deletion_multiscan(bg, deletion_length=3, num_deletions=3).named("result")

        df = result.generate_library(num_seqs=50, seed=42)
        for seq in df["seq"]:
            # Find positions of all '-' characters
            dash_positions = [i for i, c in enumerate(seq) if c == "-"]
            # Should have exactly 9 dashes (3 deletions * 3 length)
            # Deletions may be adjacent but not overlapping
            assert len(dash_positions) == 9

            # Check that total dashes equals num_deletions * deletion_length
            assert seq.count("-") == 9
            assert seq.count("A") == 9  # 18 - 9 = 9 A's remain


class TestDeletionMultiscanPositions:
    """Test positions parameter."""

    def test_positions_list(self):
        """Test with explicit positions list."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            # Only allow deletions starting at positions 0, 6, 12
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, positions=[0, 6, 12]
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 6

    def test_positions_slice(self):
        """Test with slice positions."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            # Only allow deletions in first half
            result = deletion_multiscan(
                bg, deletion_length=3, num_deletions=2, positions=slice(0, 9)
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 6
