"""Tests for the replacement_multiscan function."""

import pytest

import poolparty as pp
from poolparty.multiscan_ops import replacement_multiscan


class TestReplacementMultiscanBasics:
    """Test basic replacement_multiscan functionality."""

    def test_returns_pool(self):
        """Test that replacement_multiscan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")  # 3 chars
            result = replacement_multiscan(bg, num_replacements=2, replacement_pools=ins)
            assert hasattr(result, "operation")

    def test_replaces_correct_length(self):
        """Test that replacements have correct length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")  # 3 chars
            result = replacement_multiscan(bg, num_replacements=2, replacement_pools=ins).named(
                "result"
            )

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Should have 6 G's (2 replacements * 3 chars)
            assert seq.count("G") == 6
            assert len(seq) == 18

    def test_string_background(self):
        """Test that string background is converted to Pool."""
        with pp.Party() as party:
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                "AAAAAAAAAAAAAAAAAA", num_replacements=2, replacement_pools=ins
            ).named("result")

        df = result.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 6


class TestReplacementMultiscanPoolHandling:
    """Test replacement_pools handling."""

    def test_single_pool_creates_deepcopies(self):
        """Test that a single Pool is deepcopied for multiple replacements."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(bg, num_replacements=3, replacement_pools=ins).named(
                "result"
            )

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Should have 9 G's (3 replacements * 3 chars)
            assert seq.count("G") == 9

    def test_multiple_pools_different_content(self):
        """Test that multiple pools with different content work."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            pool1 = pp.from_seq("GGG")
            pool2 = pp.from_seq("TTT")
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=[pool1, pool2]
            ).named("result")

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Should have 3 G's and 3 T's
            assert seq.count("G") == 3
            assert seq.count("T") == 3

    def test_multiple_pools_length_mismatch_raises(self):
        """Test that mismatched pool count raises error."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            pool1 = pp.from_seq("GGG")
            pool2 = pp.from_seq("TTT")

            with pytest.raises(ValueError, match="replacement_pools length .* must equal"):
                replacement_multiscan(bg, num_replacements=3, replacement_pools=[pool1, pool2])

    def test_pools_with_different_seq_lengths(self):
        """Test that pools with different seq_lengths are now supported."""
        with pp.Party():
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            pool1 = pp.from_seq("GGG")  # 3 chars
            pool2 = pp.from_seq("TTTT")  # 4 chars

            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=[pool1, pool2],
                mode="random", num_states=5,
            )
            df = result.generate_library()
            assert len(df) == 5
            for seq in df["seq"]:
                assert "GGG" in seq
                assert "TTTT" in seq


class TestReplacementMultiscanModes:
    """Test different modes."""

    def test_random_mode(self):
        """Test random mode (default)."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=ins, mode="random"
            ).named("result")

        df = result.generate_library(num_seqs=50, seed=42)
        assert len(df) == 50

        for seq in df["seq"]:
            assert seq.count("G") == 6

    def test_random_mode_with_num_states(self):
        """Test random mode with explicit num_states."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=ins, mode="random", num_states=5
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

        for seq in df["seq"]:
            assert seq.count("G") == 6
            assert len(seq) == 18

    def test_sequential_mode(self):
        """Test that sequential mode works."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=ins, mode="sequential"
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert seq.count("G") == 6


class TestReplacementMultiscanValidation:
    """Test input validation."""

    def test_num_replacements_must_be_positive(self):
        """Test error when num_replacements < 1."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            ins = pp.from_seq("GGG")

            with pytest.raises(ValueError, match="num_replacements must be >= 1"):
                replacement_multiscan(bg, num_replacements=0, replacement_pools=ins)

    def test_cannot_fit_replacements(self):
        """Test error when replacements cannot fit without overlapping."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])  # 10 chars
            ins = pp.from_seq("GGG")  # 3 chars

            # 5 replacements of length 3 = 15 chars, but only 10 available
            with pytest.raises(ValueError, match="Cannot fit .* non-overlapping replacements"):
                replacement_multiscan(bg, num_replacements=5, replacement_pools=ins)


class TestReplacementMultiscanPositions:
    """Test positions parameter."""

    def test_positions_list(self):
        """Test with explicit positions list."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            # Only allow replacements starting at positions 0, 6, 12
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=ins, positions=[0, 6, 12]
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 6

    def test_positions_slice(self):
        """Test with slice positions."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            # Only allow replacements in first half
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=ins, positions=slice(0, 9)
            ).named("result")

        df = result.generate_library(num_seqs=20, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 6


class TestReplacementMultiscanNumReplacements:
    """Test different numbers of replacements."""

    def test_single_replacement(self):
        """Test with single replacement (equivalent to replacement_scan)."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(bg, num_replacements=1, replacement_pools=ins).named(
                "result"
            )

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 3
            assert len(seq) == 18

    def test_three_replacements(self):
        """Test with three replacements."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(bg, num_replacements=3, replacement_pools=ins).named(
                "result"
            )

        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 9  # 3 * 3
            assert len(seq) == 18


class TestReplacementMultiscanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(bg, num_replacements=2, replacement_pools=ins).named(
                "my_result"
            )

        assert result.name == "my_result"


class TestReplacementMultiscanNonOverlapping:
    """Test that replacements do not overlap."""

    def test_replacements_are_non_overlapping(self):
        """Test that multiple replacements produce correct total replacement count."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAAAAAAAAAA"])  # 18 chars
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(bg, num_replacements=3, replacement_pools=ins).named(
                "result"
            )

        df = result.generate_library(num_seqs=50, seed=42)
        for seq in df["seq"]:
            # Find positions of all 'G' characters
            g_positions = [i for i, c in enumerate(seq) if c == "G"]
            # Should have exactly 9 G's (3 replacements * 3 length)
            assert len(g_positions) == 9
            # Check that total G's equals num_replacements * replacement_length
            assert seq.count("G") == 9
            assert seq.count("A") == 9  # 18 - 9 = 9 A's remain
