"""Tests for the shuffle_scan operation."""

import poolparty as pp
from poolparty.scan_ops import shuffle_scan


class TestShuffleScanBasics:
    """Test basic shuffle_scan functionality."""

    def test_returns_pool(self):
        """Test that shuffle_scan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = shuffle_scan(bg, shuffle_length=3)
            assert hasattr(result, "operation")

    def test_sequential_mode(self):
        """Test shuffle_scan in sequential mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = shuffle_scan(bg, shuffle_length=3, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        # Should have multiple positions
        assert len(df) > 0

    def test_preserves_length(self):
        """Test that output length equals background length."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")  # 10 chars
            result = shuffle_scan(bg, shuffle_length=3, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert len(seq) == 10


class TestShuffleScanMarkerRemoval:
    """Test that shuffle_scan removes the internal _shuf marker."""

    def test_no_shuf_marker_in_output(self):
        """Test that _shuf marker tags are not present in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(bg, shuffle_length=3, mode="sequential").named("result")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # The internal _shuf marker should be removed
            assert "<_shuf>" not in seq
            assert "</_shuf>" not in seq

    def test_shuffle_scan_with_region_marker(self):
        """Test shuffle_scan within a region marker."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAA<region>TTTT</region>CCCC")
            result = shuffle_scan(bg, shuffle_length=2, region="region", mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # Outer region marker should be preserved
            assert "<region>" in seq or "</region>" in seq or "region" in seq.lower()
            # Internal _shuf marker should be removed
            assert "<_shuf>" not in seq
            assert "</_shuf>" not in seq


class TestShuffleScanStyling:
    """Test style parameter."""

    def test_style_parameter_accepted(self):
        """Test that style parameter is accepted."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(bg, shuffle_length=3, style="purple", mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        # Should work without errors
        assert len(df) > 0

    def test_style_none(self):
        """Test that style=None (default) works."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(bg, shuffle_length=3, style=None, mode="sequential").named(
                "result"
            )

        df = result.generate_library(num_cycles=1)
        assert len(df) > 0


class TestShuffleScanStringInputs:
    """Test shuffle_scan with string inputs."""

    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = shuffle_scan("AAAAAAAAAA", shuffle_length=3).named("result")

        df = result.generate_library(num_seqs=3)
        for seq in df["seq"]:
            assert len(seq) == 10
            # No _shuf markers
            assert "<_shuf>" not in seq
            assert "</_shuf>" not in seq


class TestShuffleScanNaming:
    """Test shuffle_scan naming parameters."""

    def test_prefix_combined_index(self):
        """Test prefix produces combined position*shuffle index."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(
                bg,
                shuffle_length=3,
                positions=slice(0, 3),  # 3 positions
                shuffles_per_position=2,
                prefix="shuf",
                mode="sequential",
            ).named("result")

        df = result.generate_library(num_cycles=1)
        # 3 positions * 2 shuffles = 6 states
        assert len(df) == 6
        names = df["name"].tolist()
        expected = ["shuf_0", "shuf_1", "shuf_2", "shuf_3", "shuf_4", "shuf_5"]
        assert names == expected

    def test_prefix_position_and_shuffle(self):
        """Test prefix_position and prefix_shuffle produce composite names."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(
                bg,
                shuffle_length=3,
                positions=slice(0, 3),  # 3 positions
                shuffles_per_position=2,
                prefix_position="pos",
                prefix_shuffle="var",
                mode="sequential",
            ).named("result")

        df = result.generate_library(num_cycles=1)
        # 3 positions * 2 shuffles = 6 states
        assert len(df) == 6
        names = df["name"].tolist()
        expected = [
            "pos_0.var_0",
            "pos_0.var_1",
            "pos_1.var_0",
            "pos_1.var_1",
            "pos_2.var_0",
            "pos_2.var_1",
        ]
        assert names == expected

    def test_all_prefixes(self):
        """Test all three prefix parameters together."""
        with pp.Party() as party:
            bg = pp.from_seqs(["AAAAAAAAAA"], mode="sequential")
            result = shuffle_scan(
                bg,
                shuffle_length=3,
                positions=slice(0, 2),  # 2 positions
                shuffles_per_position=2,
                prefix="shuf",
                prefix_position="pos",
                prefix_shuffle="var",
                mode="sequential",
            ).named("result")

        df = result.generate_library(num_cycles=1)
        # 2 positions * 2 shuffles = 4 states
        assert len(df) == 4
        names = df["name"].tolist()
        expected = [
            "shuf_0.pos_0.var_0",
            "shuf_1.pos_0.var_1",
            "shuf_2.pos_1.var_0",
            "shuf_3.pos_1.var_1",
        ]
        assert names == expected
