"""Tests for slice_seq - SEQUENCE slicing (slicing characters in a sequence)."""

import poolparty as pp
from poolparty.fixed_ops.fixed import FixedOp
from poolparty.fixed_ops.slice_seq import slice_seq


class TestSliceSeqFactory:
    """Test slice_seq factory function."""

    def test_returns_pool(self):
        """Test that slice_seq returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            sliced = slice_seq(pool, slice(0, 4))
            assert sliced is not None
            assert hasattr(sliced, "operation")

    def test_creates_fixed_op(self):
        """Test that slice_seq creates a FixedOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            sliced = slice_seq(pool, slice(0, 4))
            assert isinstance(sliced.operation, FixedOp)

    def test_slice_seq_with_int(self):
        """Test slice_seq with integer index."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, 0).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "A"


class TestSliceSeqIntegerIndexing:
    """Test integer indexing for sequence slicing."""

    def test_positive_index_first(self):
        """Test positive index for first character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, 0).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "A"

    def test_positive_index_middle(self):
        """Test positive index for middle character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, 2).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "C"

    def test_negative_index_last(self):
        """Test negative index for last character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, -1).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "F"


class TestSliceSeqRanges:
    """Test slice range operations for sequence slicing."""

    def test_start_to_end(self):
        """Test slice with start and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, slice(2, 6)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "CDEF"

    def test_from_start(self):
        """Test slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, slice(None, 4)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ABCD"

    def test_to_end(self):
        """Test slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, slice(4, None)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "EFGH"

    def test_full_slice(self):
        """Test full slice (copy)."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, slice(None, None)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ABCDEFGH"


class TestSliceSeqWithStep:
    """Test slice with step parameter for sequence slicing."""

    def test_step_two(self):
        """Test slice with step=2."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, slice(None, None, 2)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACEG"

    def test_reverse(self):
        """Test reverse slice with step=-1."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCD"])
            sliced = slice_seq(pool, slice(None, None, -1)).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "DCBA"


class TestSliceSeqCompute:
    """Test slice_seq compute methods directly."""

    def test_compute_with_slice(self):
        """Test compute with slice key."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, slice(0, 2))

        output_seq, card = sliced.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "AC"

    def test_compute_with_int(self):
        """Test compute with integer key."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, 0)

        output_seq, card = sliced.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "A"


class TestSliceSeqCustomName:
    """Test slice_seq name parameter."""

    def test_default_name(self):
        """Test default operation name is 'slice_seq'."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, slice(0, 2))
            assert sliced.operation.name.endswith(":slice_seq")
