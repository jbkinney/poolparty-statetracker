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
            sliced = slice_seq(pool, start=0, stop=4)
            assert sliced is not None
            assert hasattr(sliced, "operation")

    def test_creates_fixed_op(self):
        """Test that slice_seq creates a FixedOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            sliced = slice_seq(pool, start=0, stop=4)
            assert isinstance(sliced.operation, FixedOp)

    def test_slice_seq_single_char(self):
        """Test slice_seq extracting a single character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, start=0, stop=1).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "A"


class TestSliceSeqPositionalSlicing:
    """Test positional slicing for sequence slicing."""

    def test_first_char(self):
        """Test extracting first character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, start=0, stop=1).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "A"

    def test_middle_char(self):
        """Test extracting middle character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, start=2, stop=3).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "C"

    def test_negative_index_last(self):
        """Test negative index for last character."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEF"])
            sliced = slice_seq(pool, start=-1).named("char")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "F"


class TestSliceSeqRanges:
    """Test slice range operations for sequence slicing."""

    def test_start_to_end(self):
        """Test slice with start and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, start=2, stop=6).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "CDEF"

    def test_from_start(self):
        """Test slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, stop=4).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ABCD"

    def test_to_end(self):
        """Test slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, start=4).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "EFGH"

    def test_full_slice(self):
        """Test full slice (copy)."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ABCDEFGH"


class TestSliceSeqWithStep:
    """Test slice with step parameter for sequence slicing."""

    def test_step_two(self):
        """Test slice with step=2."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            sliced = slice_seq(pool, step=2).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACEG"

    def test_reverse(self):
        """Test reverse slice with step=-1."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCD"])
            sliced = slice_seq(pool, step=-1).named("sl")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "DCBA"


class TestSliceSeqCompute:
    """Test slice_seq compute methods directly."""

    def test_compute_with_slice(self):
        """Test compute with slice parameters."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, start=0, stop=2)

        output_seq, card = sliced.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "AC"

    def test_compute_single_char(self):
        """Test compute with single character extraction."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, start=0, stop=1)

        output_seq, card = sliced.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "A"


class TestSliceSeqCustomName:
    """Test slice_seq name parameter."""

    def test_default_name(self):
        """Test default operation name is 'slice_seq'."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, start=0, stop=2)
            assert sliced.operation.name.endswith(":slice_seq")


class TestSliceSeqRegion:
    """Test slice_seq with region parameter."""

    def test_region_only(self):
        """Test extracting only a named region."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = slice_seq(pool, region="orf")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ATGCCC"

    def test_region_with_start_stop(self):
        """Test slicing within a named region."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = slice_seq(pool, region="orf", start=0, stop=3)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ATG"

    def test_region_with_step(self):
        """Test slicing within a region with step."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = slice_seq(pool, region="orf", step=2)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AGC"

    def test_region_interval(self):
        """Test slicing with [start, stop] interval region."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            sliced = slice_seq(pool, region=[2, 6])

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "GTAC"

    def test_region_interval_with_slice(self):
        """Test slicing within an interval region."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            sliced = slice_seq(pool, region=[2, 6], start=0, stop=2)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "GT"


class TestSliceSeqMethod:
    """Test slice_seq as a method on Pool objects."""

    def test_method_basic(self):
        """Test basic method call."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            sliced = pool.slice_seq(start=0, stop=4)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_method_with_region(self):
        """Test method call with region."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = pool.slice_seq(region="orf")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ATGCCC"

    def test_method_with_region_and_slice(self):
        """Test method call with region and slice parameters."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = pool.slice_seq(region="orf", start=0, stop=3)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ATG"

    def test_method_step(self):
        """Test method call with step."""
        with pp.Party() as party:
            pool = pp.from_seq("ABCDEFGH")
            sliced = pool.slice_seq(step=2)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACEG"


class TestSliceSeqKeepContext:
    """Test slice_seq with keep_context parameter."""

    def test_keep_context_named_region(self):
        """Test keep_context with named region."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = slice_seq(pool, region="orf", keep_context=True)

        df = sliced.generate_library(num_seqs=1)
        # Should return full region content in context
        assert df["seq"].iloc[0] == "AAAATGCCCTTT"

    def test_keep_context_with_slice(self):
        """Test keep_context with region and slice."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = slice_seq(pool, region="orf", start=0, stop=3, keep_context=True)

        df = sliced.generate_library(num_seqs=1)
        # Should return sliced region content in context
        assert df["seq"].iloc[0] == "AAAATGTTT"

    def test_keep_context_interval_region(self):
        """Test keep_context with interval region."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            sliced = slice_seq(pool, region=[2, 6], keep_context=True)

        df = sliced.generate_library(num_seqs=1)
        # Should return full interval in context
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_keep_context_interval_with_slice(self):
        """Test keep_context with interval region and slice."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            sliced = slice_seq(pool, region=[2, 6], start=0, stop=2, keep_context=True)

        df = sliced.generate_library(num_seqs=1)
        # prefix (AC) + sliced region (GT) + suffix (GT)
        assert df["seq"].iloc[0] == "ACGTGT"

    def test_keep_context_requires_region(self):
        """Test that keep_context=True raises error without region."""
        import pytest

        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            with pytest.raises(ValueError, match="keep_context=True requires a region"):
                slice_seq(pool, start=0, stop=4, keep_context=True)

    def test_interval_region_out_of_bounds(self):
        """Test that OOB interval region raises clear ValueError, not beartype error."""
        import pytest

        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match=r"region \[10, 20\] exceeds sequence length 4"):
                slice_seq(pool, region=[10, 20])

    def test_keep_context_method(self):
        """Test keep_context via method call."""
        with pp.Party() as party:
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
            sliced = pool.slice_seq(region="orf", start=0, stop=3, keep_context=True)

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAAATGTTT"
