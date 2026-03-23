"""Tests for filter operations."""

import pandas as pd
import pytest

import poolparty as pp
from poolparty.base_ops.filter_seq import FilterOp, filter
from poolparty.types import NullSeq, is_null_seq


class TestNullSeq:
    """Test NullSeq class."""

    def test_nullseq_singleton(self):
        """Test that NullSeq is a singleton."""
        n1 = NullSeq()
        n2 = NullSeq()
        assert n1 is n2

    def test_nullseq_is_falsy(self):
        """Test that NullSeq is falsy."""
        assert not NullSeq()
        assert bool(NullSeq()) is False

    def test_nullseq_len_zero(self):
        """Test that NullSeq has length 0."""
        assert len(NullSeq()) == 0

    def test_nullseq_repr(self):
        """Test NullSeq repr."""
        assert repr(NullSeq()) == "NullSeq()"

    def test_is_null_seq(self):
        """Test is_null_seq helper."""
        assert is_null_seq(NullSeq())
        with pp.Party():
            pool = pp.from_seqs(["AAA"])
            df = pool.generate_library(num_seqs=1)
        # Normal seq should not be null
        from poolparty.utils.seq import Seq

        assert not is_null_seq(Seq.from_string("AAA"))


class TestFilterBasic:
    """Test basic filter functionality."""

    def test_filter_keeps_matching(self):
        """Test that filter keeps sequences matching predicate."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))
            df = filtered.generate_library(num_seqs=3)

            assert df.loc[0, "seq"] == "AAAA"
            assert pd.isna(df.loc[1, "seq"])
            assert pd.isna(df.loc[2, "seq"])

    def test_filter_with_discard(self):
        """Test discard_null_seqs=True removes filtered rows."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "GGGG", "ACGT", "TGCA"], mode="sequential")
            # Keep sequences starting with A (indices 0, 3)
            filtered = root.filter(lambda s: s.startswith("A"))
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 2
            assert df.loc[0, "seq"] == "AAAA"
            assert df.loc[1, "seq"] == "ACGT"

    def test_filter_all_pass(self):
        """Test filter where all sequences pass."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "ACGT", "ATGC"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))

            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 3

    def test_filter_pool_method(self):
        """Test Pool.filter() method works."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
            filtered = root.filter(lambda s: s == "AAA")
            df = filtered.generate_library(num_seqs=3)

            assert df.loc[0, "seq"] == "AAA"
            assert pd.isna(df.loc[1, "seq"])
            assert pd.isna(df.loc[2, "seq"])


class TestFilterPropagation:
    """Test NullSeq propagation through DAG."""

    def test_nullseq_propagates(self):
        """Test NullSeq propagates through downstream operations."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))
            # Apply a downstream operation
            upper = filtered.upper()

            df = upper.generate_library(num_seqs=3)

            # Only first row should have valid sequence
            assert df.loc[0, "seq"] == "AAAA"
            assert pd.isna(df.loc[1, "seq"])
            assert pd.isna(df.loc[2, "seq"])

    def test_chained_filters(self):
        """Test multiple filters in sequence."""
        with pp.Party():
            seqs = ["AAAA", "AATT", "CCCC", "GGGG", "AACC"]
            root = pp.from_seqs(seqs, mode="sequential")
            f1 = root.filter(lambda s: s.startswith("A"))  # Keeps 0, 1, 4
            f2 = f1.filter(lambda s: s.endswith("A"))  # Keeps 0

            df = f2.generate_library(num_seqs=1, discard_null_seqs=True)

            assert len(df) == 1
            assert df.loc[0, "seq"] == "AAAA"


class TestFilterEdgeCases:
    """Test edge cases and error handling."""

    def test_discard_with_num_cycles(self):
        """Test that discard_null_seqs works with num_cycles."""
        with pp.Party():
            root = pp.from_seqs(["AA", "CC", "GG"], mode="sequential")
            filtered = root.filter(lambda s: s != "CC")

            df = filtered.generate_library(num_cycles=1, discard_null_seqs=True)
            assert len(df) == 2
            assert set(df["seq"]) == {"AA", "GG"}

    def test_state_space_exhaustion_warning(self):
        """Test warning when state space exhausted before quota."""
        with pp.Party():
            root = pp.from_seqs(["A", "C", "G"], mode="sequential")  # 3 sequences
            filtered = root.filter(lambda s: s == "A")  # Only 1 passes

            # Warning is triggered when can't fulfill quota
            with pytest.warns(UserWarning):
                df = filtered.generate_library(
                    num_seqs=5,
                    discard_null_seqs=True,
                    max_iterations=100,  # High enough to hit state exhaustion first
                )

            # Should only have 1 valid sequence
            assert len(df) == 1

    def test_max_iterations_warning(self):
        """Test warning when max_iterations reached before quota or state exhaustion."""
        with pp.Party():
            # Create large state space so max_iterations is hit first
            root = pp.from_seqs(["A"] * 1000, mode="sequential")
            # Filter that rejects everything
            filtered = root.filter(lambda s: False)

            # max_iterations=50 will be hit before state space (1000) is exhausted
            with pytest.warns(UserWarning, match="max_iterations"):
                df = filtered.generate_library(
                    num_seqs=10,
                    discard_null_seqs=True,
                    max_iterations=50,
                )

            assert len(df) == 0

    def test_min_acceptance_rate_warning(self):
        """Test warning when acceptance rate too low."""
        with pp.Party():
            # Create pool where ~10% pass
            seqs = ["A"] + ["C"] * 9  # 10% start with A
            root = pp.from_seqs(seqs * 10, mode="sequential")  # 100 total
            filtered = root.filter(lambda s: s.startswith("A"))

            with pytest.warns(UserWarning, match="Acceptance rate"):
                df = filtered.generate_library(
                    num_seqs=1000,
                    discard_null_seqs=True,
                    min_acceptance_rate=0.5,  # Expect 50%, actual ~10%
                    attempts_per_rate_assessment=50,
                    max_iterations=200,
                )

    def test_filter_none_pass(self):
        """Test filter where no sequences pass."""
        with pp.Party():
            root = pp.from_seqs(["CCCC", "GGGG", "TTTT"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))

            # Warning is triggered when no sequences pass
            with pytest.warns(UserWarning):
                df = filtered.generate_library(
                    num_seqs=3,
                    discard_null_seqs=True,
                    max_iterations=100,  # Set high to ensure state exhaustion triggers
                )

            assert len(df) == 0


class TestFilterDesignCards:
    """Test design card behavior with filters."""

    def test_null_rows_have_none_name(self):
        """Test that filtered rows have None as name."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC"], mode="sequential", prefix="seq")
            filtered = root.filter(lambda s: s.startswith("A"))

            df = filtered.generate_library(num_seqs=2)

            # First row should have name, second should be None/nan
            assert pd.notna(df.loc[0, "name"]) or df.loc[0, "name"] == "seq_0"
            assert pd.isna(df.loc[1, "name"])

    def test_filter_design_card_reports_passed(self):
        """Test that filter operation reports passed status in design card when cards requested."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            # Use cards parameter to request design card output
            filtered = root.filter(lambda s: s.startswith("A"), cards=["passed"])

            df = filtered.generate_library(num_seqs=2)

            # Check design card has passed field
            passed_cols = [c for c in df.columns if "passed" in c]
            assert len(passed_cols) > 0
            # First should pass, second should fail (use == instead of is for numpy types)
            assert df[passed_cols[0]].iloc[0] == True
            assert df[passed_cols[0]].iloc[1] == False


class TestFilterFactory:
    """Test filter factory function."""

    def test_filter_returns_pool(self):
        """Test that filter returns a Pool."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])
            filtered = filter(root, lambda s: True)
            assert filtered is not None
            assert hasattr(filtered, "operation")

    def test_filter_creates_filter_op(self):
        """Test that filter creates a FilterOp."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])
            filtered = filter(root, lambda s: True)
            assert isinstance(filtered.operation, FilterOp)
