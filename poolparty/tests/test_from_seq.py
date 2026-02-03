"""Tests for the FromSeq operation."""

import poolparty as pp
from poolparty.fixed_ops.fixed import FixedOp
from poolparty.fixed_ops.from_seq import from_seq


class TestFromSeqFactory:
    """Test from_seq factory function."""

    def test_returns_pool(self):
        """Test that from_seq returns a Pool."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_fixed_op(self):
        """Test that from_seq creates a FixedOp."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert isinstance(pool.operation, FixedOp)

    def test_mode_is_fixed(self):
        """Test that the operation mode is always 'fixed'."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert pool.operation.mode == "fixed"


class TestFromSeqGeneration:
    """Test sequence generation from FromSeq."""

    def test_generates_same_sequence(self):
        """Test that the same sequence is generated repeatedly."""
        with pp.Party() as party:
            pool = from_seq("ATGC").named("seq")

        df = pool.generate_library(num_seqs=5)
        assert list(df["seq"]) == ["ATGC", "ATGC", "ATGC", "ATGC", "ATGC"]

    def test_num_states_is_one(self):
        """Test that num_states is always 1."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert pool.operation.num_states == 1
            assert pool.num_states == 1


class TestFromSeqDesignCards:
    """Test FromSeq design card output."""

    def test_no_design_card_keys(self):
        """Test that from_seq has no design card keys (uses FixedOp)."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert len(pool.operation.design_card_keys) == 0


class TestFromSeqCustomName:
    """Test FromSeq operation and pool name parameters."""

    def test_default_operation_name(self):
        """Test default operation name is 'from_seq'."""
        with pp.Party() as party:
            pool = from_seq("AAA")
            assert pool.operation.name.endswith(":from_seq")

    def test_custom_pool_name(self):
        """Test custom pool name."""
        with pp.Party() as party:
            pool = from_seq("AAA").named("my_pool")
            assert pool.name == "my_pool"


class TestFromSeqCompute:
    """Test FromSeq compute methods directly."""

    def test_compute_returns_sequence(self):
        """Test compute returns the sequence."""
        with pp.Party() as party:
            pool = from_seq("ATGC")

        output_seq, card = pool.operation.compute([])
        assert output_seq.string == "ATGC"
        # FixedOp returns empty design card
        assert card == {}
