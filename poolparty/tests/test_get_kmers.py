"""Tests for the GetKmers operation."""

import numpy as np
import pytest

import poolparty as pp
from poolparty.base_ops.get_kmers import GetKmersOp, get_kmers


class TestGetKmersFactory:
    """Test get_kmers factory function."""

    def test_returns_pool(self):
        """Test that get_kmers returns a Pool."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_get_kmers_op(self):
        """Test that get_kmers creates a GetKmersOp."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert isinstance(pool.operation, GetKmersOp)


class TestGetKmersSequentialMode:
    """Test GetKmers in sequential mode."""

    def test_sequential_all_kmers(self):
        """Test sequential mode generates all k-mers in order."""
        with pp.Party() as party:
            pool = get_kmers(length=2, mode="sequential").named("kmer")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 16  # 4^2 = 16 DNA 2-mers
        assert df["seq"].iloc[0] == "AA"
        assert df["seq"].iloc[-1] == "TT"

    def test_sequential_dna_2mers(self):
        """Test sequential mode with DNA 2-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=2, mode="sequential").named("kmer")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 16  # 4^2 = 16
        # First should be 'AA', last should be 'TT'
        assert df["seq"].iloc[0] == "AA"
        assert df["seq"].iloc[-1] == "TT"

    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = get_kmers(length=1, mode="sequential").named("kmer")

        df = pool.generate_library(num_seqs=8)
        assert list(df["seq"]) == ["A", "C", "G", "T", "A", "C", "G", "T"]

    def test_sequential_num_states(self):
        """Test num_states calculation."""
        with pp.Party() as party:
            pool = get_kmers(length=3, mode="sequential")
            assert pool.operation.num_states == 64  # 4^3


class TestGetKmersRandomMode:
    """Test GetKmers in random mode."""

    def test_random_sampling(self):
        """Test random sampling of k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=5, mode="random").named("kmer")

        df = pool.generate_library(num_seqs=100, seed=42)
        assert len(df) == 100
        # All should be valid 5-mers
        for kmer in df["seq"]:
            assert len(kmer) == 5
            assert all(c in "ACGT" for c in kmer)

    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random").named("kmer")

        df = pool.generate_library(num_seqs=100, seed=42)
        unique_kmers = df["seq"].nunique()
        assert unique_kmers > 50  # Should be quite varied

    def test_random_reproducible(self):
        """Test that random mode is reproducible with seed."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random").named("kmer")

        df1 = pool.generate_library(num_seqs=10, seed=42, init_state=0)
        df2 = pool.generate_library(num_seqs=10, seed=42, init_state=0)

        assert list(df1["seq"]) == list(df2["seq"])

    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random")
            assert pool.operation.num_states == 1


class TestGetKmersDNA:
    """Test GetKmers with dna_utils."""

    def test_dna_kmers(self):
        """Test DNA k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=3, mode="sequential").named("kmer")

        df = pool.generate_library(num_seqs=10)
        for kmer in df["seq"]:
            assert all(c in "ACGT" for c in kmer)


class TestGetKmersStateToKmer:
    """Test the _state_to_kmer conversion method."""

    def test_state_to_kmer_dna(self):
        """Test state to k-mer conversion for dna_utils."""
        with pp.Party() as party:
            pool = get_kmers(length=2, mode="sequential")
            op = pool.operation

            assert op._value_to_kmer(0) == "AA"
            assert op._value_to_kmer(1) == "AC"
            assert op._value_to_kmer(2) == "AG"
            assert op._value_to_kmer(3) == "AT"
            assert op._value_to_kmer(4) == "CA"


class TestGetKmersDesignCards:
    """Test GetKmers design card output."""

    def test_kmer_index_in_output(self):
        """Test kmer_index is in output when cards requested."""
        with pp.Party() as party:
            pool = get_kmers(length=2, mode="sequential", cards=["kmer_index"]).named("mypool")

        df = pool.generate_library(num_seqs=4)
        # Find kmer_index column (operation name is auto-generated)
        kmer_index_cols = [c for c in df.columns if "kmer_index" in c]
        assert len(kmer_index_cols) > 0
        assert list(df[kmer_index_cols[0]]) == [0, 1, 2, 3]

    def test_design_card_keys_defined(self):
        """Test design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert "kmer_index" in pool.operation.design_card_keys


class TestGetKmersErrors:
    """Test GetKmers error handling."""

    def test_length_zero_error(self):
        """Test error for length=0."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="length must be >= 1"):
                get_kmers(length=0)

    def test_length_negative_error(self):
        """Test error for negative length."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="length must be >= 1"):
                get_kmers(length=-1)

    def test_works_with_default_party(self):
        """Test that get_kmers works with default party context."""
        pool = get_kmers(length=4, mode="random")
        assert pool is not None
        assert hasattr(pool, "operation")


class TestGetKmersLargeSpace:
    """Test GetKmers with large state spaces."""

    def test_large_kmer_random_mode(self):
        """Test large k-mer space works in random mode."""
        with pp.Party() as party:
            # 4^20 = huge number, but random mode should work
            pool = get_kmers(length=20, mode="random").named("kmer")

        df = pool.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        for kmer in df["seq"]:
            assert len(kmer) == 20

    def test_large_kmer_random_num_states_is_one(self):
        """Test that random mode with large k-mer still has num_states=1."""
        with pp.Party() as party:
            # Random mode has num_states=1
            pool = get_kmers(length=20, mode="random")
            assert pool.operation.num_states == 1


class TestGetKmersCompute:
    """Test GetKmers compute methods directly."""

    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = get_kmers(length=2, mode="sequential")

        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([])
        assert output_seq.string == "AA"
        assert card["kmer_index"] == 0

        pool.operation.state._value = 1
        output_seq, card = pool.operation.compute([])
        assert output_seq.string == "AC"
        assert card["kmer_index"] == 1

    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random")

        rng = np.random.default_rng(42)
        output_seq, card = pool.operation.compute([], rng)
        assert len(output_seq.string) == 4
        assert all(c in "ACGT" for c in output_seq.string)


class TestGetKmersCustomName:
    """Test GetKmers operation name parameter."""

    def test_default_operation_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert pool.operation.name.startswith("op[")
            assert ":get_kmers" in pool.operation.name

    def test_custom_operation_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = get_kmers(length=4).named("barcode")
            assert pool.name == "barcode"

    def test_custom_name_in_design_card(self):
        """Test design card columns are present (operation name is auto-generated)."""
        with pp.Party() as party:
            pool = get_kmers(length=4, cards=["kmer_index", "kmer"]).named("mypool")

        df = pool.generate_library(num_seqs=1, seed=42)
        # Find kmer_index and kmer columns (operation name is auto-generated)
        kmer_index_cols = [c for c in df.columns if "kmer_index" in c]
        kmer_cols = [c for c in df.columns if "kmer" in c and "kmer_index" not in c]
        assert len(kmer_index_cols) > 0
        assert len(kmer_cols) > 0
