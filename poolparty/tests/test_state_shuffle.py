"""Tests for state_shuffle operation - randomly permute pool states."""

import pytest

import poolparty as pp
from poolparty.state_ops.state_shuffle import StateShuffleOp, state_shuffle


class TestStateShuffleFactory:
    """Test state_shuffle factory function."""

    def test_returns_pool(self):
        """Test that state_shuffle returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42)
            assert shuffled is not None
            assert hasattr(shuffled, "operation")

    def test_creates_state_shuffle_op(self):
        """Test that state_shuffle creates a StateShuffleOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42)
            assert isinstance(shuffled.operation, StateShuffleOp)


class TestStateShuffleNumStates:
    """Test state shuffling preserves num_states."""

    def test_preserves_num_states(self):
        """Test that state shuffling preserves num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")  # 5 states
            shuffled = state_shuffle(pool, seed=42)
            assert shuffled.num_states == 5

    def test_preserves_num_states_larger(self):
        """Test with larger pool."""
        with pp.Party() as party:
            seqs = [chr(65 + i) for i in range(20)]  # A-T
            pool = pp.from_seqs(seqs, mode="sequential")
            shuffled = state_shuffle(pool, seed=123)
            assert shuffled.num_states == 20


class TestStateShuffleOutput:
    """Test state shuffling output."""

    def test_output_is_permutation(self):
        """Test that output is a permutation of input states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42).named("sh")

        df = shuffled.generate_library(num_cycles=1)
        output_seqs = sorted(df["seq"].tolist())
        assert output_seqs == ["A", "B", "C", "D", "E"]

    def test_order_is_different(self):
        """Test that shuffled order is different from original."""
        with pp.Party() as party:
            seqs = ["A", "B", "C", "D", "E", "F", "G", "H"]
            pool = pp.from_seqs(seqs, mode="sequential")
            shuffled = state_shuffle(pool, seed=42).named("sh")

        df = shuffled.generate_library(num_cycles=1)
        output_seqs = df["seq"].tolist()
        # With 8 elements and seed=42, order should be different
        assert output_seqs != seqs


class TestStateShuffleDeterminism:
    """Test deterministic behavior with seeds."""

    def test_same_seed_same_result(self):
        """Test that same seed produces same result."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled1 = state_shuffle(pool1, seed=42).named("sh1")
        df1 = shuffled1.generate_library(num_cycles=1)

        with pp.Party() as party:
            pool2 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled2 = state_shuffle(pool2, seed=42).named("sh2")
        df2 = shuffled2.generate_library(num_cycles=1)

        assert df1["seq"].tolist() == df2["seq"].tolist()

    def test_different_seed_different_result(self):
        """Test that different seeds produce different results."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["A", "B", "C", "D", "E", "F", "G", "H"], mode="sequential")
            shuffled1 = state_shuffle(pool1, seed=42).named("sh1")
        df1 = shuffled1.generate_library(num_cycles=1)

        with pp.Party() as party:
            pool2 = pp.from_seqs(["A", "B", "C", "D", "E", "F", "G", "H"], mode="sequential")
            shuffled2 = state_shuffle(pool2, seed=123).named("sh2")
        df2 = shuffled2.generate_library(num_cycles=1)

        assert df1["seq"].tolist() != df2["seq"].tolist()


class TestStateShuffleNoSeed:
    """Test behavior without explicit seed."""

    def test_no_seed_runs(self):
        """Test that state_shuffle works without explicit seed."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            shuffled = state_shuffle(pool).named("sh")  # No seed

        df = shuffled.generate_library(num_cycles=1)
        # Should still be a valid permutation
        assert sorted(df["seq"].tolist()) == ["A", "B", "C", "D", "E"]


class TestStateShuffleCustomName:
    """Test StateShuffleOp name parameter."""

    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42)
            assert shuffled.operation.name.startswith("op[")
            assert ":state_shuffle" in shuffled.operation.name

    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42).named("my_shuffle")
            assert shuffled.name == "my_shuffle"


class TestStateShuffleCompute:
    """Test StateShuffleOp compute methods directly."""

    def test_compute_returns_parent_sequence(self):
        """Test compute returns parent sequence."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"], mode="sequential")
            shuffled = state_shuffle(pool, seed=42)

        output_seq, card = shuffled.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "ACGT"


class TestStateShuffleWithPermutation:
    """Test state_shuffle with custom permutation argument."""

    def test_with_permutation(self):
        """Test state_shuffle with custom permutation."""
        with pp.Party() as party:
            seqs = ["A", "B", "C", "D", "E"]
            pool = pp.from_seqs(seqs, mode="sequential")
            perm = [4, 3, 2, 1, 0]  # Reverse order
            shuffled = state_shuffle(pool, permutation=perm).named("sh")

        df = shuffled.generate_library(num_cycles=1)
        output_seqs = df["seq"].tolist()
        # With reverse permutation, order should be E, D, C, B, A
        assert output_seqs == ["E", "D", "C", "B", "A"]

    def test_with_permutation_identity(self):
        """Test state_shuffle with identity permutation preserves order."""
        with pp.Party() as party:
            seqs = ["A", "B", "C", "D", "E"]
            pool = pp.from_seqs(seqs, mode="sequential")
            perm = [0, 1, 2, 3, 4]
            shuffled = state_shuffle(pool, permutation=perm).named("sh")

        df = shuffled.generate_library(num_cycles=1)
        output_seqs = df["seq"].tolist()
        assert output_seqs == seqs

    def test_seed_and_permutation_raises(self):
        """Test that specifying both seed and permutation raises ValueError."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            with pytest.raises(ValueError, match="mutually exclusive"):
                state_shuffle(pool, seed=42, permutation=[0, 1, 2, 3, 4])

    def test_permutation_wrong_length_raises(self):
        """Test that permutation with wrong length raises ValueError."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            with pytest.raises(ValueError, match="length"):
                state_shuffle(pool, permutation=[0, 1, 2])

    def test_permutation_wrong_contents_raises(self):
        """Test that permutation with wrong contents raises ValueError."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            with pytest.raises(ValueError, match="must contain exactly"):
                state_shuffle(pool, permutation=[0, 1, 2, 3, 5])

    def test_permutation_stored_in_op(self):
        """Test that permutation is stored in StateShuffleOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            perm = [2, 0, 1]
            shuffled = state_shuffle(pool, permutation=perm)
            assert shuffled.operation.permutation == perm
