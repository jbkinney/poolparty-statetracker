"""Tests for sample operation - sample states from a pool."""

import pytest

import poolparty as pp
from poolparty.state_ops.sample import SampleOp, sample


class TestSampleFactory:
    """Test sample factory function."""

    def test_returns_pool(self):
        """Test that sample returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = sample(pool, num_seqs=3, seed=42)
            assert sampled is not None
            assert hasattr(sampled, "operation")

    def test_creates_sample_op(self):
        """Test that sample creates a SampleOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = sample(pool, num_seqs=3, seed=42)
            assert isinstance(sampled.operation, SampleOp)


class TestSampleNumStates:
    """Test state sampling num_states."""

    def test_num_states_less_than_parent(self):
        """Test sampling fewer states than parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")  # 5 states
            sampled = sample(pool, num_seqs=3, seed=42)
            assert sampled.num_states == 3

    def test_num_states_equal_to_parent(self):
        """Test sampling same number of states as parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")  # 5 states
            sampled = sample(pool, num_seqs=5, seed=42, with_replacement=False)
            assert sampled.num_states == 5

    def test_num_states_greater_with_replacement(self):
        """Test sampling more states than parent with replacement."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")  # 3 states
            sampled = sample(pool, num_seqs=10, seed=42, with_replacement=True)
            assert sampled.num_states == 10


class TestSampleOutput:
    """Test state sampling output."""

    def test_output_with_seq_states(self):
        """Test that output matches seq_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = sample(pool, seq_states=[0, 2, 4]).named("samp")

        df = sampled.generate_library(num_cycles=1)
        output_seqs = df["seq"].tolist()
        assert output_seqs == ["A", "C", "E"]

    def test_output_with_duplicates_in_seq_states(self):
        """Test that seq_states with duplicates works."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, seq_states=[0, 0, 1, 2, 2]).named("samp")

        df = sampled.generate_library(num_cycles=1)
        output_seqs = df["seq"].tolist()
        assert output_seqs == ["A", "A", "B", "C", "C"]

    def test_all_outputs_from_parent(self):
        """Test that all sampled outputs come from parent sequences."""
        with pp.Party() as party:
            seqs = ["A", "B", "C", "D", "E"]
            pool = pp.from_seqs(seqs, mode="sequential")
            sampled = sample(pool, num_seqs=3, seed=42).named("samp")

        df = sampled.generate_library(num_cycles=1)
        for seq in df["seq"].tolist():
            assert seq in seqs


class TestSampleDeterminism:
    """Test deterministic behavior with seeds."""

    def test_same_seed_same_result(self):
        """Test that same seed produces same result."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled1 = sample(pool1, num_seqs=3, seed=42).named("samp1")
        df1 = sampled1.generate_library(num_cycles=1)

        with pp.Party() as party:
            pool2 = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled2 = sample(pool2, num_seqs=3, seed=42).named("samp2")
        df2 = sampled2.generate_library(num_cycles=1)

        assert df1["seq"].tolist() == df2["seq"].tolist()

    def test_different_seed_different_result(self):
        """Test that different seeds produce different results."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["A", "B", "C", "D", "E", "F", "G", "H"], mode="sequential")
            sampled1 = sample(pool1, num_seqs=4, seed=42).named("samp1")
        df1 = sampled1.generate_library(num_cycles=1)

        with pp.Party() as party:
            pool2 = pp.from_seqs(["A", "B", "C", "D", "E", "F", "G", "H"], mode="sequential")
            sampled2 = sample(pool2, num_seqs=4, seed=123).named("samp2")
        df2 = sampled2.generate_library(num_cycles=1)

        # Very likely to be different with different seeds
        assert df1["seq"].tolist() != df2["seq"].tolist()


class TestSampleNoSeed:
    """Test behavior without explicit seed."""

    def test_no_seed_runs(self):
        """Test that sample works without explicit seed."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = sample(pool, num_seqs=3).named("samp")  # No seed

        df = sampled.generate_library(num_cycles=1)
        # Should still output 3 valid sequences
        assert len(df) == 3
        for seq in df["seq"].tolist():
            assert seq in ["A", "B", "C", "D", "E"]


class TestSampleCustomName:
    """Test SampleOp name parameter."""

    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, num_seqs=2, seed=42)
            assert sampled.operation.name.startswith("op[")
            assert ":sample" in sampled.operation.name

    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, num_seqs=2, seed=42).named("my_sample")
            assert sampled.name == "my_sample"


class TestSampleCompute:
    """Test SampleOp compute methods directly."""

    def test_compute_returns_parent_sequence(self):
        """Test compute returns parent sequence."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"], mode="sequential")
            sampled = sample(pool, seq_states=[0])

        output_seq, card = sampled.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert output_seq.string == "ACGT"


class TestSampleWithReplacement:
    """Test sample with/without replacement."""

    def test_with_replacement_allows_more_than_parent(self):
        """Test that with_replacement=True allows sampling more states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, num_seqs=10, seed=42, with_replacement=True).named("samp")

        assert sampled.num_states == 10
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == 10

    def test_without_replacement_valid(self):
        """Test that with_replacement=False works when num_states <= parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sampled = sample(pool, num_seqs=3, seed=42, with_replacement=False).named("samp")

        assert sampled.num_states == 3
        df = sampled.generate_library(num_cycles=1)
        # All sequences should be unique
        assert len(df["seq"].unique()) == 3

    def test_without_replacement_exceeds_raises(self):
        """Test that with_replacement=False raises when num_states > parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="exceeds parent"):
                sample(pool, num_seqs=10, with_replacement=False)


class TestSampleValidation:
    """Test sample validation."""

    def test_must_specify_num_seqs_or_seq_states(self):
        """Test that either num_seqs or seq_states must be specified."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="Must specify either"):
                sample(pool)

    def test_cannot_specify_both_num_seqs_and_seq_states(self):
        """Test that num_seqs and seq_states are mutually exclusive."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="Cannot specify both"):
                sample(pool, num_seqs=2, seq_states=[0, 1])

    def test_cannot_specify_seed_with_seq_states(self):
        """Test that seed cannot be used with seq_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="Cannot specify 'seed' with 'sampled_states'"):
                sample(pool, seq_states=[0, 1], seed=42)

    def test_seq_states_out_of_range_raises(self):
        """Test that seq_states with out-of-range values raises."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="out of range"):
                sample(pool, seq_states=[0, 1, 10])

    def test_seq_states_negative_raises(self):
        """Test that seq_states with negative values raises."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            with pytest.raises(ValueError, match="out of range"):
                sample(pool, seq_states=[0, -1, 2])


class TestSampleGetCopyParams:
    """Test SampleOp._get_copy_params method."""

    def test_get_copy_params_with_num_seqs(self):
        """Test _get_copy_params with num_seqs."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, num_seqs=2, seed=42, with_replacement=False)
            params = sampled.operation._get_copy_params()

            assert params["parent_pool"] is pool
            assert params["num_seqs"] == 2
            assert params["seq_states"] is None
            assert params["seed"] == 42
            assert params["with_replacement"] == False

    def test_get_copy_params_with_seq_states(self):
        """Test _get_copy_params with seq_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sampled = sample(pool, seq_states=[0, 2])
            params = sampled.operation._get_copy_params()

            assert params["parent_pool"] is pool
            assert params["num_seqs"] is None
            assert params["seq_states"] == [0, 2]
            assert params["seed"] is None
