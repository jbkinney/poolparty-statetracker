"""Tests for the FromMotif operation."""

import numpy as np
import pandas as pd
import pytest

import poolparty as pp
from poolparty.base_ops.from_motif import FromMotifOp, from_motif


class TestFromMotifFactory:
    """Test from_motif factory function."""

    def test_returns_pool(self):
        """Test that from_motif returns a Pool."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df)
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_from_motif_op(self):
        """Test that from_motif creates a FromMotifOp."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df)
            assert isinstance(pool.operation, FromMotifOp)

    def test_with_names(self):
        """Test from_motif with custom names."""
        prob_df = pd.DataFrame({"A": [1.0], "C": [0.0], "G": [0.0], "T": [0.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, cards=["prob_state"]).named("mypool")

        df = pool.generate_library(num_seqs=1, seed=42)
        # Check for prob_state in design card columns (operation name is auto-generated)
        prob_cols = [c for c in df.columns if "prob_state" in c]
        assert len(prob_cols) > 0


class TestFromMotifRandomMode:
    """Test FromMotif in random mode."""

    def test_random_sampling(self):
        """Test random sampling from probability matrix."""
        prob_df = pd.DataFrame({"A": [0.25] * 4, "C": [0.25] * 4, "G": [0.25] * 4, "T": [0.25] * 4})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random").named("motif")

        df = pool.generate_library(num_seqs=100, seed=42)
        assert len(df) == 100
        # All chars should be from DNA alphabet
        for seq in df["seq"]:
            assert all(c in "ACGT" for c in seq)

    def test_biased_sampling(self):
        """Test that biased probabilities produce expected distribution."""
        # 100% A at position 0
        prob_df = pd.DataFrame({"A": [1.0], "C": [0.0], "G": [0.0], "T": [0.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random").named("motif")

        df = pool.generate_library(num_seqs=100, seed=42)
        # All should be 'A'
        assert all(seq == "A" for seq in df["seq"])

    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")
            assert pool.operation.num_states == 1

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        prob_df = pd.DataFrame({"A": [0.25] * 4, "C": [0.25] * 4, "G": [0.25] * 4, "T": [0.25] * 4})
        with pp.Party() as party:
            pool1 = from_motif(prob_df, mode="random").named("motif")
        df1 = pool1.generate_library(num_seqs=10, seed=42)

        with pp.Party() as party:
            pool2 = from_motif(prob_df, mode="random").named("motif")
        df2 = pool2.generate_library(num_seqs=10, seed=42)

        assert list(df1["seq"]) == list(df2["seq"])


class TestFromMotifRandomWithNumStates:
    """Test FromMotif in random mode with explicit num_states."""

    def test_random_mode_num_states(self):
        """Test that random mode with num_states has specified num_states."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random", num_states=10)
            assert pool.operation.num_states == 10


class TestFromMotifModeValidation:
    """Test FromMotif mode validation."""

    def test_sequential_mode_rejected(self):
        """Test that sequential mode is rejected."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="only supports mode='random'"):
                from_motif(prob_df, mode="sequential")

    def test_fixed_mode_rejected(self):
        """Test that fixed mode is rejected."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="only supports mode='random'"):
                from_motif(prob_df, mode="fixed")


class TestFromMotifValidation:
    """Test FromMotif validation."""

    def test_empty_prob_df_error(self):
        """Test error for empty DataFrame."""
        prob_df = pd.DataFrame()
        with pp.Party() as party:
            with pytest.raises(ValueError, match="non-empty DataFrame"):
                from_motif(prob_df)

    def test_invalid_column_error(self):
        """Test error for columns not in alphabet."""
        prob_df = pd.DataFrame({"A": [0.5], "X": [0.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Extra columns"):
                from_motif(prob_df)

    def test_nan_values_error(self):
        """Test error for NaN values."""
        prob_df = pd.DataFrame({"A": [0.5, np.nan], "T": [0.5, 0.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="NaN"):
                from_motif(prob_df)

    def test_negative_values_error(self):
        """Test error for negative values."""
        prob_df = pd.DataFrame({"A": [-0.5], "T": [1.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match=">= 0"):
                from_motif(prob_df)

    def test_zero_row_sum_error(self):
        """Test error for rows summing to zero."""
        prob_df = pd.DataFrame({"A": [0.0], "T": [0.0]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="must not sum to zero"):
                from_motif(prob_df)

    def test_works_with_default_party(self):
        """Test that from_motif works with default party context."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        pool = from_motif(prob_df)
        assert pool is not None
        assert hasattr(pool, "operation")


class TestFromMotifNormalization:
    """Test FromMotif probability normalization."""

    def test_auto_normalization(self):
        """Test that rows are auto-normalized to sum to 1."""
        # Rows sum to 2, should be normalized
        prob_df = pd.DataFrame({"A": [0.5, 1.0], "T": [0.5, 1.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")

        # Check internal prob_df is normalized
        row_sums = pool.operation.prob_df.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_missing_columns_filled_with_zeros(self):
        """Test that missing alphabet columns are filled with zeros."""
        # Only A and T specified, C and G should be filled with 0
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")

        # All alphabet chars should be present
        assert set(pool.operation.prob_df.columns) == {"A", "C", "G", "T"}
        # C and G should be 0
        assert pool.operation.prob_df["C"].iloc[0] == 0.0
        assert pool.operation.prob_df["G"].iloc[0] == 0.0


class TestFromMotifDNA:
    """Test FromMotif DNA handling."""

    def test_dna_columns(self):
        """Test that DNA columns are accepted."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")
            # prob_df columns should be normalized
            assert set(pool.operation.prob_df.columns) == {"A", "C", "G", "T"}

    def test_non_dna_column_rejected(self):
        """Test that non-DNA columns are rejected."""
        prob_df = pd.DataFrame({"A": [0.5], "U": [0.5]})
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Extra columns"):
                from_motif(prob_df)


class TestFromMotifDesignCards:
    """Test FromMotif design card output."""

    def test_prob_state_in_output(self):
        """Test prob_state is in output when requested via cards."""
        prob_df = pd.DataFrame({"A": [1.0], "C": [0.0], "G": [0.0], "T": [0.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, cards=["prob_state"]).named("mypool")

        df = pool.generate_library(num_seqs=1, seed=42)
        # Check for prob_state in design card columns (operation name is auto-generated)
        prob_cols = [c for c in df.columns if "prob_state" in c]
        assert len(prob_cols) > 0

    def test_prob_state_is_indices(self):
        """Test that prob_state contains indices."""
        # 100% A, so index should always be 0
        prob_df = pd.DataFrame({"A": [1.0, 1.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, cards=["prob_state"]).named("mypool")

        df = pool.generate_library(num_seqs=1, seed=42)
        # Find prob_state column (operation name is auto-generated)
        prob_cols = [c for c in df.columns if "prob_state" in c]
        assert len(prob_cols) > 0
        prob_state = df[prob_cols[0]].iloc[0]
        assert prob_state == [0, 0]

    def test_design_card_keys_defined(self):
        """Test design_card_keys is defined correctly."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df)
            assert "prob_state" in pool.operation.design_card_keys


class TestFromMotifCompute:
    """Test FromMotif compute methods directly."""

    def test_compute_requires_rng(self):
        """Test that compute requires RNG."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")

        with pytest.raises(RuntimeError, match="requires RNG"):
            pool.operation.compute([])

    def test_compute_produces_correct_sequence(self):
        """Test compute produces correct sequence."""
        prob_df = pd.DataFrame({"A": [0.25], "C": [0.25], "G": [0.25], "T": [0.25]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")

        # For testing, we can't directly set prob_state, but we can verify
        # the structure of the result
        rng = np.random.default_rng(42)
        output_seq, card = pool.operation.compute([], rng)
        assert "prob_state" in card
        assert output_seq.string in ["A", "C", "G", "T"]

    def test_compute_with_rng(self):
        """Test compute in random mode uses RNG correctly."""
        prob_df = pd.DataFrame({"A": [1.0]})  # Always A
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")

        rng = np.random.default_rng(42)
        output_seq, card = pool.operation.compute([], rng)
        assert output_seq.string == "A"


class TestFromMotifCustomName:
    """Test FromMotif operation and pool name parameters."""

    def test_default_operation_name(self):
        """Test default operation name is op[{id}]:from_motif."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df)
            assert pool.operation.name.startswith("op[")
            assert ":from_motif" in pool.operation.name

    def test_custom_operation_name(self):
        """Test custom operation name."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df).named("my_motif")
            assert pool.name == "my_motif"

    def test_custom_pool_name(self):
        """Test custom pool name."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df).named("my_pool")
            assert pool.name == "my_pool"


class TestFromMotifSeqLength:
    """Test FromMotif sequence length."""

    def test_seq_length_single_position(self):
        """Test sequence length for single position motif."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")
            assert pool.operation.seq_length == 1

    def test_seq_length_multiple_positions(self):
        """Test sequence length for multi-position motif."""
        prob_df = pd.DataFrame(
            {"A": [0.25] * 10, "C": [0.25] * 10, "G": [0.25] * 10, "T": [0.25] * 10}
        )
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")
            assert pool.operation.seq_length == 10


class TestFromMotifCopyParams:
    """Test FromMotif _get_copy_params method."""

    def test_copy_params_random_mode(self):
        """Test _get_copy_params in random mode."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random")
            params = pool.operation._get_copy_params()

        assert "prob_df" in params
        assert params["mode"] == "random"
        assert params["num_states"] is None
        assert params["name"] is None

    def test_copy_params_random_mode_with_num_states(self):
        """Test _get_copy_params in random mode with num_states."""
        prob_df = pd.DataFrame({"A": [0.5], "T": [0.5]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random", num_states=10)
            params = pool.operation._get_copy_params()

        assert params["mode"] == "random"
        assert params["num_states"] == 10


class TestFromMotifIntegration:
    """Test FromMotif integration with other operations."""

    def test_join_with_fixed_flanks(self):
        """Test joining motif with fixed sequences."""
        prob_df = pd.DataFrame({"A": [1.0], "C": [0.0], "G": [0.0], "T": [0.0]})
        with pp.Party() as party:
            left = pp.from_seq("ATG").named("left")
            var = from_motif(prob_df, mode="random").named("var")
            right = pp.from_seq("TAA").named("right")
            construct = pp.join([left, var, right]).named("construct")

        df = construct.generate_library(num_seqs=5, seed=42)
        # All sequences should be ATGATAA (ATG + A + TAA)
        assert all(seq == "ATGATAA" for seq in df["seq"])

    def test_multi_position_sampling(self):
        """Test multi-position probability sampling."""
        # Position 0: 100% A, Position 1: 100% T
        prob_df = pd.DataFrame({"A": [1.0, 0.0], "C": [0.0, 0.0], "G": [0.0, 0.0], "T": [0.0, 1.0]})
        with pp.Party() as party:
            pool = from_motif(prob_df, mode="random").named("motif")

        df = pool.generate_library(num_seqs=10, seed=42)
        # All should be 'AT'
        assert all(seq == "AT" for seq in df["seq"])
