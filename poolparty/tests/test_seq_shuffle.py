"""Tests for the SeqShuffle operation."""

import numpy as np

import poolparty as pp
from poolparty.base_ops.shuffle_seq import SeqShuffleOp, shuffle_seq


class TestSeqShuffleFactory:
    """Factory function behavior."""

    def test_returns_pool(self):
        with pp.Party():
            pool = shuffle_seq("ACGT")
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_seqshuffle_op(self):
        with pp.Party():
            pool = shuffle_seq("ACGT")
            assert isinstance(pool.operation, SeqShuffleOp)


class TestSeqShuffleBehavior:
    """Core behavior tests."""

    def test_preserves_length(self):
        with pp.Party():
            pool = shuffle_seq("ACGTAC", region=[1, 5]).named("shuf")
        df = pool.generate_library(num_seqs=10, seed=123)
        for seq in df["seq"]:
            assert len(seq) == 6

    def test_random_variability(self):
        with pp.Party():
            # Use explicit num_states to get varied outputs
            pool = shuffle_seq("ACGTACGT", region=[0, 8], num_states=50).named("shuf")
        df = pool.generate_library(num_cycles=1, seed=42)
        assert df["seq"].nunique() > 5

    def test_random_num_states(self):
        with pp.Party():
            pool = shuffle_seq("ACGT", mode="random", num_states=10).named("shuf")
        assert pool.operation.num_states == 10
        df = pool.generate_library(num_cycles=1, seed=99)
        assert len(df) == 10

    def test_region_only_shuffled(self):
        with pp.Party():
            pool = shuffle_seq("ABCD", region=[1, 3]).named("shuf")
        df = pool.generate_library(num_seqs=5, seed=7)
        for seq in df["seq"]:
            assert seq[0] == "A"
            assert seq[3] == "D"
            middle = seq[1:3]
            assert sorted(middle) == sorted("BC")

    def test_zero_length_region_noop(self):
        with pp.Party():
            pool = shuffle_seq("ABCDE", region=[2, 2]).named("shuf")
        df = pool.generate_library(num_seqs=3, seed=1)
        assert set(df["seq"]) == {"ABCDE"}


class TestSeqShuffleDesignCard:
    """Design card correctness."""

    def test_compute_and_apply_permutation(self):
        with pp.Party():
            pool = shuffle_seq("WXYZ", region=[0, 4])
            rng = np.random.default_rng(42)
            output_seq, card = pool.operation.compute([pp.types.Seq.from_string("WXYZ")], rng)
        shuffled = output_seq.string
        assert set(shuffled) == set("WXYZ")
        assert len(shuffled) == 4
        # Verify the permutation is stored correctly
        assert "permutation" in card


class TestSeqShuffleWithMarker:
    """Tests for marker-based region specification."""

    def test_shuffle_marker_region(self):
        with pp.Party():
            pool = shuffle_seq("AA<r>BCDE</r>FF", region="r").named("shuf")
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            # AA and FF should be preserved
            assert seq.startswith("AA")
            assert seq.endswith("FF")
            # Middle should be a permutation of BCDE
            # Extract content between markers (markers are still present)
            middle = seq[2:-2]  # Skip AA and FF
            # After shuffle, the marker content is shuffled
            assert sorted(middle.replace("<r>", "").replace("</r>", "")) == sorted("BCDE")

    def test_shuffle_marker_region_remove_tags(self):
        """Test that _remove_tags=True removes marker tags."""
        with pp.Party():
            pool = shuffle_seq("AA<r>BCDE</r>FF", region="r", _remove_tags=True).named("shuf")
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            # AA and FF should be preserved
            assert seq.startswith("AA")
            assert seq.endswith("FF")
            # Marker tags should be removed
            assert "<r>" not in seq
            assert "</r>" not in seq
            # Middle should be a permutation of BCDE (no markers)
            middle = seq[2:-2]
            assert sorted(middle) == sorted("BCDE")

    def test_shuffle_marker_region_keep_marker(self):
        """Test that _remove_tags=False (default) keeps marker tags."""
        with pp.Party():
            pool = shuffle_seq("AA<r>BCDE</r>FF", region="r", _remove_tags=False).named("shuf")
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            # Marker tags should be present
            assert "<r>" in seq
            assert "</r>" in seq

    def test_shuffle_whole_sequence_with_none_region(self):
        with pp.Party():
            pool = shuffle_seq("ABCD").named("shuf")
        df = pool.generate_library(num_seqs=10, seed=123)
        # All results should be permutations of ABCD
        for seq in df["seq"]:
            assert sorted(seq) == sorted("ABCD")


class TestSeqShuffleStyling:
    """Tests for style parameter."""

    def test_style_shuffle_applied(self):
        """Test that style applies styling to shuffled characters."""
        with pp.Party():
            pool = shuffle_seq("ACGT", region=[1, 3], style="purple").named("shuf")
        df = pool.generate_library(num_seqs=3, seed=42)
        # Check that styles are present in the output
        for _, row in df.iterrows():
            assert "style" in row or hasattr(pool.operation, "_style")
            # The actual style application is tested via visual inspection or style checking
            # Here we just verify the parameter is accepted

    def test_style_shuffle_none(self):
        """Test that style=None (default) doesn't apply styling."""
        with pp.Party():
            pool = shuffle_seq("ACGT", region=[1, 3], style=None).named("shuf")
        df = pool.generate_library(num_seqs=3, seed=42)
        # Should work without errors
        assert len(df) == 3
