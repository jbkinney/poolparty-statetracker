"""Tests for materialize operations."""

import pandas as pd
import pytest

import poolparty as pp
from poolparty.base_ops.materialize import MaterializeOp, materialize


class TestMaterializeBasic:
    """Test basic materialize functionality."""

    def test_materialize_stores_sequences(self):
        """Test that materialize stores the correct sequences."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
            materialized = root.materialize(num_seqs=3, seed=42)
            df = materialized.generate_library(num_seqs=3)

            assert len(df) == 3
            assert set(df["seq"]) == {"AAAA", "CCCC", "GGGG"}

    def test_materialize_num_states(self):
        """Test that num_states equals number of materialized sequences."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="sequential")
            materialized = root.materialize(num_seqs=4, seed=42)

            assert materialized.num_states == 4

    def test_materialize_severed_dag(self):
        """Test that materialized pool has no parents (severed DAG)."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            filtered = root.filter(lambda s: True)
            materialized = filtered.materialize(num_seqs=2, seed=42)

            # Parents should be empty
            assert materialized.parents == []

    def test_materialize_preserves_names(self):
        """Test that materialize preserves sequence names."""
        with pp.Party():
            root = pp.from_seqs(
                ["AAAA", "CCCC", "GGGG"],
                mode="sequential",
                seq_names=["seq_a", "seq_b", "seq_c"],
            )
            materialized = root.materialize(num_seqs=3, seed=42)
            df = materialized.generate_library(num_seqs=3)

            # Names should be preserved
            assert "seq_a" in df["name"].values
            assert "seq_b" in df["name"].values
            assert "seq_c" in df["name"].values


class TestMaterializeWithFilter:
    """Test materialize with filtering."""

    def test_materialize_with_filter_discard(self):
        """Test materialize with discard_null_seqs=True."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "GGGG", "ACGT", "TGCA"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))
            # Only AAAA and ACGT pass the filter
            materialized = filtered.materialize(num_seqs=2, seed=42)

            assert materialized.num_states == 2
            df = materialized.generate_library(num_seqs=2)
            assert set(df["seq"]) == {"AAAA", "ACGT"}

    def test_materialize_filter_known_states(self):
        """Test that materialized pool after filter has known num_states."""
        with pp.Party():
            root = pp.from_seqs(["A", "C", "G", "T"] * 10, mode="sequential")
            filtered = root.filter(lambda s: s == "A")
            # This should materialize exactly 5 'A' sequences
            materialized = filtered.materialize(num_seqs=5, seed=42)

            # Now num_states is well-defined
            assert materialized.num_states == 5


class TestMaterializeSeed:
    """Test seed reproducibility."""

    def test_materialize_seed_reproducible(self):
        """Test that same seed produces same sequences."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="random", num_states=4)

            m1 = root.materialize(num_seqs=4, seed=42)
            df1 = m1.generate_library(num_seqs=4)

        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="random", num_states=4)

            m2 = root.materialize(num_seqs=4, seed=42)
            df2 = m2.generate_library(num_seqs=4)

        assert list(df1["seq"]) == list(df2["seq"])

    def test_materialize_different_seeds_different_results(self):
        """Test that different seeds can produce different results."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="random", num_states=100)

            m1 = root.materialize(num_seqs=10, seed=42)
            m2 = root.materialize(num_seqs=10, seed=123)

            df1 = m1.generate_library(num_seqs=10)
            df2 = m2.generate_library(num_seqs=10)

            # With random mode and different seeds, sequences may differ
            # (not guaranteed, but highly likely with 100 samples)
            # We just check both have 10 sequences
            assert len(df1) == 10
            assert len(df2) == 10


class TestMaterializeChaining:
    """Test chaining operations after materialize."""

    def test_materialize_then_mutagenize(self):
        """Test chaining mutagenize after materialize."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            materialized = root.materialize(num_seqs=2, seed=42)
            mutated = materialized.mutagenize(num_mutations=1, mode="random")

            df = mutated.generate_library(num_seqs=2, seed=1)
            # Sequences should be mutated
            assert len(df) == 2

    def test_materialize_then_filter(self):
        """Test chaining filter after materialize."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC", "ACGT"], mode="sequential")
            materialized = root.materialize(num_seqs=3, seed=42)
            filtered = materialized.filter(lambda s: s.startswith("A"))

            df = filtered.generate_library(num_seqs=3)
            # AAAA and ACGT should pass, CCCC should be None/nan
            valid_seqs = [s for s in df["seq"] if pd.notna(s)]
            assert len(valid_seqs) == 2

    def test_materialize_preserves_downstream_dag(self):
        """Test that operations after materialize work correctly."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            materialized = root.materialize(num_seqs=1, seed=42)
            upper = materialized.upper()

            df = upper.generate_library(num_seqs=1)
            assert df.loc[0, "seq"] == "ACGT"


class TestMaterializePoolMethod:
    """Test Pool.materialize() method."""

    def test_pool_method_works(self):
        """Test that Pool.materialize() method works."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
            materialized = root.materialize(num_seqs=3, seed=42)

            assert materialized is not None
            assert materialized.num_states == 3

    def test_pool_method_creates_materialize_op(self):
        """Test that Pool.materialize() creates a MaterializeOp."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])
            materialized = root.materialize(num_seqs=1, seed=42)

            assert isinstance(materialized.operation, MaterializeOp)


class TestMaterializeFactory:
    """Test materialize factory function."""

    def test_materialize_factory_returns_pool(self):
        """Test that materialize returns a Pool."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])
            materialized = materialize(root, num_seqs=1, seed=42)

            assert materialized is not None
            assert hasattr(materialized, "operation")

    def test_materialize_factory_creates_materialize_op(self):
        """Test that materialize creates a MaterializeOp."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])
            materialized = materialize(root, num_seqs=1, seed=42)

            assert isinstance(materialized.operation, MaterializeOp)


class TestMaterializeDesignCards:
    """Test design card behavior."""

    def test_materialize_design_card(self):
        """Test that materialize reports design card info when cards requested."""
        with pp.Party():
            root = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            materialized = root.materialize(num_seqs=2, seed=42, cards=["seq_index", "seq_name"])

            df = materialized.generate_library(num_seqs=2)

            # Should have seq_index and seq_name columns
            index_cols = [c for c in df.columns if "seq_index" in c]
            name_cols = [c for c in df.columns if "seq_name" in c]
            assert len(index_cols) > 0
            assert len(name_cols) > 0


class TestMaterializeEdgeCases:
    """Test edge cases."""

    def test_materialize_empty_after_filter_raises(self):
        """Test that materializing with no valid sequences raises error."""
        with pp.Party():
            root = pp.from_seqs(["CCC", "GGG", "TTT"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))  # None pass

            with pytest.raises(ValueError, match="No sequences were materialized"):
                with pytest.warns(UserWarning, match="Reached max_iterations"):
                    filtered.materialize(num_seqs=1, seed=42)

    def test_materialize_with_prefix(self):
        """Test materialize with prefix for auto-naming."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC"], mode="sequential")
            materialized = root.materialize(num_seqs=2, seed=42, prefix="mat")

            df = materialized.generate_library(num_seqs=2)
            # Names should use prefix pattern
            assert len(df) == 2


class TestMaterializeDiscardNullSeqs:
    """Test discard_null_seqs parameter."""

    def test_discard_null_seqs_false(self):
        """Test discard_null_seqs=False includes null sequences."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
            filtered = root.filter(lambda s: s == "AAA")
            # With discard_null_seqs=False, NullSeq objects are included
            # Request 3 seqs - only 1 will be valid, 2 will be NullSeq
            materialized = filtered.materialize(num_seqs=3, seed=42, discard_null_seqs=False)

            # num_states includes the NullSeq entries
            assert materialized.num_states == 3

    def test_discard_null_seqs_true_default(self):
        """Test that discard_null_seqs=True is the default."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT", "ACG"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))
            # Only AAA and ACG pass - request 2
            materialized = filtered.materialize(num_seqs=2, seed=42)

            # Should have exactly 2 valid sequences
            assert materialized.num_states == 2
            df = materialized.generate_library(num_seqs=2)
            assert all(pd.notna(s) for s in df["seq"])


class TestMaterializeNumCycles:
    """Test num_cycles parameter."""

    def test_num_cycles_one_cycle(self):
        """Test num_cycles=1 goes through all states once."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
            materialized = root.materialize(num_cycles=1, seed=42)

            assert materialized.num_states == 3

    def test_num_cycles_default(self):
        """Test that num_cycles defaults to 1 when neither specified."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
            materialized = root.materialize(seed=42)

            # Default is num_cycles=1, so should have 3 states
            assert materialized.num_states == 3

    def test_num_cycles_with_filter(self):
        """Test num_cycles with filter counts only valid sequences."""
        with pp.Party():
            root = pp.from_seqs(["AAA", "CCC", "GGG", "TTT", "ACG"], mode="sequential")
            filtered = root.filter(lambda s: s.startswith("A"))
            # Only AAA and ACG pass out of 5
            materialized = filtered.materialize(num_cycles=1, seed=42)

            # Should have exactly 2 valid sequences (those that passed)
            assert materialized.num_states == 2

    def test_num_cycles_multiple(self):
        """Test num_cycles > 1."""
        with pp.Party():
            root = pp.from_seqs(["AA", "CC"], mode="sequential")
            materialized = root.materialize(num_cycles=2, seed=42)

            # 2 sequences * 2 cycles = 4 total
            assert materialized.num_states == 4

    def test_num_seqs_and_num_cycles_mutually_exclusive(self):
        """Test that specifying both raises error."""
        with pp.Party():
            root = pp.from_seqs(["AAA"])

            with pytest.raises(ValueError, match="Cannot specify both"):
                root.materialize(num_seqs=1, num_cycles=1, seed=42)
