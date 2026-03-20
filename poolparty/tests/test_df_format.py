"""Tests for DataFrame formatting utilities."""

import pandas as pd

import poolparty as pp
from poolparty.utils.df_utils import (
    counter_col_name,
    get_pools_reverse_topo,
)


class TestStateColName:
    """Test counter_col_name function."""

    def test_uses_counter_name_if_present(self):
        """Test that counter name is used when available."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state
            pool = pp.from_seqs(["AAA", "TTT"], mode="sequential")
            pool.state.name = "my_counter"

            result = counter_col_name(pool.state, 0)
            assert result == "my_counter"

    def test_uses_counter_id_if_no_name(self):
        """Test fallback to id_{counter.id} when no name."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state
            pool = pp.from_seqs(["AAA", "TTT"], mode="sequential")
            pool.state.name = ""  # Clear name

            result = counter_col_name(pool.state, 0)
            assert result == f"id_{pool.state.id}"

    def test_uses_index_fallback(self):
        """Test fallback to id_{index} when no name or id."""

        # Create a mock counter with no name and no id
        class MockState:
            name = ""
            id = None

        mock_counter = MockState()
        result = counter_col_name(mock_counter, 5)
        assert result == "id_5"


class TestGetPoolsReverseTopo:
    """Test get_pools_reverse_topo function."""

    def test_single_pool(self):
        """Test with a single pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            result = get_pools_reverse_topo({pool})

            assert len(result) == 1
            assert result[0] is pool

    def test_parent_child_order(self):
        """Test that children come before parents."""
        with pp.Party() as party:
            parent = pp.from_seqs(["AAA", "TTT"])
            child = pp.mutagenize(parent, num_mutations=1)

            result = get_pools_reverse_topo({parent, child})

            # Child should come before parent in reverse topo order
            child_idx = result.index(child)
            parent_idx = result.index(parent)
            assert child_idx < parent_idx

    def test_chain_of_pools(self):
        """Test with a chain of operations."""
        with pp.Party() as party:
            p1 = pp.from_seqs(["AAA"])
            p2 = pp.mutagenize(p1, num_mutations=1)
            p3 = pp.mutagenize(p2, num_mutations=1)

            result = get_pools_reverse_topo({p1, p2, p3})

            # Order should be: p3, p2, p1 (most derived first)
            assert result.index(p3) < result.index(p2)
            assert result.index(p2) < result.index(p1)


class TestIntegrationWithGenerate:
    """Test that generate_library produces correctly formatted output."""

    def test_generate_basic_output(self):
        """Test that generate_library produces name and seq columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT", "GGG"], mode="sequential")
            pool.name = "test_pool"

            df = pool.generate_library(num_cycles=1)

            # Should have 'name' and 'seq' columns by default
            assert "name" in df.columns
            assert "seq" in df.columns
            # Should have the sequences
            assert list(df["seq"]) == ["AAA", "TTT", "GGG"]

    def test_generate_with_cards(self):
        """Test generate with cards parameter on operation."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ["AAA", "TTT"], 
                mode="sequential",
                cards=["seq_index"]
            )

            df = pool.generate_library(num_cycles=1)

            # Should have design card column with op name prefix
            seq_index_cols = [c for c in df.columns if "seq_index" in c]
            assert len(seq_index_cols) == 1
