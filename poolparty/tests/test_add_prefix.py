"""Tests for add_prefix operation."""

import re

import poolparty as pp
from poolparty.fixed_ops.add_prefix import AddPrefixOp, add_prefix


class TestAddPrefixBasic:
    """Test basic add_prefix functionality."""

    def test_returns_pool(self):
        """Test that add_prefix returns a Pool."""
        with pp.Party():
            root = pp.from_seq("ACGT")
            result = root.add_prefix("myprefix")
            assert result is not None
            assert hasattr(result, "operation")

    def test_creates_add_prefix_op(self):
        """Test that add_prefix creates an AddPrefixOp."""
        with pp.Party():
            root = pp.from_seq("ACGT")
            result = add_prefix(root, "myprefix")
            assert isinstance(result.operation, AddPrefixOp)

    def test_sequence_unchanged(self):
        """Test that sequence is passed through unchanged."""
        with pp.Party():
            pool = pp.from_seq("ACGT").add_prefix("test")

        df = pool.generate_library(num_cycles=1)
        assert df.loc[0, "seq"] == "ACGT"

    def test_prefix_in_name(self):
        """Test that prefix appears in sequence name."""
        with pp.Party():
            pool = pp.from_seq("ACGT").add_prefix("myprefix")

        df = pool.generate_library(num_cycles=1)
        assert "myprefix" in df.loc[0, "name"]


class TestAddPrefixChaining:
    """Test add_prefix in chains with other operations."""

    def test_chained_with_mutagenize(self):
        """Test add_prefix after mutagenize."""
        with pp.Party():
            pool = (
                pp.from_seq("ACGTACGT")
                .mutagenize(num_mutations=1, mode="sequential", prefix="mut")
                .add_prefix("final")
            )

        df = pool.generate_library(num_cycles=1)

        for idx, name in enumerate(df["name"]):
            # Check for zero-padded index (e.g., "mut_01" for idx=1)
            # Use regex to match mut_ followed by zero-padded number
            pattern = rf"mut_0*{idx}(?:\D|$)"
            assert re.search(pattern, name), f"Name should contain 'mut_' with index {idx}: {name}"
            assert "final" in name, f"Name should contain 'final': {name}"

    def test_chained_with_from_seq_prefix(self):
        """Test add_prefix combined with from_seq prefix."""
        with pp.Party():
            pool = pp.from_seq("ACGT", prefix="bg").add_prefix("after")

        df = pool.generate_library(num_cycles=1)
        name = df.loc[0, "name"]
        assert "bg" in name, f"Name should contain 'bg': {name}"
        assert "after" in name, f"Name should contain 'after': {name}"

    def test_multiple_add_prefix_calls(self):
        """Test multiple add_prefix calls in sequence."""
        with pp.Party():
            pool = pp.from_seq("ACGT").add_prefix("first").add_prefix("second").add_prefix("third")

        df = pool.generate_library(num_cycles=1)
        name = df.loc[0, "name"]
        assert "first" in name, f"Name should contain 'first': {name}"
        assert "second" in name, f"Name should contain 'second': {name}"
        assert "third" in name, f"Name should contain 'third': {name}"


class TestAddPrefixWithVariableOps:
    """Test add_prefix with variable operations."""

    def test_with_from_seqs(self):
        """Test add_prefix with from_seqs."""
        with pp.Party():
            pool = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential", prefix="seq").add_prefix(
                "tagged"
            )

        df = pool.generate_library(num_cycles=1)

        for idx, name in enumerate(df["name"]):
            assert f"seq_{idx}" in name, f"Name should contain 'seq_{idx}': {name}"
            assert "tagged" in name, f"Name should contain 'tagged': {name}"

    def test_sequence_integrity(self):
        """Test that sequences are unchanged after add_prefix."""
        with pp.Party():
            original_seqs = ["AAAA", "CCCC", "GGGG"]
            pool = pp.from_seqs(original_seqs, mode="sequential").add_prefix("test")

        df = pool.generate_library(num_cycles=1)

        for idx, seq in enumerate(original_seqs):
            assert df.loc[idx, "seq"] == seq, f"Sequence {idx} should be unchanged"


class TestAddPrefixFactoryFunction:
    """Test add_prefix factory function directly."""

    def test_factory_function(self):
        """Test calling add_prefix as factory function."""
        with pp.Party():
            root = pp.from_seq("ACGT")
            result = pp.add_prefix(root, "factory_test")

        df = result.generate_library(num_cycles=1)
        assert "factory_test" in df.loc[0, "name"]
