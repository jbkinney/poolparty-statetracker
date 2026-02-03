"""Tests for num_states override functionality in sequential mode operations."""

import poolparty as pp
from poolparty.base_ops.from_iupac import from_iupac
from poolparty.base_ops.from_seqs import from_seqs
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.base_ops.recombine import recombine


class TestFromSeqsNumStatesOverride:
    """Test num_states override for from_seqs operation."""

    def test_clipping_num_states_less_than_natural(self):
        """Test clipping: num_states < natural count."""
        with pp.Party() as party:
            # Natural count is 4 (4 sequences)
            pool = from_seqs(["A", "B", "C", "D"], mode="sequential", num_states=2).named("test")

        # Should only see first 2 sequences
        df = pool.generate_library(num_seqs=4)
        assert list(df["seq"]) == ["A", "B", "A", "B"]
        assert pool.operation.num_states == 2
        assert pool.operation.natural_num_states == 4

    def test_cycling_num_states_greater_than_natural(self):
        """Test cycling: num_states > natural count."""
        with pp.Party() as party:
            # Natural count is 2 (2 sequences), but request 6 states
            pool = from_seqs(["A", "B"], mode="sequential", num_states=6).named("test")

        # Should cycle through A, B repeatedly for 6 states
        df = pool.generate_library(num_seqs=6)
        assert list(df["seq"]) == ["A", "B", "A", "B", "A", "B"]
        assert pool.operation.num_states == 6
        assert pool.operation.natural_num_states == 2

    def test_num_states_equals_natural(self):
        """Test when num_states equals natural count (no change)."""
        with pp.Party() as party:
            pool = from_seqs(["A", "B", "C"], mode="sequential", num_states=3).named("test")

        df = pool.generate_library(num_seqs=6)
        assert list(df["seq"]) == ["A", "B", "C", "A", "B", "C"]
        assert pool.operation.num_states == 3
        assert pool.operation.natural_num_states == 3

    def test_num_states_none_uses_natural(self):
        """Test that num_states=None uses natural count (default behavior)."""
        with pp.Party() as party:
            pool = from_seqs(["A", "B", "C"], mode="sequential").named("test")

        assert pool.operation.num_states == 3
        assert pool.operation.natural_num_states == 3


class TestMutagenizeNumStatesOverride:
    """Test num_states override for mutagenize operation."""

    def test_clipping_num_states_less_than_natural(self):
        """Test clipping: num_states < natural count."""
        with pp.Party() as party:
            # Natural count is 6 for 'AC' with 1 mutation (2 positions × 3 mutations)
            pool = mutagenize("AC", num_mutations=1, mode="sequential", num_states=3).named("test")

        # Should only see first 3 states
        df = pool.generate_library(num_seqs=6)
        first_half = list(df["seq"][:3])
        second_half = list(df["seq"][3:])
        assert first_half == second_half  # Cycles at 3
        assert pool.operation.num_states == 3
        assert pool.operation.natural_num_states == 6

    def test_cycling_num_states_greater_than_natural(self):
        """Test cycling: num_states > natural count."""
        with pp.Party() as party:
            # Natural count is 6 for 'AC' with 1 mutation
            pool = mutagenize("AC", num_mutations=1, mode="sequential", num_states=12).named("test")

        # Should cycle through 6 natural states twice
        df = pool.generate_library(num_seqs=12)
        first_half = list(df["seq"][:6])
        second_half = list(df["seq"][6:])
        assert first_half == second_half  # Perfect cycling
        assert pool.operation.num_states == 12
        assert pool.operation.natural_num_states == 6

    def test_natural_num_states_property(self):
        """Test natural_num_states is correctly computed."""
        with pp.Party() as party:
            # 4 positions × 3 mutations = 12 natural states
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential").named("test")

        assert pool.operation.num_states == 12
        assert pool.operation.natural_num_states == 12


class TestFromIupacNumStatesOverride:
    """Test num_states override for from_iupac operation."""

    def test_clipping_num_states_less_than_natural(self):
        """Test clipping: num_states < natural count."""
        with pp.Party() as party:
            # 'RY' = purine × pyrimidine = 2 × 2 = 4 natural states
            pool = from_iupac("RY", mode="sequential", num_states=2).named("test")

        df = pool.generate_library(num_seqs=4)
        # Should only see first 2 states, cycling
        assert pool.operation.num_states == 2
        assert pool.operation.natural_num_states == 4
        # First 2 states should repeat
        assert list(df["seq"][:2]) == list(df["seq"][2:])

    def test_cycling_num_states_greater_than_natural(self):
        """Test cycling: num_states > natural count."""
        with pp.Party() as party:
            # 'RY' = 2 × 2 = 4 natural states
            pool = from_iupac("RY", mode="sequential", num_states=8).named("test")

        df = pool.generate_library(num_seqs=8)
        first_half = list(df["seq"][:4])
        second_half = list(df["seq"][4:])
        assert first_half == second_half  # Cycles at 4
        assert pool.operation.num_states == 8
        assert pool.operation.natural_num_states == 4


class TestGetKmersNumStatesOverride:
    """Test num_states override for get_kmers operation."""

    def test_clipping_num_states_less_than_natural(self):
        """Test clipping: num_states < natural count."""
        with pp.Party() as party:
            # Length 2 k-mers: 4^2 = 16 natural states
            pool = get_kmers(2, mode="sequential", num_states=4).named("test")

        df = pool.generate_library(num_seqs=8)
        # Should only see first 4 k-mers, cycling
        first_half = list(df["seq"][:4])
        second_half = list(df["seq"][4:])
        assert first_half == second_half
        assert pool.operation.num_states == 4
        assert pool.operation.natural_num_states == 16

    def test_cycling_num_states_greater_than_natural(self):
        """Test cycling: num_states > natural count."""
        with pp.Party() as party:
            # Length 1 k-mers: 4^1 = 4 natural states
            pool = get_kmers(1, mode="sequential", num_states=8).named("test")

        df = pool.generate_library(num_seqs=8)
        first_half = list(df["seq"][:4])
        second_half = list(df["seq"][4:])
        assert first_half == second_half  # Cycles at 4
        assert pool.operation.num_states == 8
        assert pool.operation.natural_num_states == 4


class TestRecombineNumStatesOverride:
    """Test num_states override for recombine operation."""

    def test_clipping_num_states_less_than_natural(self):
        """Test clipping: num_states < natural count."""
        with pp.Party() as party:
            # 2 sources, length 4, 1 breakpoint
            # Natural: C(3,1) positions × 2 sources × 1 alternate = 3 × 2 × 1 = 6 states
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, mode="sequential", num_states=3
            ).named("test")

        df = pool.generate_library(num_seqs=6)
        first_half = list(df["seq"][:3])
        second_half = list(df["seq"][3:])
        assert first_half == second_half  # Cycles at 3
        assert pool.operation.num_states == 3
        assert pool.operation.natural_num_states == 6

    def test_cycling_num_states_greater_than_natural(self):
        """Test cycling: num_states > natural count."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, mode="sequential", num_states=12
            ).named("test")

        df = pool.generate_library(num_seqs=12)
        first_half = list(df["seq"][:6])
        second_half = list(df["seq"][6:])
        assert first_half == second_half  # Cycles at 6
        assert pool.operation.num_states == 12
        assert pool.operation.natural_num_states == 6


class TestNaturalNumStatesProperty:
    """Test the natural_num_states property across operations."""

    def test_natural_num_states_is_one_for_stateless_random_mode(self):
        """Test that natural_num_states is 1 for stateless random mode operations."""
        with pp.Party() as party:
            pool = from_seqs(["A", "B"], mode="random").named("test")

        assert pool.operation.natural_num_states == 1

    def test_natural_num_states_for_fixed_mode(self):
        """Test that natural_num_states equals num_states for fixed mode."""
        with pp.Party() as party:
            pool = from_seqs(["A"], mode="fixed").named("test")

        # Fixed mode has num_states=1, and natural_num_states defaults to same
        assert pool.operation.num_states == 1
        assert pool.operation.natural_num_states == 1

    def test_natural_num_states_preserved_with_override(self):
        """Test that natural_num_states reflects computed value even with override."""
        with pp.Party() as party:
            # Natural count is 4, override to 100
            pool = from_seqs(["A", "B", "C", "D"], mode="sequential", num_states=100).named("test")

        assert pool.operation.num_states == 100
        assert pool.operation.natural_num_states == 4
