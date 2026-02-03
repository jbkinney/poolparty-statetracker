"""Tests for state_slice - STATE slicing (selecting subset of states)."""

import poolparty as pp
from poolparty.state_ops.state_slice import StateSliceOp, state_slice


class TestStateSliceFactory:
    """Test state_slice factory function."""

    def test_returns_pool(self):
        """Test that state_slice returns a Pool."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state (required for slicing)
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = state_slice(pool, slice(1, 4))
            assert sliced is not None
            assert hasattr(sliced, "operation")

    def test_creates_state_slice_op(self):
        """Test that state_slice creates a StateSliceOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state (required for slicing)
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = state_slice(pool, slice(1, 4))
            assert isinstance(sliced.operation, StateSliceOp)


class TestStateSliceNumStates:
    """Test state slicing affects num_states correctly."""

    def test_slice_reduces_num_states(self):
        """Test that state slicing reduces num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")  # 5 states
            sliced = state_slice(pool, slice(1, 4))  # 3 states
            assert sliced.num_states == 3

    def test_slice_with_step(self):
        """Test state slicing with step."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E", "F"], mode="sequential")  # 6 states
            sliced = state_slice(pool, slice(None, None, 2))  # Every other: A, C, E -> 3 states
            assert sliced.num_states == 3


class TestStateSliceOutput:
    """Test state slicing output."""

    def test_correct_states_selected(self):
        """Test that correct states are selected."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = state_slice(pool, slice(1, 4)).named("sl")  # B, C, D

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["B", "C", "D"]

    def test_from_start(self):
        """Test state slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = state_slice(pool, slice(None, 3)).named("sl")  # A, B, C

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "B", "C"]

    def test_to_end(self):
        """Test state slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = state_slice(pool, slice(2, None)).named("sl")  # C, D, E

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["C", "D", "E"]


class TestStateSliceCustomName:
    """Test StateSliceOp name parameter."""

    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sliced = state_slice(pool, slice(0, 2))
            assert sliced.operation.name.startswith("op[")
            assert ":state_slice" in sliced.operation.name

    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sliced = state_slice(pool, slice(0, 2)).named("my_state_slice")
            assert sliced.name == "my_state_slice"


# =============================================================================
# Tests for Pool.__getitem__ - should now do STATE slicing
# =============================================================================


class TestPoolGetitemOperator:
    """Test Pool.__getitem__ operator (now does STATE slicing)."""

    def test_getitem_does_state_slice(self):
        """Test that pool[key] does state slicing, not sequence slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AA", "BB", "CC", "DD", "EE"], mode="sequential")  # 5 states
            sliced = pool[1:4]  # Should select states 1, 2, 3
            assert sliced.num_states == 3

    def test_getitem_with_slice(self):
        """Test pool[start:stop] for state slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AA", "BB", "CC", "DD", "EE"], mode="sequential")
            sliced = pool[1:4].named("sl")  # States 1, 2, 3 -> BB, CC, DD

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["BB", "CC", "DD"]

    def test_getitem_with_int(self):
        """Test pool[index] for single state selection."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AA", "BB", "CC"], mode="sequential")
            sliced = pool[1].named("sl")  # State 1 -> BB

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "BB"

    def test_getitem_negative_index(self):
        """Test pool[-1] for last state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AA", "BB", "CC"], mode="sequential")
            sliced = pool[-1].named("sl")  # Last state -> CC

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "CC"

    def test_getitem_uses_state_slice_op(self):
        """Test that Pool.__getitem__ creates StateSliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sliced = pool[0:2]
            assert isinstance(sliced.operation, StateSliceOp)

    def test_getitem_default_name(self):
        """Test that Pool.__getitem__ uses default state_slice name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            sliced = pool[0:2]
            assert sliced.operation.name.startswith("op[")
            assert ":state_slice" in sliced.operation.name
