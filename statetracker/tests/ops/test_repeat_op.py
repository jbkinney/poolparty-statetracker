"""Tests for RepeatOp and repeat_state()."""

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from statetracker import Manager, RepeatOp, State, repeat, stack


class TestRepeatOp:
    """Test RepeatOp class directly."""

    def test_repeat_co_op_compute_num_states(self):
        op = RepeatOp(times=3)
        assert op.compute_num_states((2,)) == 6
        assert op.compute_num_states((5,)) == 15

    def test_repeat_co_op_decompose(self):
        op = RepeatOp(times=3)
        # For a state with 2 states repeated 3 times:
        # States 0-5 map to parent states: 0, 1, 0, 1, 0, 1
        assert op.decompose(0, (2,)) == (0,)
        assert op.decompose(1, (2,)) == (1,)
        assert op.decompose(2, (2,)) == (0,)
        assert op.decompose(3, (2,)) == (1,)
        assert op.decompose(4, (2,)) == (0,)
        assert op.decompose(5, (2,)) == (1,)

    def test_repeat_co_op_inactive(self):
        op = RepeatOp(times=3)
        assert op.decompose(None, (2,)) == (None,)


class TestRepeatState:
    """Test repeat_state() function."""

    def test_repeat_state_num_states(self):
        """repeat_state(A, 3) has correct num_states."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = repeat(A, 3, name="B")
            assert B.num_values == 6

    def test_repeat_state_iteration(self):
        """Iterating repeat_state cycles through A's states multiple times."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = repeat(A, 3, name="B")

            results = []
            for b_state in B:
                results.append((b_state, A.value))

            # A cycles through 0, 1 three times
            expected = [
                (0, 0),
                (1, 1),  # First cycle
                (2, 0),
                (3, 1),  # Second cycle
                (4, 0),
                (5, 1),  # Third cycle
            ]
            assert results == expected

    def test_repeat_state_a_never_inactive(self):
        """A never becomes None during iteration of repeat_state."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = repeat(A, 4, name="B")

            for _ in B:
                assert A.value is not None
                assert A.is_active

    def test_repeat_state_times_one(self):
        """repeat_state(A, 1) works correctly."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = repeat(A, 1, name="B")

            assert B.num_values == 3

            results = []
            for b_state in B:
                results.append((b_state, A.value))

            expected = [(0, 0), (1, 1), (2, 2)]
            assert results == expected

    def test_repeat_state_times_zero_raises(self):
        """repeat_state with times=0 raises ValueError."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat(A, 0)

    def test_repeat_state_negative_times_raises(self):
        """repeat_state with negative times raises ValueError."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat(A, -1)

    def test_repeat_state_non_state_raises(self):
        """repeat_state with non-State raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            repeat("not a state", 3)

    def test_repeat_state_with_name(self):
        """repeat_state with name parameter."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = repeat(A, 3, name="Repeated")
            assert B.name == "Repeated"

    def test_repeat_state_composition(self):
        """repeat_state can be composed with other operations."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")

            # Repeat A twice, then sum with B
            A_rep = repeat(A, 2, name="A_rep")
            C = stack([B, A_rep])
            C.name = "C"

            assert C.num_values == 4 + 3  # 7

            # Parents are in order as provided: (B, A_rep)
            # First 3 states: B cycles
            # Last 4 states: A_rep cycles (A cycles twice)
            results = []
            for c_state in C:
                results.append((c_state, A.value, B.value))

            expected = [
                (0, None, 0),  # B active
                (1, None, 1),  # B active
                (2, None, 2),  # B active
                (3, 0, None),  # A_rep active, A=0
                (4, 1, None),  # A_rep active, A=1
                (5, 0, None),  # A_rep active, A=0
                (6, 1, None),  # A_rep active, A=1
            ]
            assert results == expected
