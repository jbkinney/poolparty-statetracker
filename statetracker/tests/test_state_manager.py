"""Tests for Manager class."""

import pandas as pd
import pytest

from statetracker import Manager, State, product, stack


class TestManager:
    """Test Manager context manager."""

    def test_context_manager_basic(self):
        """Manager works as context manager and restores previous on exit."""
        previous = Manager._active_manager
        with Manager() as mgr:
            assert Manager._active_manager is mgr
        assert Manager._active_manager is previous

    def test_auto_registration(self):
        """States created in context auto-register."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            assert len(mgr._states) == 2
            assert A in mgr._states
            assert B in mgr._states

    def test_composite_states_register(self):
        """Composite states also register."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"
            # A, B, and C should all be registered
            assert len(mgr._states) == 3

    def test_state_requires_manager(self):
        """States created outside any Manager context raise error."""
        previous = Manager._active_manager
        Manager._active_manager = None
        try:
            with pytest.raises(RuntimeError, match="must be created within a Manager context"):
                State(num_values=2, name="A")
        finally:
            Manager._active_manager = previous

    def test_get_state_names(self):
        """get_state_names returns list of state names."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            names = mgr.get_all_names()
            assert names == ["A", "B", "C"]

    def test_get_state_names_with_auto_name(self):
        """Unnamed states get auto-generated names like id_N."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3)  # No name -> auto-named 'State[1]'

            names = mgr.get_all_names()
            assert names == ["A", "State[1]"]

    def test_get_state_by_name(self):
        """get_state returns state by name."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")

            assert mgr.get_by_name("A") is A
            assert mgr.get_by_name("B") is B

    def test_get_state_not_found(self):
        """get_state raises KeyError for unknown name."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")

            with pytest.raises(KeyError, match="No state with name 'X' found"):
                mgr.get_by_name("X")

    def test_reset_states_all(self):
        """reset_states resets all states when no arg given."""
        with Manager() as mgr:
            A = State(num_values=5, name="A")
            B = State(num_values=5, name="B")

            A.value = 3
            B.value = 4

            mgr.reset_all()

            assert A.value == 0
            assert B.value == 0

    def test_reset_states_specific(self):
        """reset_states resets only specified states."""
        with Manager() as mgr:
            A = State(num_values=5, name="A")
            B = State(num_values=5, name="B")

            A.value = 3
            B.value = 4

            mgr.reset_all([A])

            assert A.value == 0
            assert B.value == 4  # unchanged

    def test_inactivate_states_all(self):
        """inactivate_states inactivates all states when no arg given."""
        with Manager() as mgr:
            A = State(num_values=5, name="A")
            B = State(num_values=5, name="B")

            A.value = 2
            B.value = 3

            mgr.inactivate_all()

            assert A.value is None
            assert B.value is None

    def test_inactivate_states_specific(self):
        """inactivate_states inactivates only specified states."""
        with Manager() as mgr:
            A = State(num_values=5, name="A")
            B = State(num_values=5, name="B")

            A.value = 2
            B.value = 3

            mgr.inactivate_all([A])

            assert A.value is None
            assert B.value == 3  # unchanged

    def test_get_iteration_df_basic(self):
        """get_iteration_df returns DataFrame of states."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            df = mgr.get_iteration_df(C)

            assert isinstance(df, pd.DataFrame)
            assert df.index.name == "C"
            assert list(df.columns) == ["A", "B"]
            assert len(df) == 6

    def test_get_iteration_df_values(self):
        """get_iteration_df has correct state values."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            df = mgr.get_iteration_df(C)

            # Check expected values for product state
            # C=0: A=0, B=0; C=1: A=1, B=0; ...
            assert df["A"].tolist() == [0, 1, 0, 1, 0, 1]
            assert df["B"].tolist() == [0, 0, 1, 1, 2, 2]
            assert df.index.tolist() == [0, 1, 2, 3, 4, 5]

    def test_get_iteration_df_specific_states(self):
        """get_iteration_df with specific states only shows those."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            df = mgr.get_iteration_df(C, states=[A, C])

            assert list(df.columns) == ["A"]
            assert len(df) == 6

    def test_get_iteration_df_stack_shows_inactive(self):
        """get_iteration_df shows NaN for inactive states in sum."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = stack([A, B])
            C.name = "C"

            df = mgr.get_iteration_df(C)

            # First 2 rows: A active, B inactive (NaN in DataFrame)
            assert df["A"].iloc[0] == 0
            assert pd.isna(df["B"].iloc[0])
            assert df["A"].iloc[1] == 1
            assert pd.isna(df["B"].iloc[1])

            # Last 3 rows: A inactive (NaN in DataFrame), B active
            assert pd.isna(df["A"].iloc[2])
            assert df["B"].iloc[2] == 0
            assert pd.isna(df["A"].iloc[4])
            assert df["B"].iloc[4] == 2


class TestStateIdAssignment:
    """Test automatic ID assignment and auto-naming."""

    def test_sequential_ids(self):
        """States get sequential IDs starting from 0."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = State(num_values=4, name="C")

            assert A.id == 0
            assert B.id == 1
            assert C.id == 2

    def test_composite_states_get_ids(self):
        """Composite states also get IDs."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])

            assert A.id == 0
            assert B.id == 1
            assert C.id == 2

    def test_auto_name_for_unnamed_states(self):
        """Unnamed states get auto-generated names like id_N."""
        with Manager() as mgr:
            A = State(num_values=2)  # No name
            B = State(num_values=3)  # No name
            C = State(num_values=4, name="C")  # Has name

            assert A.name == "State[0]"
            assert B.name == "State[1]"
            assert C.name == "C"  # Keeps original name

    def test_named_states_keep_name(self):
        """Named states keep their original name."""
        with Manager() as mgr:
            A = State(num_values=2, name="MyState")

            assert A.id == 0
            assert A.name == "MyState"

    def test_id_property_readonly(self):
        """ID property is read-only."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            assert A.id == 0
            # Cannot set id via property (no setter)
            with pytest.raises(AttributeError):
                A.id = 5

    def test_separate_managers_have_independent_ids(self):
        """Different Manager instances have independent ID sequences."""
        with Manager() as mgr1:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            assert A.id == 0
            assert B.id == 1

        with Manager() as mgr2:
            C = State(num_values=4, name="C")
            D = State(num_values=5, name="D")
            assert C.id == 0  # Starts fresh
            assert D.id == 1

    def test_nested_managers_restore_previous(self):
        """Nested Manager contexts restore previous active manager on exit."""
        previous = Manager._active_manager
        with Manager() as outer:
            assert Manager._active_manager is outer
            A = State(num_values=2, name="A")

            with Manager() as inner:
                assert Manager._active_manager is inner
                B = State(num_values=3, name="B")
                assert B._manager is inner

            assert Manager._active_manager is outer
            C = State(num_values=4, name="C")
            assert C._manager is outer

        assert Manager._active_manager is previous

    def test_triple_nested_managers(self):
        """Three levels of nested Manager contexts restore correctly."""
        previous = Manager._active_manager
        with Manager() as m1:
            with Manager() as m2:
                with Manager() as m3:
                    assert Manager._active_manager is m3
                assert Manager._active_manager is m2
            assert Manager._active_manager is m1
        assert Manager._active_manager is previous


class TestDAGSupport:
    """Test that computation graphs support DAGs (same state reachable via multiple paths)."""

    def test_same_state_in_stack_allowed(self):
        """stack([A,A]) is allowed - creates a DAG with A reachable twice."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = stack([A, A])  # Should not raise - DAGs are allowed, sum preserves duplicates
            assert B.num_values == 4  # Sum of 2 + 2

    def test_same_state_in_product_raises_error(self):
        """product([A, A]) raises ValueError - duplicate states not allowed."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            with pytest.raises(ValueError, match="product\\(\\) does not allow duplicate states"):
                B = product([A, A])

    def test_same_state_nested_allowed(self):
        """C = stack([A,B]) where B = product([A, X]) is allowed - A reachable via two paths."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            X = State(num_values=3, name="X")
            B = product([A, X])
            B.name = "B"
            C = stack([A, B])  # A and B are different objects, both kept
            assert C.num_values == 2 + 6  # A has 2, B has 2*3=6

    def test_distinct_states_work(self):
        """C = stack([A,B]) with distinct A, B works fine."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = stack([A, B])  # Should not raise
            assert C.num_values == 5

    def test_shared_state_across_pools_allowed(self):
        """Multiple pools sharing the same state can be combined."""
        with Manager() as mgr:
            shared = State(num_values=3, name="shared")
            # Simulate two pools sharing the same state - sum preserves duplicates
            sum_state = stack([shared, shared])
            assert sum_state.num_values == 6  # 3 + 3
