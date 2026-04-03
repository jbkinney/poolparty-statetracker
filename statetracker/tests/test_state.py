"""Tests for State class."""

import pytest

from statetracker import Manager, State, product, stack, sync


class TestStateCreation:
    """Test State creation methods."""

    def test_leaf_state(self):
        with Manager():
            A = State(num_values=5, name="A")
            assert A.num_values == 5
            assert A.name == "A"
            assert A.value is None  # States default to None (inactive)

    def test_leaf_state_without_name(self):
        with Manager():
            A = State(num_values=3)
            assert A.num_values == 3
            # Auto-named when registered
            assert A.name == "State[0]"

    def test_leaf_state_with_none_num_values_defaults_to_one(self):
        """Creating State with num_values=None defaults to num_values=1."""
        with Manager():
            A = State()  # No num_values defaults to 1
            assert A.num_values == 1

    def test_name_setter(self):
        with Manager():
            A = State(num_values=2)
            A.name = "MyState"
            assert A.name == "MyState"

    def test_named_method(self):
        """named() sets name and returns self for chaining."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = stack([A, B]).named("C")

            assert C.name == "C"
            assert C.num_values == 5  # Sum of A and B
            # Verify it's the actual state, not a copy
            C.value = 3
            assert B.value == 1  # B is active at state 1

    def test_state_requires_manager(self):
        """State raises error when created outside any Manager context."""
        previous = Manager._active_manager
        Manager._active_manager = None
        try:
            with pytest.raises(RuntimeError, match="must be created within a Manager context"):
                State(num_values=3, name="A")
        finally:
            Manager._active_manager = previous


class TestStateManagement:
    """Test state management methods."""

    def test_advance(self):
        with Manager():
            A = State(num_values=3, name="A")
            A.reset()  # Activate state at value 0
            assert A.value == 0
            A.advance()
            assert A.value == 1
            A.advance()
            assert A.value == 2
            A.advance()
            assert A.value == 0  # wraps around

    def test_reset(self):
        with Manager():
            A = State(num_values=5, name="A")
            A.value = 3
            A.reset()
            assert A.value == 0

    def test_iteration(self):
        with Manager():
            A = State(num_values=4, name="A")
            states = list(A)
            assert states == [0, 1, 2, 3]

    def test_iteration_resets_state(self):
        """Iteration should reset state before and after."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = 2
            list(A)  # iterate
            assert A.value == 0


class TestComplexGraphs:
    """Test more complex compositions."""

    def test_chained_products(self):
        """Test A * B * C (left-to-right association)."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = State(num_values=2, name="C")

            AB = product([A, B])
            ABC = product([AB, C])

            assert ABC.num_values == 12

            # Parents in ABC are in order as provided: (AB, C)
            # AB cycles fastest, C cycles slowest
            # For ABC.value = 7:
            #   AB = 7 % 6 = 1
            #   C = (7 // 6) % 2 = 1 % 2 = 1
            # For AB.value = 1:
            #   A = 1 % 2 = 1
            #   B = 1 // 2 = 0
            ABC.value = 7
            assert AB.value == 1
            assert C.value == 1
            assert A.value == 1
            assert B.value == 0

    def test_product_and_stack_combined(self):
        """Test combining product and stack operations."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            D = State(num_values=2, name="D")
            E = State(num_values=4, name="E")
            F = stack([D, E])
            F.name = "F"

            # Both C and F have num_values=6
            assert C.num_values == 6
            assert F.num_values == 6

            # Test that values propagate correctly
            C.value = 3
            assert A.value == 1  # 3 % 2
            assert B.value == 1  # 3 // 2


class TestRepr:
    """Test string representations."""

    def test_leaf_repr(self):
        with Manager():
            A = State(num_values=3, name="A")
            assert "name='A'" in repr(A)
            assert "num_values=3" in repr(A)
            assert "value=None" in repr(A)  # States default to None

    def test_composite_repr(self):
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            assert "ProductOp" in repr(C)
            assert "num_values=6" in repr(C)


class TestInactiveState:
    """Test inactive state (None) behavior."""

    def test_stacked_sets_inactive_branch(self):
        """Stack decomposition sets inactive branch to None."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = stack([A, B])

            # A branch active -> B inactive
            C.value = 0
            assert A.value == 0
            assert B.value is None

            # B branch active -> A inactive
            C.value = 3
            assert A.value is None
            assert B.value == 1

    def test_advance_on_inactive_raises(self):
        """Cannot advance an inactive state."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = None
            with pytest.raises(RuntimeError, match="Cannot advance an inactive state"):
                A.advance()

    def test_reset_on_inactive_sets_zero(self):
        """Reset on inactive state sets to 0."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = None
            assert A.value is None
            A.reset()
            assert A.value == 0

    def test_reset_with_value_argument(self):
        """Reset with value argument sets to that value."""
        with Manager():
            A = State(num_values=5, name="A")
            A.reset(value=3)
            assert A.value == 3

            # Also works from inactive
            A.value = None
            A.reset(value=2)
            assert A.value == 2

    def test_manual_set_inactive(self):
        """Can manually set state to inactive."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = None
            assert A.value is None

    def test_inactive_method(self):
        """inactive() method sets state to None."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = 2
            A.value = None
            assert A.value is None

    def test_is_active_true(self):
        """is_active returns True for active states."""
        with Manager():
            A = State(num_values=3, name="A")
            assert A.is_active is False  # States default to inactive
            A.reset()  # Activate at value 0
            assert A.is_active is True
            A.value = 2
            assert A.is_active is True

    def test_is_active_false(self):
        """is_active returns False for inactive states."""
        with Manager():
            A = State(num_values=3, name="A")
            A.value = None
            assert A.is_active is False

    def test_is_active_after_inactive_method(self):
        """is_active returns False after setting to None."""
        with Manager():
            A = State(num_values=3, name="A")
            A.reset()  # Activate first
            assert A.is_active is True
            A.value = None
            assert A.is_active is False

    def test_stack_iteration_with_inactive(self):
        """Iterate sum shows None for inactive branch."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=2, name="B")
            C = stack([A, B])

            results = []
            for _ in C:
                results.append((A.value, B.value, A.is_active, B.is_active))

            expected = [
                (0, None, True, False),  # A active
                (1, None, True, False),  # A active
                (None, 0, False, True),  # B active
                (None, 1, False, True),  # B active
            ]
            assert results == expected


class TestCopy:
    """Test State.copy() method."""

    def test_copy_leaf_state(self):
        """Copy a leaf state creates independent object."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = 3
            B = A.copy()

            # B is a different object
            assert B is not A
            assert B.num_values == 5
            assert B.value == 3

            # Modifying B doesn't affect A
            B.value = 1
            assert A.value == 3

    def test_copy_composite_state_shares_parents(self):
        """Copy of composite state shares parent references."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.value = 4

            D = C.copy()

            # D is a different object
            assert D is not C
            assert D.num_values == 6
            assert D.value == 4

            # D shares the same parent state objects (sorted by id)
            assert set(D._parents) == set(C._parents)
            assert A in D._parents
            assert B in D._parents

    def test_copy_shares_parents_iterating_affects_shared(self):
        """Iterating copy affects shared parent states."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])

            D = C.copy()

            # Set D's state - this propagates to shared parents A and B
            D.value = 5
            assert A.value == 1  # Shared parent affected
            assert B.value == 2  # Shared parent affected

    def test_copy_preserves_state(self):
        """Copy preserves current state."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = 2
            B = A.copy()
            assert B.value == 2

    def test_copy_preserves_inactive_state(self):
        """Copy preserves inactive state (None)."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = None
            B = A.copy()
            assert B.value is None
            assert not B.is_active

    def test_copy_with_name(self):
        """Copy with name parameter sets the new name."""
        with Manager():
            A = State(num_values=3, name="A")
            B = A.copy(name="B_copy")
            assert B.name == "B_copy"

    def test_copy_without_name_gets_auto_name_in_manager(self):
        """Copy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = A.copy()
            # B should have an auto-generated name like 'State[1]'
            assert B.name == "State[1]"

    def test_copy_registers_with_manager(self):
        """Copied state registers with active manager."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = A.copy(name="B")

            assert len(mgr._states) == 2
            assert B in mgr._states


class TestDeepcopy:
    """Test State.deepcopy() method."""

    def test_deepcopy_leaf_state(self):
        """Deepcopy a leaf state creates independent object."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = 3
            B = A.deepcopy()

            # B is a different object
            assert B is not A
            assert B.num_values == 5
            assert B.value == 3

            # Modifying B doesn't affect A
            B.value = 1
            assert A.value == 3

    def test_deepcopy_composite_state_copies_parents(self):
        """Deepcopy of composite state copies all ancestors."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.value = 4

            D = C.deepcopy()

            # D is a different object
            assert D is not C
            assert D.num_values == 6
            assert D.value == 4

            # D has different parents (copied)
            assert D._parents is not C._parents
            assert D._parents[0] is not A
            assert D._parents[1] is not B

    def test_deepcopy_is_fully_independent(self):
        """Deepcopy creates independent graph, but setting values clears all states."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.value = 0  # A=0, B=0

            D = C.deepcopy()

            # Set D's state - this clears ALL states (including A and B)
            D.value = 5

            # Original A and B are cleared by clear_all_values()
            assert A.value is None
            assert B.value is None

            # D's copied parents have the propagated values
            assert D._parents[0].value == 1  # 5 % 2 = 1
            assert D._parents[1].value == 2  # 5 // 2 = 2

    def test_deepcopy_preserves_state(self):
        """Deepcopy preserves current state."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = 2
            B = A.deepcopy()
            assert B.value == 2

    def test_deepcopy_preserves_inactive_state(self):
        """Deepcopy preserves inactive state (None)."""
        with Manager():
            A = State(num_values=5, name="A")
            A.value = None
            B = A.deepcopy()
            assert B.value is None
            assert not B.is_active

    def test_deepcopy_with_name(self):
        """Deepcopy with name parameter sets the new name."""
        with Manager():
            A = State(num_values=3, name="A")
            B = A.deepcopy(name="B_deepcopy")
            assert B.name == "B_deepcopy"

    def test_deepcopy_without_name_gets_auto_name_in_manager(self):
        """Deepcopy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = A.deepcopy()
            # B should have an auto-generated name like 'State[1]'
            assert B.name == "State[1]"

    def test_deepcopy_registers_with_manager(self):
        """Deepcopied state registers with active manager."""
        with Manager() as mgr:
            A = State(num_values=3, name="A")
            B = A.deepcopy(name="B")

            assert len(mgr._states) == 2
            assert B in mgr._states

    def test_deepcopy_all_ancestors_register(self):
        """All copied ancestors register with active manager."""
        with Manager() as mgr:
            A = State(num_values=2, name="A")
            B = State(num_values=3, name="B")
            C = product([A, B])
            C.name = "C"

            # 3 states registered so far
            assert len(mgr._states) == 3

            D = C.deepcopy(name="D")

            # D plus its 2 copied parents = 3 more states
            assert len(mgr._states) == 6

    def test_deepcopy_nested_structure(self):
        """Deepcopy works for deeply nested structures."""
        with Manager():
            A = State(num_values=2, name="A")
            B = State(num_values=2, name="B")
            C = product([A, B])
            C.name = "C"

            D = State(num_values=2, name="D")
            E = stack([C, D])
            E.name = "E"

            # E has structure: E = stack([(A * B),D])
            E.value = 2  # In D's branch (D=0), A and B are inactive

            F = E.deepcopy(name="F")

            # F should be independent object
            assert F is not E
            assert F.num_values == E.num_values
            assert F.value == 2

            # Changing F clears all states (including E)
            F.value = 0
            assert E.value is None  # Cleared by clear_all_values()


class TestConflictDetection:
    """Test conflict detection during state propagation."""

    def test_product_no_conflict(self):
        """Product states should not trigger false positives."""
        with Manager():
            A = State(num_values=3, name="A")
            B = State(num_values=4, name="B")
            C = product([A, B])

            # Should work fine - iterate through all states
            for state in C:
                pass


class TestFlexibleSync:
    """Test syncing states with different num_values."""

    def test_sync_different_num_values_basic(self):
        """Sync states with different num_values - basic case."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)

            # Group should have max num_values
            assert A._synced_group.num_values == 5
            assert A._synced_group is B._synced_group

            # Value in range for both
            A.value = 2
            assert A.value == 2
            assert B.value == 2

            # Value out of range for B
            A.value = 4
            assert A.value == 4
            assert B.value is None  # Out of range for B

    def test_sync_different_num_values_iteration(self):
        """Iteration uses state's own num_values."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)

            # Iterating A goes through A's 5 values
            a_values = list(A)
            assert a_values == [0, 1, 2, 3, 4]

            # Iterating B goes through only B's 3 values
            b_values = list(B)
            assert b_values == [0, 1, 2]

    def test_sync_different_num_values_bidirectional(self):
        """Setting value on smaller state syncs to larger."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)

            # Set via B (smaller)
            B.value = 2
            assert A.value == 2
            assert B.value == 2

    def test_sync_different_num_values_group_value(self):
        """Group tracks logical value even when some states are None."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)

            A.value = 4
            assert A._synced_group.value == 4
            assert A.value == 4
            assert B.value is None

    def test_sync_three_states_different_sizes(self):
        """Sync three states with different num_values."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)
            C = synced_to(A, name="C", num_values=7)

            # Group should have max = 7
            assert A._synced_group.num_values == 7

            # Value=2 in range for all
            A.value = 2
            assert A.value == 2
            assert B.value == 2
            assert C.value == 2

            # Value=4 out of range for B
            A.value = 4
            assert A.value == 4
            assert B.value is None
            assert C.value == 4

            # Value=6 out of range for A and B
            A.value = 6
            assert A.value is None
            assert B.value is None
            assert C.value == 6

    def test_sync_with_product_propagation(self):
        """Synced states with product - propagation respects range."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B", num_values=3)
            C = State(num_values=4, name="C")
            D = product([A, C]).named("D")

            # D.value = 5 means A=0, C=1 (5 = 0 + 5*1, but actually 5 % 5 = 0, 5 // 5 = 1)
            # Actually: D has 5*4=20 values
            # value=5: A = 5 % 5 = 0, C = 5 // 5 = 1
            D.value = 5
            assert A.value == 0
            assert B.value == 0  # In range
            assert C.value == 1

            # value=8: A = 8 % 5 = 3, C = 8 // 5 = 1
            D.value = 8
            assert A.value == 3
            assert B.value is None  # 3 >= 3, out of range
            assert C.value == 1

    def test_sync_advance_uses_state_num_values(self):
        """advance() wraps around using the state's own num_values."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=3, name="A")
            B = synced_to(A, name="B", num_values=5)

            # Set to value 2 (valid for both A and B)
            A.value = 2
            assert A.value == 2
            assert B.value == 2

            # A.advance() wraps using A's num_values (3): 2 -> 0
            A.advance()
            assert A._synced_group.value == 0
            assert A.value == 0
            assert B.value == 0

            # B.advance() wraps using B's num_values (5): 0 -> 1
            B.value = 4  # A is None (out of range), B is 4
            assert A.value is None
            assert B.value == 4
            B.advance()  # 4 -> 0 (wraps at B's num_values=5)
            assert B.value == 0
            assert A.value == 0

    def test_sync_merge_different_sizes(self):
        """Merging groups with different num_values takes max."""
        with Manager():
            A = State(num_values=5, name="A")
            B = State(num_values=3, name="B")

            # Initially separate groups
            assert A._synced_group.num_values == 5
            assert B._synced_group.num_values == 3

            # Sync them
            sync(A, B)

            # Merged group has max
            assert A._synced_group.num_values == 5
            assert A._synced_group is B._synced_group

            # Value out of range for B
            A.value = 4
            assert A.value == 4
            assert B.value is None

    def test_synced_to_without_num_values_uses_default(self):
        """synced_to without num_values parameter uses child's num_values."""
        from statetracker import synced_to

        with Manager():
            A = State(num_values=5, name="A")
            B = synced_to(A, name="B")  # No num_values specified

            assert B.num_values == 5
            assert A._synced_group.num_values == 5
