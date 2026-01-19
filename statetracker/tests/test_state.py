"""Tests for State class."""
import pytest
from statetracker import (
    State, Manager, sync, product, stack,
    ConflictingValueAssignmentError, passthrough
)


class TestStateCreation:
    """Test State creation methods."""
    
    def test_leaf_state(self):
        with Manager():
            A = State(num_values=5, name='A')
            assert A.num_values == 5
            assert A.name == 'A'
            assert A.value == 0
    
    def test_leaf_state_without_name(self):
        with Manager():
            A = State(num_values=3)
            assert A.num_values == 3
            # Auto-named when registered
            assert A.name == 'State[0]'
    
    def test_leaf_state_requires_num_values(self):
        with Manager():
            with pytest.raises(ValueError, match="require num_values"):
                State()
    
    def test_name_setter(self):
        with Manager():
            A = State(num_values=2)
            A.name = 'MyState'
            assert A.name == 'MyState'
    
    def test_named_method(self):
        """named() sets name and returns self for chaining."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = stack([A,B]).named('C')
            
            assert C.name == 'C'
            assert C.num_values == 5  # Sum of A and B
            # Verify it's the actual state, not a copy
            C.value = 3
            assert B.value == 1  # B is active at state 1
    
    def test_state_requires_manager(self):
        """State raises error when created outside Manager."""
        with pytest.raises(RuntimeError, match="must be created within a Manager context"):
            State(num_values=3, name='A')


class TestStateManagement:
    """Test state management methods."""
    
    def test_advance(self):
        with Manager():
            A = State(num_values=3, name='A')
            assert A.value == 0
            A.advance()
            assert A.value == 1
            A.advance()
            assert A.value == 2
            A.advance()
            assert A.value == 0  # wraps around
    
    def test_reset(self):
        with Manager():
            A = State(num_values=5, name='A')
            A.value = 3
            A.reset()
            assert A.value == 0
    
    def test_iteration(self):
        with Manager():
            A = State(num_values=4, name='A')
            states = list(A)
            assert states == [0, 1, 2, 3]
    
    def test_iteration_resets_state(self):
        """Iteration should reset state before and after."""
        with Manager():
            A = State(num_values=3, name='A')
            A.value = 2
            list(A)  # iterate
            assert A.value == 0


class TestComplexGraphs:
    """Test more complex compositions."""
    
    def test_chained_products(self):
        """Test A * B * C (left-to-right association)."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=2, name='C')
            
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
        """Test combining product and sum operations."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            D = State(num_values=2, name='D')
            E = State(num_values=4, name='E')
            F = stack([D,E])
            F.name = 'F'
            
            # Both C and F have num_values=6
            assert C.num_values == 6
            assert F.num_values == 6
            
            # Sync them together
            G = sync([C, F], name='G')
            
            # Setting G should propagate to both branches
            G.value = 3
            assert C.value == 3
            assert F.value == 3
            assert A.value == 1  # 3 % 2
            assert B.value == 1  # 3 // 2


class TestRepr:
    """Test string representations."""
    
    def test_leaf_repr(self):
        with Manager():
            A = State(num_values=3, name='A')
            assert "name='A'" in repr(A)
            assert "num_values=3" in repr(A)
            assert "value=0" in repr(A)
    
    def test_composite_repr(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            assert "ProductOp" in repr(C)
            assert "num_values=6" in repr(C)


class TestInactiveState:
    """Test inactive state (None) behavior."""
    
    def test_stacked_sets_inactive_branch(self):
        """Stack decomposition sets inactive branch to None."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = stack([A,B])
            
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
            A = State(num_values=3, name='A')
            A.value = None
            with pytest.raises(RuntimeError, match="Cannot advance an inactive state"):
                A.advance()
    
    def test_reset_on_inactive_sets_zero(self):
        """Reset on inactive state sets to 0."""
        with Manager():
            A = State(num_values=3, name='A')
            A.value = None
            assert A.value is None
            A.reset()
            assert A.value == 0
    
    def test_reset_with_value_argument(self):
        """Reset with value argument sets to that value."""
        with Manager():
            A = State(num_values=5, name='A')
            A.reset(value=3)
            assert A.value == 3
            
            # Also works from inactive
            A.value = None
            A.reset(value=2)
            assert A.value == 2
    
    def test_manual_set_inactive(self):
        """Can manually set state to inactive."""
        with Manager():
            A = State(num_values=3, name='A')
            A.value = None
            assert A.value is None
    
    def test_inactive_method(self):
        """inactive() method sets state to None."""
        with Manager():
            A = State(num_values=3, name='A')
            A.value = 2
            A.value = None
            assert A.value is None
    
    def test_is_active_true(self):
        """is_active() returns True for active states."""
        with Manager():
            A = State(num_values=3, name='A')
            assert A.is_active() is True
            A.value = 2
            assert A.is_active() is True
    
    def test_is_active_false(self):
        """is_active() returns False for inactive states."""
        with Manager():
            A = State(num_values=3, name='A')
            A.value = None
            assert A.is_active() is False
    
    def test_is_active_after_inactive_method(self):
        """is_active() returns False after calling inactive()."""
        with Manager():
            A = State(num_values=3, name='A')
            assert A.is_active() is True
            A.value = None
            assert A.is_active() is False
    
    def test_stack_iteration_with_inactive(self):
        """Iterate sum shows None for inactive branch."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=2, name='B')
            C = stack([A,B])
            
            results = []
            for _ in C:
                results.append((A.value, B.value, A.is_active(), B.is_active()))
            
            expected = [
                (0, None, True, False),   # A active
                (1, None, True, False),   # A active
                (None, 0, False, True),   # B active
                (None, 1, False, True),   # B active
            ]
            assert results == expected


class TestCopy:
    """Test State.copy() method."""
    
    def test_copy_leaf_state(self):
        """Copy a leaf state creates independent object."""
        with Manager():
            A = State(num_values=5, name='A')
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
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
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
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            
            D = C.copy()
            
            # Set D's state - this propagates to shared parents A and B
            D.value = 5
            assert A.value == 1  # Shared parent affected
            assert B.value == 2  # Shared parent affected
    
    def test_copy_preserves_state(self):
        """Copy preserves current state."""
        with Manager():
            A = State(num_values=5, name='A')
            A.value = 2
            B = A.copy()
            assert B.value == 2
    
    def test_copy_preserves_inactive_state(self):
        """Copy preserves inactive state (None)."""
        with Manager():
            A = State(num_values=5, name='A')
            A.value = None
            B = A.copy()
            assert B.value is None
            assert not B.is_active()
    
    def test_copy_with_name(self):
        """Copy with name parameter sets the new name."""
        with Manager():
            A = State(num_values=3, name='A')
            B = A.copy(name='B_copy')
            assert B.name == 'B_copy'
    
    def test_copy_without_name_gets_auto_name_in_manager(self):
        """Copy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = State(num_values=3, name='A')
            B = A.copy()
            # B should have an auto-generated name like 'State[1]'
            assert B.name == 'State[1]'
    
    def test_copy_registers_with_manager(self):
        """Copied state registers with active manager."""
        with Manager() as mgr:
            A = State(num_values=3, name='A')
            B = A.copy(name='B')
            
            assert len(mgr._states) == 2
            assert B in mgr._states


class TestDeepcopy:
    """Test State.deepcopy() method."""
    
    def test_deepcopy_leaf_state(self):
        """Deepcopy a leaf state creates independent object."""
        with Manager():
            A = State(num_values=5, name='A')
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
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
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
        """Iterating deepcopy does NOT affect original ancestors."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            C.value = 0  # A=0, B=0
            
            D = C.deepcopy()
            
            # Set D's state - this should NOT affect original A and B
            D.value = 5
            
            # Original A and B should still be at their original state (0)
            # Note: C.value=0 propagated to A=0, B=0 initially
            assert A.value == 0
            assert B.value == 0
            
            # D's copied parents should have the new value
            assert D._parents[0].value == 1  # 5 % 2 = 1
            assert D._parents[1].value == 2  # 5 // 2 = 2
    
    def test_deepcopy_preserves_state(self):
        """Deepcopy preserves current state."""
        with Manager():
            A = State(num_values=5, name='A')
            A.value = 2
            B = A.deepcopy()
            assert B.value == 2
    
    def test_deepcopy_preserves_inactive_state(self):
        """Deepcopy preserves inactive state (None)."""
        with Manager():
            A = State(num_values=5, name='A')
            A.value = None
            B = A.deepcopy()
            assert B.value is None
            assert not B.is_active()
    
    def test_deepcopy_with_name(self):
        """Deepcopy with name parameter sets the new name."""
        with Manager():
            A = State(num_values=3, name='A')
            B = A.deepcopy(name='B_deepcopy')
            assert B.name == 'B_deepcopy'
    
    def test_deepcopy_without_name_gets_auto_name_in_manager(self):
        """Deepcopy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = State(num_values=3, name='A')
            B = A.deepcopy()
            # B should have an auto-generated name like 'State[1]'
            assert B.name == 'State[1]'
    
    def test_deepcopy_registers_with_manager(self):
        """Deepcopied state registers with active manager."""
        with Manager() as mgr:
            A = State(num_values=3, name='A')
            B = A.deepcopy(name='B')
            
            assert len(mgr._states) == 2
            assert B in mgr._states
    
    def test_deepcopy_all_ancestors_register(self):
        """All copied ancestors register with active manager."""
        with Manager() as mgr:
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            # 3 states registered so far
            assert len(mgr._states) == 3
            
            D = C.deepcopy(name='D')
            
            # D plus its 2 copied parents = 3 more states
            assert len(mgr._states) == 6
    
    def test_deepcopy_nested_structure(self):
        """Deepcopy works for deeply nested structures."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=2, name='B')
            C = product([A, B])
            C.name = 'C'
            
            D = State(num_values=2, name='D')
            E = stack([C,D])
            E.name = 'E'
            
            # E has structure: E = stack([(A * B),D])
            E.value = 2  # In D's branch (D=0), A and B are inactive
            
            F = E.deepcopy(name='F')
            
            # F should be independent
            assert F is not E
            assert F.num_values == E.num_values
            assert F.value == 2
            
            # Changing F doesn't affect E
            F.value = 0
            assert E.value == 2


class TestConflictDetection:
    """Test conflict detection during state propagation."""
    
    def test_sync_with_reversed_slice_raises_error(self):
        """Syncing A with A[::-1] should raise conflict error."""
        with Manager():
            A = State(num_values=5, name='A')
            B = A[::-1]
            B.name = 'B'
            C = sync([A, B], name='C')
            
            with pytest.raises(ConflictingValueAssignmentError):
                C.value = 0
    
    def test_sync_with_same_state_no_conflict(self):
        """Syncing A with itself (via passthrough) should NOT raise error."""
        with Manager():
            A = State(num_values=5, name='A')
            B = passthrough(A, name='B')
            C = sync([A, B], name='C')
            
            # Should not raise - A gets same value from both paths
            C.value = 3
            assert A.value == 3
    
    def test_product_no_conflict(self):
        """Product states should not trigger false positives."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=4, name='B')
            C = product([A, B])
            
            # Should work fine - iterate through all states
            for state in C:
                pass
    
    def test_conflict_on_advance(self):
        """Conflict should be detected during advance() too."""
        with Manager():
            A = State(num_values=5, name='A')
            B = A[::-1]
            B.name = 'B'
            C = sync([A, B], name='C')
            
            # With A defaulting to 0, C.reset() would cause immediate conflict
            # So we manually set C to state 0 (which will raise conflict)
            # But to test advance(), we need to set up a non-conflicting state first
            # Since sync([A, A[::-1]]) always conflicts, we test that reset() raises
            # and then test advance() on a state that's already active
            with pytest.raises(ConflictingValueAssignmentError):
                C.reset()  # reset() causes conflict because A defaults to 0
            
            # For advance() test, manually set C to a state that would work
            # if A were at the right state, then advance should cause conflict
            # Actually, any state assignment to C will conflict, so we test that
            # advance() on an already-active state also detects conflicts
            # But C can't be active without conflict, so we test a different scenario:
            # Set A to a specific state, then try to advance C
            A.value = 2  # Set A to state 2
            C.value = 2  # This should work: A=2, B=2 means A[::-1]=2 means A=2 (no conflict)
            # Now advance C: C.value = 3 means A=3, B=3 means A[::-1]=3 means A=1 (conflict!)
            with pytest.raises(ConflictingValueAssignmentError):
                C.advance()  # advance() calls state setter, detects conflict
