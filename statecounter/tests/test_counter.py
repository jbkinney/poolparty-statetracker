"""Tests for Counter class."""
import pytest
from statecounter import (
    Counter, Manager, sync, product, stack,
    ConflictingStateAssignmentError, passthrough
)


class TestCounterCreation:
    """Test Counter creation methods."""
    
    def test_leaf_counter(self):
        with Manager():
            A = Counter(num_states=5, name='A')
            assert A.num_states == 5
            assert A.name == 'A'
            assert A.state == 0
    
    def test_leaf_counter_without_name(self):
        with Manager():
            A = Counter(num_states=3)
            assert A.num_states == 3
            # Auto-named when registered
            assert A.name == 'Counter[0]'
    
    def test_leaf_counter_requires_num_states(self):
        with Manager():
            with pytest.raises(ValueError, match="require num_states"):
                Counter()
    
    def test_name_setter(self):
        with Manager():
            A = Counter(num_states=2)
            A.name = 'MyCounter'
            assert A.name == 'MyCounter'
    
    def test_named_method(self):
        """named() sets name and returns self for chaining."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = stack([A,B]).named('C')
            
            assert C.name == 'C'
            assert C.num_states == 5  # Sum of A and B
            # Verify it's the actual counter, not a copy
            C.state = 3
            assert B.state == 1  # B is active at state 1
    
    def test_counter_requires_manager(self):
        """Counter raises error when created outside Manager."""
        with pytest.raises(RuntimeError, match="must be created within a Manager context"):
            Counter(num_states=3, name='A')


class TestStateManagement:
    """Test state management methods."""
    
    def test_advance(self):
        with Manager():
            A = Counter(num_states=3, name='A')
            assert A.state == 0
            A.advance()
            assert A.state == 1
            A.advance()
            assert A.state == 2
            A.advance()
            assert A.state == 0  # wraps around
    
    def test_reset(self):
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = 3
            A.reset()
            assert A.state == 0
    
    def test_iteration(self):
        with Manager():
            A = Counter(num_states=4, name='A')
            states = list(A)
            assert states == [0, 1, 2, 3]
    
    def test_iteration_resets_state(self):
        """Iteration should reset state before and after."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = 2
            list(A)  # iterate
            assert A.state == 0


class TestComplexGraphs:
    """Test more complex compositions."""
    
    def test_chained_products(self):
        """Test A * B * C (left-to-right association)."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=2, name='C')
            
            AB = product([A, B])
            ABC = product([AB, C])
            
            assert ABC.num_states == 12
            
            # Parents in ABC are in order as provided: (AB, C)
            # AB cycles fastest, C cycles slowest
            # For ABC.state = 7:
            #   AB = 7 % 6 = 1
            #   C = (7 // 6) % 2 = 1 % 2 = 1
            # For AB.state = 1:
            #   A = 1 % 2 = 1
            #   B = 1 // 2 = 0
            ABC.state = 7
            assert AB.state == 1
            assert C.state == 1
            assert A.state == 1
            assert B.state == 0
    
    def test_product_and_stack_combined(self):
        """Test combining product and sum operations."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            D = Counter(num_states=2, name='D')
            E = Counter(num_states=4, name='E')
            F = stack([D,E])
            F.name = 'F'
            
            # Both C and F have num_states=6
            assert C.num_states == 6
            assert F.num_states == 6
            
            # Sync them together
            G = sync([C, F], name='G')
            
            # Setting G should propagate to both branches
            G.state = 3
            assert C.state == 3
            assert F.state == 3
            assert A.state == 1  # 3 % 2
            assert B.state == 1  # 3 // 2


class TestRepr:
    """Test string representations."""
    
    def test_leaf_repr(self):
        with Manager():
            A = Counter(num_states=3, name='A')
            assert "name='A'" in repr(A)
            assert "num_states=3" in repr(A)
            assert "state=0" in repr(A)
    
    def test_composite_repr(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            assert "ProductOp" in repr(C)
            assert "num_states=6" in repr(C)


class TestInactiveState:
    """Test inactive state (None) behavior."""
    
    def test_stacked_sets_inactive_branch(self):
        """Stack decomposition sets inactive branch to None."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = stack([A,B])
            
            # A branch active -> B inactive
            C.state = 0
            assert A.state == 0
            assert B.state is None
            
            # B branch active -> A inactive
            C.state = 3
            assert A.state is None
            assert B.state == 1
    
    def test_advance_on_inactive_raises(self):
        """Cannot advance an inactive counter."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = None
            with pytest.raises(RuntimeError, match="Cannot advance an inactive counter"):
                A.advance()
    
    def test_reset_on_inactive_sets_zero(self):
        """Reset on inactive counter sets to 0."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = None
            assert A.state is None
            A.reset()
            assert A.state == 0
    
    def test_reset_with_state_argument(self):
        """Reset with state argument sets to that state."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.reset(state=3)
            assert A.state == 3
            
            # Also works from inactive
            A.state = None
            A.reset(state=2)
            assert A.state == 2
    
    def test_manual_set_inactive(self):
        """Can manually set counter to inactive."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = None
            assert A.state is None
    
    def test_inactive_method(self):
        """inactive() method sets state to None."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = 2
            A.state = None
            assert A.state is None
    
    def test_is_active_true(self):
        """is_active() returns True for active counters."""
        with Manager():
            A = Counter(num_states=3, name='A')
            assert A.is_active() is True
            A.state = 2
            assert A.is_active() is True
    
    def test_is_active_false(self):
        """is_active() returns False for inactive counters."""
        with Manager():
            A = Counter(num_states=3, name='A')
            A.state = None
            assert A.is_active() is False
    
    def test_is_active_after_inactive_method(self):
        """is_active() returns False after calling inactive()."""
        with Manager():
            A = Counter(num_states=3, name='A')
            assert A.is_active() is True
            A.state = None
            assert A.is_active() is False
    
    def test_stack_iteration_with_inactive(self):
        """Iterate sum shows None for inactive branch."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=2, name='B')
            C = stack([A,B])
            
            results = []
            for _ in C:
                results.append((A.state, B.state, A.is_active(), B.is_active()))
            
            expected = [
                (0, None, True, False),   # A active
                (1, None, True, False),   # A active
                (None, 0, False, True),   # B active
                (None, 1, False, True),   # B active
            ]
            assert results == expected


class TestCopy:
    """Test Counter.copy() method."""
    
    def test_copy_leaf_counter(self):
        """Copy a leaf counter creates independent object."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = 3
            B = A.copy()
            
            # B is a different object
            assert B is not A
            assert B.num_states == 5
            assert B.state == 3
            
            # Modifying B doesn't affect A
            B.state = 1
            assert A.state == 3
    
    def test_copy_composite_counter_shares_parents(self):
        """Copy of composite counter shares parent references."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.state = 4
            
            D = C.copy()
            
            # D is a different object
            assert D is not C
            assert D.num_states == 6
            assert D.state == 4
            
            # D shares the same parent counter objects (sorted by id)
            assert set(D._parents) == set(C._parents)
            assert A in D._parents
            assert B in D._parents
    
    def test_copy_shares_parents_iterating_affects_shared(self):
        """Iterating copy affects shared parent counters."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            
            D = C.copy()
            
            # Set D's state - this propagates to shared parents A and B
            D.state = 5
            assert A.state == 1  # Shared parent affected
            assert B.state == 2  # Shared parent affected
    
    def test_copy_preserves_state(self):
        """Copy preserves current state."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = 2
            B = A.copy()
            assert B.state == 2
    
    def test_copy_preserves_inactive_state(self):
        """Copy preserves inactive state (None)."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = None
            B = A.copy()
            assert B.state is None
            assert not B.is_active()
    
    def test_copy_with_name(self):
        """Copy with name parameter sets the new name."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = A.copy(name='B_copy')
            assert B.name == 'B_copy'
    
    def test_copy_without_name_gets_auto_name_in_manager(self):
        """Copy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = A.copy()
            # B should have an auto-generated name like 'Counter[1]'
            assert B.name == 'Counter[1]'
    
    def test_copy_registers_with_manager(self):
        """Copied counter registers with active manager."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = A.copy(name='B')
            
            assert len(mgr._counters) == 2
            assert B in mgr._counters


class TestDeepcopy:
    """Test Counter.deepcopy() method."""
    
    def test_deepcopy_leaf_counter(self):
        """Deepcopy a leaf counter creates independent object."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = 3
            B = A.deepcopy()
            
            # B is a different object
            assert B is not A
            assert B.num_states == 5
            assert B.state == 3
            
            # Modifying B doesn't affect A
            B.state = 1
            assert A.state == 3
    
    def test_deepcopy_composite_counter_copies_parents(self):
        """Deepcopy of composite counter copies all ancestors."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.state = 4
            
            D = C.deepcopy()
            
            # D is a different object
            assert D is not C
            assert D.num_states == 6
            assert D.state == 4
            
            # D has different parents (copied)
            assert D._parents is not C._parents
            assert D._parents[0] is not A
            assert D._parents[1] is not B
    
    def test_deepcopy_is_fully_independent(self):
        """Iterating deepcopy does NOT affect original ancestors."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.state = 0  # A=0, B=0
            
            D = C.deepcopy()
            
            # Set D's state - this should NOT affect original A and B
            D.state = 5
            
            # Original A and B should still be at their original state (0)
            # Note: C.state=0 propagated to A=0, B=0 initially
            assert A.state == 0
            assert B.state == 0
            
            # D's copied parents should have the new state
            assert D._parents[0].state == 1  # 5 % 2 = 1
            assert D._parents[1].state == 2  # 5 // 2 = 2
    
    def test_deepcopy_preserves_state(self):
        """Deepcopy preserves current state."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = 2
            B = A.deepcopy()
            assert B.state == 2
    
    def test_deepcopy_preserves_inactive_state(self):
        """Deepcopy preserves inactive state (None)."""
        with Manager():
            A = Counter(num_states=5, name='A')
            A.state = None
            B = A.deepcopy()
            assert B.state is None
            assert not B.is_active()
    
    def test_deepcopy_with_name(self):
        """Deepcopy with name parameter sets the new name."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = A.deepcopy(name='B_deepcopy')
            assert B.name == 'B_deepcopy'
    
    def test_deepcopy_without_name_gets_auto_name_in_manager(self):
        """Deepcopy without name gets auto-generated name in manager."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = A.deepcopy()
            # B should have an auto-generated name like 'Counter[1]'
            assert B.name == 'Counter[1]'
    
    def test_deepcopy_registers_with_manager(self):
        """Deepcopied counter registers with active manager."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = A.deepcopy(name='B')
            
            assert len(mgr._counters) == 2
            assert B in mgr._counters
    
    def test_deepcopy_all_ancestors_register(self):
        """All copied ancestors register with active manager."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            # 3 counters registered so far
            assert len(mgr._counters) == 3
            
            D = C.deepcopy(name='D')
            
            # D plus its 2 copied parents = 3 more counters
            assert len(mgr._counters) == 6
    
    def test_deepcopy_nested_structure(self):
        """Deepcopy works for deeply nested structures."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=2, name='B')
            C = product([A, B])
            C.name = 'C'
            
            D = Counter(num_states=2, name='D')
            E = stack([C,D])
            E.name = 'E'
            
            # E has structure: E = stack([(A * B),D])
            E.state = 2  # In D's branch (D=0), A and B are inactive
            
            F = E.deepcopy(name='F')
            
            # F should be independent
            assert F is not E
            assert F.num_states == E.num_states
            assert F.state == 2
            
            # Changing F doesn't affect E
            F.state = 0
            assert E.state == 2


class TestConflictDetection:
    """Test conflict detection during state propagation."""
    
    def test_sync_with_reversed_slice_raises_error(self):
        """Syncing A with A[::-1] should raise conflict error."""
        with Manager():
            A = Counter(num_states=5, name='A')
            B = A[::-1]
            B.name = 'B'
            C = sync([A, B], name='C')
            
            with pytest.raises(ConflictingStateAssignmentError):
                C.state = 0
    
    def test_sync_with_same_counter_no_conflict(self):
        """Syncing A with itself (via passthrough) should NOT raise error."""
        with Manager():
            A = Counter(num_states=5, name='A')
            B = passthrough(A, name='B')
            C = sync([A, B], name='C')
            
            # Should not raise - A gets same value from both paths
            C.state = 3
            assert A.state == 3
    
    def test_product_no_conflict(self):
        """Product counters should not trigger false positives."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=4, name='B')
            C = product([A, B])
            
            # Should work fine - iterate through all states
            for state in C:
                pass
    
    def test_conflict_on_advance(self):
        """Conflict should be detected during advance() too."""
        with Manager():
            A = Counter(num_states=5, name='A')
            B = A[::-1]
            B.name = 'B'
            C = sync([A, B], name='C')
            
            with pytest.raises(ConflictingStateAssignmentError):
                C.advance()  # advance() calls state setter
