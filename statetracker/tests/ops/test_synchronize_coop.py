"""Tests for SynchronizeOp and synchronize_states()."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, SyncOp, sync


class TestSyncOperation:
    """Test sync operation."""
    
    def test_sync_num_states(self):
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            S = sync([A, B])
            assert S.num_values == 3
    
    def test_sync_propagates_to_both(self):
        """Setting sync state propagates to both parents."""
        with Manager():
            A = State(num_values=4, name='A')
            B = State(num_values=4, name='B')
            S = sync([A, B])
            
            S.value = 2
            assert A.value == 2
            assert B.value == 2
    
    def test_sync_iteration(self):
        """Iterate sync and check both parents stay in lockstep."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            S = sync([A, B])
            
            results = []
            for s_state in S:
                results.append((s_state, A.value, B.value))
            
            expected = [
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
            ]
            assert results == expected
    
    def test_sync_different_num_states_raises(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            with pytest.raises(ValueError, match="different num_values"):
                sync([A, B])


class TestSynchronizeOp:
    """Test SynchronizeOp class directly."""
    
    def test_sync_co_op(self):
        op = SyncOp()
        assert op.compute_num_states((3, 3)) == 3
        assert op.decompose(2, (3, 3)) == (2, 2)
    
    def test_sync_co_op_n_ary(self):
        """Test SynchronizeOp with 3+ states."""
        op = SyncOp()
        assert op.compute_num_states((4, 4, 4)) == 4
        assert op.decompose(2, (4, 4, 4)) == (2, 2, 2)
        assert op.decompose(0, (3, 3, 3, 3)) == (0, 0, 0, 0)
    
    def test_sync_co_op_different_sizes_raises(self):
        op = SyncOp()
        with pytest.raises(ValueError, match="different num_values"):
            op.compute_num_states((2, 3))
    
    def test_sync_co_op_different_sizes_n_ary_raises(self):
        """Test SynchronizeOp raises for N states with different sizes."""
        op = SyncOp()
        with pytest.raises(ValueError, match="different num_values"):
            op.compute_num_states((3, 3, 4))


class TestSynchronizeStates:
    """Test synchronize_states() function for N-ary sync."""
    
    def test_sync_two_states(self):
        """sync([A, B]) creates 2-way sync."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            
            S = sync([A, B])
            assert S.num_values == 3
    
    def test_sync_three_states(self):
        """sync([A, B, C]) creates 3-way sync."""
        with Manager():
            A = State(num_values=4, name='A')
            B = State(num_values=4, name='B')
            C = State(num_values=4, name='C')
            
            S = sync([A, B, C])
            assert S.num_values == 4
    
    def test_sync_iteration_three_states(self):
        """Iterate 3-way sync and check all parents stay in lockstep."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=3, name='C')
            
            S = sync([A, B, C])
            
            results = []
            for s_state in S:
                results.append((s_state, A.value, B.value, C.value))
            
            expected = [
                (0, 0, 0, 0),
                (1, 1, 1, 1),
                (2, 2, 2, 2),
            ]
            assert results == expected
    
    def test_sync_four_states(self):
        """sync works with 4 states."""
        with Manager():
            A = State(num_values=5, name='A')
            B = State(num_values=5, name='B')
            C = State(num_values=5, name='C')
            D = State(num_values=5, name='D')
            
            S = sync([A, B, C, D])
            assert S.num_values == 5
            
            S.value = 3
            assert A.value == 3
            assert B.value == 3
            assert C.value == 3
            assert D.value == 3
    
    def test_sync_accepts_zero_states(self):
        """sync with 0 states returns State(1)."""
        with Manager():
            S = sync([])
            assert S.num_values == 1
    
    def test_sync_accepts_one_state(self):
        """sync with 1 state returns linked state."""
        with Manager():
            A = State(num_values=3, name='A')
            S = sync([A])
            assert S.num_values == 3
            S.value = 2
            assert A.value == 2
    
    def test_sync_requires_states(self):
        """sync raises for non-State arguments."""
        with Manager():
            A = State(num_values=2, name='A')
            with pytest.raises(Exception):
                sync([A, "not a state"])
    
    def test_sync_different_num_states_three_states(self):
        """sync raises when any state has different num_states."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')  # Different!
            
            with pytest.raises(ValueError, match="different num_values"):
                sync([A, B, C])
    
    def test_sync_with_name(self):
        """sync with name parameter."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=3, name='B')
            S = sync([A, B], name='Synced')
            assert S.name == 'Synced'
