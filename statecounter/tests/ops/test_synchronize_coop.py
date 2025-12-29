"""Tests for SynchronizeOp and synchronize_counters()."""
import pytest
from statecounter import Counter, Manager, SyncOp, sync


class TestSyncOperation:
    """Test sync operation."""
    
    def test_sync_num_states(self):
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = sync(A, B)
            assert S.num_states == 3
    
    def test_sync_propagates_to_both(self):
        """Setting sync state propagates to both parents."""
        with Manager():
            A = Counter(num_states=4, name='A')
            B = Counter(num_states=4, name='B')
            S = sync(A, B)
            
            S.state = 2
            assert A.state == 2
            assert B.state == 2
    
    def test_sync_iteration(self):
        """Iterate sync and check both parents stay in lockstep."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = sync(A, B)
            
            results = []
            for s_state in S:
                results.append((s_state, A.state, B.state))
            
            expected = [
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
            ]
            assert results == expected
    
    def test_sync_different_num_states_raises(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            with pytest.raises(ValueError, match="different num_states"):
                sync(A, B)


class TestSynchronizeOp:
    """Test SynchronizeOp class directly."""
    
    def test_sync_co_op(self):
        op = SyncOp()
        assert op.compute_num_states((3, 3)) == 3
        assert op.decompose(2, (3, 3)) == (2, 2)
    
    def test_sync_co_op_n_ary(self):
        """Test SynchronizeOp with 3+ counters."""
        op = SyncOp()
        assert op.compute_num_states((4, 4, 4)) == 4
        assert op.decompose(2, (4, 4, 4)) == (2, 2, 2)
        assert op.decompose(0, (3, 3, 3, 3)) == (0, 0, 0, 0)
    
    def test_sync_co_op_different_sizes_raises(self):
        op = SyncOp()
        with pytest.raises(ValueError, match="different num_states"):
            op.compute_num_states((2, 3))
    
    def test_sync_co_op_different_sizes_n_ary_raises(self):
        """Test SynchronizeOp raises for N counters with different sizes."""
        op = SyncOp()
        with pytest.raises(ValueError, match="different num_states"):
            op.compute_num_states((3, 3, 4))


class TestSynchronizeCounters:
    """Test synchronize_counters() function for N-ary sync."""
    
    def test_sync_two_counters(self):
        """synchronize_counters(A, B) creates 2-way sync."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            
            S = sync(A, B)
            assert S.num_states == 3
    
    def test_sync_three_counters(self):
        """synchronize_counters(A, B, C) creates 3-way sync."""
        with Manager():
            A = Counter(num_states=4, name='A')
            B = Counter(num_states=4, name='B')
            C = Counter(num_states=4, name='C')
            
            S = sync(A, B, C)
            assert S.num_states == 4
    
    def test_sync_iteration_three_counters(self):
        """Iterate 3-way sync and check all parents stay in lockstep."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=3, name='C')
            
            S = sync(A, B, C)
            
            results = []
            for s_state in S:
                results.append((s_state, A.state, B.state, C.state))
            
            expected = [
                (0, 0, 0, 0),
                (1, 1, 1, 1),
                (2, 2, 2, 2),
            ]
            assert results == expected
    
    def test_sync_four_counters(self):
        """synchronize_counters works with 4 counters."""
        with Manager():
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            C = Counter(num_states=5, name='C')
            D = Counter(num_states=5, name='D')
            
            S = sync(A, B, C, D)
            assert S.num_states == 5
            
            S.state = 3
            assert A.state == 3
            assert B.state == 3
            assert C.state == 3
            assert D.state == 3
    
    def test_sync_accepts_zero_counters(self):
        """synchronize_counters with 0 counters returns Counter(1)."""
        with Manager():
            S = sync()
            assert S.num_states == 1
    
    def test_sync_accepts_one_counter(self):
        """synchronize_counters with 1 counter returns linked counter."""
        with Manager():
            A = Counter(num_states=3, name='A')
            S = sync(A)
            assert S.num_states == 3
            S.state = 2
            assert A.state == 2
    
    def test_sync_requires_counters(self):
        """synchronize_counters raises for non-Counter arguments."""
        with Manager():
            A = Counter(num_states=2, name='A')
            with pytest.raises(TypeError, match="Expected Counter"):
                sync(A, "not a counter")
    
    def test_sync_different_num_states_three_counters(self):
        """synchronize_counters raises when any counter has different num_states."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')  # Different!
            
            with pytest.raises(ValueError, match="different num_states"):
                sync(A, B, C)
    
    def test_sync_with_name(self):
        """synchronize_counters with name parameter."""
        with Manager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = sync(A, B, name='Synced')
            assert S.name == 'Synced'
