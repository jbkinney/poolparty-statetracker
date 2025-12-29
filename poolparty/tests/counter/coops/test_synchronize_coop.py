"""Tests for SynchronizeCoOp and synchronize_counters()."""
import pytest
from poolparty.counter import Counter, CounterManager, SynchronizeCoOp, synchronize_counters


class TestSyncOperation:
    """Test sync operation."""
    
    def test_sync_num_states(self):
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = synchronize_counters(A, B)
            assert S.num_states == 3
    
    def test_sync_propagates_to_both(self):
        """Setting sync state propagates to both parents."""
        with CounterManager():
            A = Counter(num_states=4, name='A')
            B = Counter(num_states=4, name='B')
            S = synchronize_counters(A, B)
            
            S.state = 2
            assert A.state == 2
            assert B.state == 2
    
    def test_sync_iteration(self):
        """Iterate sync and check both parents stay in lockstep."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = synchronize_counters(A, B)
            
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
        with CounterManager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            with pytest.raises(ValueError, match="different num_states"):
                synchronize_counters(A, B)


class TestSynchronizeCoOp:
    """Test SynchronizeCoOp class directly."""
    
    def test_sync_co_op(self):
        op = SynchronizeCoOp()
        assert op.compute_num_states((3, 3)) == 3
        assert op.decompose(2, (3, 3)) == (2, 2)
    
    def test_sync_co_op_n_ary(self):
        """Test SynchronizeCoOp with 3+ counters."""
        op = SynchronizeCoOp()
        assert op.compute_num_states((4, 4, 4)) == 4
        assert op.decompose(2, (4, 4, 4)) == (2, 2, 2)
        assert op.decompose(0, (3, 3, 3, 3)) == (0, 0, 0, 0)
    
    def test_sync_co_op_different_sizes_raises(self):
        op = SynchronizeCoOp()
        with pytest.raises(ValueError, match="different num_states"):
            op.compute_num_states((2, 3))
    
    def test_sync_co_op_different_sizes_n_ary_raises(self):
        """Test SynchronizeCoOp raises for N counters with different sizes."""
        op = SynchronizeCoOp()
        with pytest.raises(ValueError, match="different num_states"):
            op.compute_num_states((3, 3, 4))


class TestSynchronizeCounters:
    """Test synchronize_counters() function for N-ary sync."""
    
    def test_sync_two_counters(self):
        """synchronize_counters(A, B) creates 2-way sync."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            
            S = synchronize_counters(A, B)
            assert S.num_states == 3
    
    def test_sync_three_counters(self):
        """synchronize_counters(A, B, C) creates 3-way sync."""
        with CounterManager():
            A = Counter(num_states=4, name='A')
            B = Counter(num_states=4, name='B')
            C = Counter(num_states=4, name='C')
            
            S = synchronize_counters(A, B, C)
            assert S.num_states == 4
    
    def test_sync_iteration_three_counters(self):
        """Iterate 3-way sync and check all parents stay in lockstep."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=3, name='C')
            
            S = synchronize_counters(A, B, C)
            
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
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            C = Counter(num_states=5, name='C')
            D = Counter(num_states=5, name='D')
            
            S = synchronize_counters(A, B, C, D)
            assert S.num_states == 5
            
            S.state = 3
            assert A.state == 3
            assert B.state == 3
            assert C.state == 3
            assert D.state == 3
    
    def test_sync_accepts_zero_counters(self):
        """synchronize_counters with 0 counters returns Counter(1)."""
        with CounterManager():
            S = synchronize_counters()
            assert S.num_states == 1
    
    def test_sync_accepts_one_counter(self):
        """synchronize_counters with 1 counter returns linked counter."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            S = synchronize_counters(A)
            assert S.num_states == 3
            S.state = 2
            assert A.state == 2
    
    def test_sync_requires_counters(self):
        """synchronize_counters raises for non-Counter arguments."""
        with CounterManager():
            A = Counter(num_states=2, name='A')
            with pytest.raises(TypeError, match="Expected Counter"):
                synchronize_counters(A, "not a counter")
    
    def test_sync_different_num_states_three_counters(self):
        """synchronize_counters raises when any counter has different num_states."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')  # Different!
            
            with pytest.raises(ValueError, match="different num_states"):
                synchronize_counters(A, B, C)
    
    def test_sync_with_name(self):
        """synchronize_counters with name parameter."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            S = synchronize_counters(A, B, name='Synced')
            assert S.name == 'Synced'
