"""Tests for RepeatCoOp and repeat_counter()."""
import pytest
from poolparty.counter import Counter, CounterManager, RepeatCoOp, repeat_counter


class TestRepeatCoOp:
    """Test RepeatCoOp class directly."""
    
    def test_repeat_co_op_compute_num_states(self):
        op = RepeatCoOp(times=3)
        assert op.compute_num_states((2,)) == 6
        assert op.compute_num_states((5,)) == 15
    
    def test_repeat_co_op_decompose(self):
        op = RepeatCoOp(times=3)
        # For a counter with 2 states repeated 3 times:
        # States 0-5 map to parent states: 0, 1, 0, 1, 0, 1
        assert op.decompose(0, (2,)) == (0,)
        assert op.decompose(1, (2,)) == (1,)
        assert op.decompose(2, (2,)) == (0,)
        assert op.decompose(3, (2,)) == (1,)
        assert op.decompose(4, (2,)) == (0,)
        assert op.decompose(5, (2,)) == (1,)
    
    def test_repeat_co_op_inactive(self):
        op = RepeatCoOp(times=3)
        assert op.decompose(-1, (2,)) == (-1,)


class TestRepeatCounter:
    """Test repeat_counter() function."""
    
    def test_repeat_counter_num_states(self):
        """repeat_counter(A, 3) has correct num_states."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat_counter(A, 3, name='B')
            assert B.num_states == 6
    
    def test_repeat_counter_iteration(self):
        """Iterating repeat_counter cycles through A's states multiple times."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat_counter(A, 3, name='B')
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            # A cycles through 0, 1 three times
            expected = [
                (0, 0), (1, 1),  # First cycle
                (2, 0), (3, 1),  # Second cycle
                (4, 0), (5, 1),  # Third cycle
            ]
            assert results == expected
    
    def test_repeat_counter_a_never_inactive(self):
        """A never becomes -1 during iteration of repeat_counter."""
        with CounterManager() as mgr:
            A = Counter(num_states=3, name='A')
            B = repeat_counter(A, 4, name='B')
            
            for _ in B:
                assert A.state != -1
                assert A.is_active()
    
    def test_repeat_counter_times_one(self):
        """repeat_counter(A, 1) works correctly."""
        with CounterManager() as mgr:
            A = Counter(num_states=3, name='A')
            B = repeat_counter(A, 1, name='B')
            
            assert B.num_states == 3
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            expected = [(0, 0), (1, 1), (2, 2)]
            assert results == expected
    
    def test_repeat_counter_times_zero_raises(self):
        """repeat_counter with times=0 raises ValueError."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat_counter(A, 0)
    
    def test_repeat_counter_negative_times_raises(self):
        """repeat_counter with negative times raises ValueError."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat_counter(A, -1)
    
    def test_repeat_counter_non_counter_raises(self):
        """repeat_counter with non-Counter raises TypeError."""
        with pytest.raises(TypeError, match="Expected Counter"):
            repeat_counter("not a counter", 3)
    
    def test_repeat_counter_with_name(self):
        """repeat_counter with name parameter."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat_counter(A, 3, name='Repeated')
            assert B.name == 'Repeated'
    
    def test_repeat_counter_composition(self):
        """repeat_counter can be composed with other operations."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            
            # Repeat A twice, then sum with B
            A_rep = repeat_counter(A, 2, name='A_rep')
            C = A_rep + B
            C.name = 'C'
            
            assert C.num_states == 4 + 3  # 7
            
            # A_rep comes first (as passed to +), then B
            # First 4 states: A_rep cycles (A cycles twice)
            # Last 3 states: B cycles
            results = []
            for c_state in C:
                results.append((c_state, A.state, B.state))
            
            expected = [
                (0, 0, -1),   # A_rep active, A=0
                (1, 1, -1),   # A_rep active, A=1
                (2, 0, -1),   # A_rep active, A=0
                (3, 1, -1),   # A_rep active, A=1
                (4, -1, 0),   # B active
                (5, -1, 1),   # B active
                (6, -1, 2),   # B active
            ]
            assert results == expected


class TestIntegerMultiplication:
    """Test integer multiplication syntax for repeat_counter."""
    
    def test_counter_times_int(self):
        """A * 3 creates a repeat counter."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = A * 3
            
            assert B.num_states == 6
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            expected = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)]
            assert results == expected
    
    def test_int_times_counter(self):
        """3 * A creates a repeat counter."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = 3 * A
            
            assert B.num_states == 6
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            expected = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)]
            assert results == expected
    
    def test_int_mul_equivalent_to_repeat_counter(self):
        """A * 3 and 3 * A are equivalent to repeat_counter(A, 3)."""
        with CounterManager() as mgr:
            A = Counter(num_states=3, name='A')
            B1 = A * 2
            B2 = 2 * A
            B3 = repeat_counter(A, 2)
            
            # All should have same num_states
            assert B1.num_states == B2.num_states == B3.num_states == 6
    
    def test_int_mul_zero_raises(self):
        """A * 0 raises ValueError."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="at least 1"):
                B = A * 0
    
    def test_int_rmul_zero_raises(self):
        """0 * A raises ValueError."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="at least 1"):
                B = 0 * A
    
    def test_int_mul_negative_raises(self):
        """A * -1 raises ValueError."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="at least 1"):
                B = A * -1
    
    def test_int_mul_one(self):
        """A * 1 creates a valid repeat counter with same num_states."""
        with CounterManager() as mgr:
            A = Counter(num_states=3, name='A')
            B = A * 1
            
            assert B.num_states == 3
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            expected = [(0, 0), (1, 1), (2, 2)]
            assert results == expected

