"""Tests for RepeatOp and repeat_counter()."""
import pytest
from statecounter import Counter, Manager, RepeatOp, repeat, stack


class TestRepeatOp:
    """Test RepeatOp class directly."""
    
    def test_repeat_co_op_compute_num_states(self):
        op = RepeatOp(times=3)
        assert op.compute_num_states((2,)) == 6
        assert op.compute_num_states((5,)) == 15
    
    def test_repeat_co_op_decompose(self):
        op = RepeatOp(times=3)
        # For a counter with 2 states repeated 3 times:
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


class TestRepeatCounter:
    """Test repeat_counter() function."""
    
    def test_repeat_counter_num_states(self):
        """repeat_counter(A, 3) has correct num_states."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat(A, 3, name='B')
            assert B.num_states == 6
    
    def test_repeat_counter_iteration(self):
        """Iterating repeat_counter cycles through A's states multiple times."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat(A, 3, name='B')
            
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
        """A never becomes None during iteration of repeat_counter."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = repeat(A, 4, name='B')
            
            for _ in B:
                assert A.state is not None
                assert A.is_active()
    
    def test_repeat_counter_times_one(self):
        """repeat_counter(A, 1) works correctly."""
        with Manager() as mgr:
            A = Counter(num_states=3, name='A')
            B = repeat(A, 1, name='B')
            
            assert B.num_states == 3
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            expected = [(0, 0), (1, 1), (2, 2)]
            assert results == expected
    
    def test_repeat_counter_times_zero_raises(self):
        """repeat_counter with times=0 raises ValueError."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat(A, 0)
    
    def test_repeat_counter_negative_times_raises(self):
        """repeat_counter with negative times raises ValueError."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="times must be at least 1"):
                repeat(A, -1)
    
    def test_repeat_counter_non_counter_raises(self):
        """repeat_counter with non-Counter raises TypeError."""
        with pytest.raises(TypeError, match="Expected Counter"):
            repeat("not a counter", 3)
    
    def test_repeat_counter_with_name(self):
        """repeat_counter with name parameter."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = repeat(A, 3, name='Repeated')
            assert B.name == 'Repeated'
    
    def test_repeat_counter_composition(self):
        """repeat_counter can be composed with other operations."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            
            # Repeat A twice, then sum with B
            A_rep = repeat(A, 2, name='A_rep')
            C = stack([B,A_rep])
            C.name = 'C'
            
            assert C.num_states == 4 + 3  # 7
            
            # Parents are sorted by (iteration_order, _id)
            # B has id=1, A_rep has id=2, so B comes first in sum
            # First 3 states: B cycles
            # Last 4 states: A_rep cycles (A cycles twice)
            results = []
            for c_state in C:
                results.append((c_state, A.state, B.state))
            
            expected = [
                (0, None, 0),   # B active
                (1, None, 1),   # B active
                (2, None, 2),   # B active
                (3, 0, None),   # A_rep active, A=0
                (4, 1, None),   # A_rep active, A=1
                (5, 0, None),   # A_rep active, A=0
                (6, 1, None),   # A_rep active, A=1
            ]
            assert results == expected
