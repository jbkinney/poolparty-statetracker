"""Tests for SliceCoOp and slice_counter()."""
import pytest
from poolparty.counter import Counter, SliceCoOp, slice_counter


class TestSliceOperation:
    """Test slice operation."""
    
    def test_slice_basic(self):
        """Basic slicing: A[1:5]."""
        A = Counter(num_states=8, name='A')
        B = A[1:5]
        assert B.num_states == 4
    
    def test_slice_propagation(self):
        """Slice propagates state to parent correctly."""
        A = Counter(num_states=8, name='A')
        B = A[1:5]  # States 1, 2, 3, 4
        
        B.state = 0
        assert A.state == 1
        
        B.state = 3
        assert A.state == 4
    
    def test_slice_iteration(self):
        """Iterate sliced counter and check parent states."""
        A = Counter(num_states=8, name='A')
        B = A[2:6]  # States 2, 3, 4, 5
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
        ]
        assert results == expected
    
    def test_slice_with_step(self):
        """Step slicing: A[::2]."""
        A = Counter(num_states=8, name='A')
        B = A[::2]  # States 0, 2, 4, 6
        assert B.num_states == 4
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 0),
            (1, 2),
            (2, 4),
            (3, 6),
        ]
        assert results == expected
    
    def test_slice_with_step_and_start(self):
        """Step slicing with start: A[1::2]."""
        A = Counter(num_states=8, name='A')
        B = A[1::2]  # States 1, 3, 5, 7
        assert B.num_states == 4
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 1),
            (1, 3),
            (2, 5),
            (3, 7),
        ]
        assert results == expected
    
    def test_slice_negative_indices(self):
        """Negative indices: A[-3:]."""
        A = Counter(num_states=8, name='A')
        B = A[-3:]  # States 5, 6, 7
        assert B.num_states == 3
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 5),
            (1, 6),
            (2, 7),
        ]
        assert results == expected
    
    def test_slice_negative_step_reverse(self):
        """Negative step (reverse): A[::-1]."""
        A = Counter(num_states=4, name='A')
        B = A[::-1]  # States 3, 2, 1, 0 (reversed)
        assert B.num_states == 4
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
        ]
        assert results == expected
    
    def test_slice_negative_step_partial(self):
        """Negative step partial: A[5:1:-1]."""
        A = Counter(num_states=8, name='A')
        B = A[5:1:-1]  # States 5, 4, 3, 2
        assert B.num_states == 4
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 5),
            (1, 4),
            (2, 3),
            (3, 2),
        ]
        assert results == expected
    
    def test_slice_negative_step_with_step_size(self):
        """Negative step with step size: A[::-2]."""
        A = Counter(num_states=8, name='A')
        B = A[::-2]  # States 7, 5, 3, 1
        assert B.num_states == 4
        
        results = []
        for b_state in B:
            results.append((b_state, A.state))
        
        expected = [
            (0, 7),
            (1, 5),
            (2, 3),
            (3, 1),
        ]
        assert results == expected
    
    def test_slice_composition_with_product(self):
        """Composition with other operations: (A * B)[::2]."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B  # 6 states
        D = C[::2]  # 3 states (0, 2, 4)
        assert D.num_states == 3
        
        results = []
        for d_state in D:
            results.append((d_state, C.state, A.state, B.state))
        
        # D=0 -> C=0 -> A=0, B=0
        # D=1 -> C=2 -> A=0, B=1
        # D=2 -> C=4 -> A=0, B=2
        expected = [
            (0, 0, 0, 0),
            (1, 2, 0, 1),
            (2, 4, 0, 2),
        ]
        assert results == expected
    
    def test_slice_integer_raises(self):
        """Indexing with integer raises TypeError."""
        A = Counter(num_states=8, name='A')
        with pytest.raises(TypeError, match="Counter indices must be slices"):
            A[3]
    
    def test_slice_empty_result(self):
        """Empty slice returns counter with 0 states."""
        A = Counter(num_states=8, name='A')
        B = A[5:2]  # Empty (start > stop with positive step)
        assert B.num_states == 0
    
    def test_slice_inactive_state(self):
        """Slice handles inactive state (-1)."""
        A = Counter(num_states=8, name='A')
        B = A[1:5]
        B.state = -1
        assert A.state == -1


class TestSliceCoOp:
    """Test SliceCoOp class directly."""
    
    def test_slice_co_op_compute_num_states(self):
        op = SliceCoOp(1, 5, 1)
        assert op.compute_num_states((8,)) == 4
    
    def test_slice_co_op_decompose(self):
        op = SliceCoOp(1, 5, 1)
        assert op.decompose(0, (8,)) == (1,)
        assert op.decompose(3, (8,)) == (4,)
    
    def test_slice_co_op_with_step(self):
        op = SliceCoOp(0, 8, 2)
        assert op.compute_num_states((8,)) == 4
        assert op.decompose(0, (8,)) == (0,)
        assert op.decompose(1, (8,)) == (2,)
        assert op.decompose(3, (8,)) == (6,)
    
    def test_slice_co_op_negative_step(self):
        op = SliceCoOp(7, -1, -1)  # Reverse
        assert op.compute_num_states((8,)) == 8
        assert op.decompose(0, (8,)) == (7,)
        assert op.decompose(7, (8,)) == (0,)
    
    def test_slice_co_op_inactive(self):
        op = SliceCoOp(1, 5, 1)
        assert op.decompose(-1, (8,)) == (-1,)


class TestSliceCounterFunction:
    """Test slice_counter() helper function."""
    
    def test_slice_counter_basic(self):
        A = Counter(num_states=8, name='A')
        B = slice_counter(A, 1, 5)
        assert B.num_states == 4
    
    def test_slice_counter_with_step(self):
        A = Counter(num_states=8, name='A')
        B = slice_counter(A, step=2)
        assert B.num_states == 4
    
    def test_slice_counter_reverse(self):
        A = Counter(num_states=4, name='A')
        B = slice_counter(A, step=-1)
        assert B.num_states == 4
        
        B.state = 0
        assert A.state == 3
    
    def test_slice_counter_with_name(self):
        A = Counter(num_states=8, name='A')
        B = slice_counter(A, 1, 5, name='Sliced')
        assert B.name == 'Sliced'
    
    def test_slice_counter_not_counter_raises(self):
        with pytest.raises(TypeError, match="Expected Counter"):
            slice_counter("not a counter", 0, 5)
    
    def test_slice_counter_partial_args(self):
        """slice_counter with partial arguments."""
        A = Counter(num_states=8, name='A')
        
        # Just start
        B = slice_counter(A, start=3)
        assert B.num_states == 5  # 3, 4, 5, 6, 7
        
        # Just stop
        C = slice_counter(A, stop=3)
        assert C.num_states == 3  # 0, 1, 2

