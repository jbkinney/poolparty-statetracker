"""Tests for SliceOp and slice_state()."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, SliceOp, slice, product


class TestSliceOperation:
    """Test slice operation."""
    
    def test_slice_basic(self):
        """Basic slicing: A[1:5]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[1:5]
            assert B.num_values == 4
    
    def test_slice_propagation(self):
        """Slice propagates state to parent correctly."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[1:5]  # States 1, 2, 3, 4
            
            B.value = 0
            assert A.value == 1
            
            B.value = 3
            assert A.value == 4
    
    def test_slice_iteration(self):
        """Iterate sliced state and check parent states."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[2:6]  # States 2, 3, 4, 5
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 5),
            ]
            assert results == expected
    
    def test_slice_with_step(self):
        """Step slicing: A[::2]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[::2]  # States 0, 2, 4, 6
            assert B.num_values == 4
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 0),
                (1, 2),
                (2, 4),
                (3, 6),
            ]
            assert results == expected
    
    def test_slice_with_step_and_start(self):
        """Step slicing with start: A[1::2]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[1::2]  # States 1, 3, 5, 7
            assert B.num_values == 4
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 1),
                (1, 3),
                (2, 5),
                (3, 7),
            ]
            assert results == expected
    
    def test_slice_negative_indices(self):
        """Negative indices: A[-3:]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[-3:]  # States 5, 6, 7
            assert B.num_values == 3
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 5),
                (1, 6),
                (2, 7),
            ]
            assert results == expected
    
    def test_slice_negative_step_reverse(self):
        """Negative step (reverse): A[::-1]."""
        with Manager():
            A = State(num_values=4, name='A')
            B = A[::-1]  # States 3, 2, 1, 0 (reversed)
            assert B.num_values == 4
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 3),
                (1, 2),
                (2, 1),
                (3, 0),
            ]
            assert results == expected
    
    def test_slice_negative_step_partial(self):
        """Negative step partial: A[5:1:-1]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[5:1:-1]  # States 5, 4, 3, 2
            assert B.num_values == 4
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 5),
                (1, 4),
                (2, 3),
                (3, 2),
            ]
            assert results == expected
    
    def test_slice_negative_step_with_step_size(self):
        """Negative step with step size: A[::-2]."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[::-2]  # States 7, 5, 3, 1
            assert B.num_values == 4
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            expected = [
                (0, 7),
                (1, 5),
                (2, 3),
                (3, 1),
            ]
            assert results == expected
    
    def test_slice_composition_with_product(self):
        """Composition with other operations: product([A, B])[::2]."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])  # 6 states
            D = C[::2]  # 3 states (0, 2, 4)
            assert D.num_values == 3
            
            results = []
            for d_state in D:
                results.append((d_state, C.value, A.value, B.value))
            
            # D=0 -> C=0 -> A=0, B=0
            # D=1 -> C=2 -> A=0, B=1
            # D=2 -> C=4 -> A=0, B=2
            expected = [
                (0, 0, 0, 0),
                (1, 2, 0, 1),
                (2, 4, 0, 2),
            ]
            assert results == expected
    
    def test_slice_integer(self):
        """Indexing with integer creates single-state slice."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[3]
            assert B.num_values == 1
            B.value = 0
            assert B.value == 0  # B's state is 0, which maps to A's state 3
            assert A.value == 3
    
    def test_slice_empty_result(self):
        """Empty slice returns state with 0 states."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[5:2]  # Empty (start > stop with positive step)
            assert B.num_values == 0
    
    def test_slice_inactive_state(self):
        """Slice handles inactive state (None)."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[1:5]
            B.value = None
            # Setting derived state to None doesn't propagate to parent
            # A remains at its default state (0 for leaf state)
            assert A.value == 0
            assert B.value is None


class TestSliceOp:
    """Test SliceOp class directly."""
    
    def test_slice_co_op_compute_num_states(self):
        op = SliceOp(1, 5, 1)
        assert op.compute_num_states((8,)) == 4
    
    def test_slice_co_op_decompose(self):
        op = SliceOp(1, 5, 1)
        assert op.decompose(0, (8,)) == (1,)
        assert op.decompose(3, (8,)) == (4,)
    
    def test_slice_co_op_with_step(self):
        op = SliceOp(0, 8, 2)
        assert op.compute_num_states((8,)) == 4
        assert op.decompose(0, (8,)) == (0,)
        assert op.decompose(1, (8,)) == (2,)
        assert op.decompose(3, (8,)) == (6,)
    
    def test_slice_co_op_negative_step(self):
        op = SliceOp(7, -1, -1)  # Reverse
        assert op.compute_num_states((8,)) == 8
        assert op.decompose(0, (8,)) == (7,)
        assert op.decompose(7, (8,)) == (0,)
    
    def test_slice_co_op_inactive(self):
        op = SliceOp(1, 5, 1)
        assert op.decompose(None, (8,)) == (None,)


class TestSliceStateFunction:
    """Test slice_state() helper function."""
    
    def test_slice_state_basic(self):
        with Manager():
            A = State(num_values=8, name='A')
            B = slice(A, 1, 5)
            assert B.num_values == 4
    
    def test_slice_state_with_step(self):
        with Manager():
            A = State(num_values=8, name='A')
            B = slice(A, step=2)
            assert B.num_values == 4
    
    def test_slice_state_reverse(self):
        with Manager():
            A = State(num_values=4, name='A')
            B = slice(A, step=-1)
            assert B.num_values == 4
            
            B.value = 0
            assert A.value == 3
    
    def test_slice_state_with_name(self):
        with Manager():
            A = State(num_values=8, name='A')
            B = slice(A, 1, 5, name='Sliced')
            assert B.name == 'Sliced'
    
    def test_slice_state_not_state_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation):
            slice("not a state", 0, 5)
    
    def test_slice_state_partial_args(self):
        """slice_state with partial arguments."""
        with Manager():
            A = State(num_values=8, name='A')
            
            # Just start
            B = slice(A, start=3)
            assert B.num_values == 5  # 3, 4, 5, 6, 7
            
            # Just stop
            C = slice(A, stop=3)
            assert C.num_values == 3  # 0, 1, 2
