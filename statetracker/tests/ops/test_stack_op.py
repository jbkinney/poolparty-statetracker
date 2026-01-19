"""Tests for StackOp and sum_states()."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, StackOp, stack


class TestStackOperation:
    """Test sum (addition) operation."""
    
    def test_stack_num_states(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=4, name='B')
            C = stack([A,B])
            assert C.num_values == 6
    
    def test_stack_initial_state(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=4, name='B')
            C = stack([A,B])
            assert C.value is None  # Derived state defaults to inactive
    
    def test_stack_propagation_a_branch(self):
        """Sum in A's branch propagates correctly, B is inactive."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=4, name='B')
            C = stack([A,B])
            
            # State 0-1 are in A's branch
            C.value = 1
            assert A.value == 1
            assert B.value is None  # B is inactive
    
    def test_stack_propagation_b_branch(self):
        """Sum in B's branch propagates correctly, A is inactive."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=4, name='B')
            C = stack([A,B])
            
            # State 2-5 are in B's branch (offset by 2)
            C.value = 4
            assert A.value is None  # A is inactive
            assert B.value == 2  # 4 - 2 = 2
    
    def test_stack_iteration(self):
        """Iterate sum and check parent states with inactive markers."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = stack([A,B])
            
            expected = [
                (0, 0, None),   # C=0: A=0 (A branch), B inactive
                (1, 1, None),   # C=1: A=1 (A branch), B inactive
                (2, None, 0),   # C=2: B=0 (B branch), A inactive
                (3, None, 1),   # C=3: B=1 (B branch), A inactive
                (4, None, 2),   # C=4: B=2 (B branch), A inactive
            ]
            
            results = []
            for c_state in C:
                results.append((c_state, A.value, B.value))
            
            assert results == expected


class TestStackOp:
    """Test StackOp class directly."""
    
    def test_stack_co_op(self):
        op = StackOp()
        assert op.compute_num_states((2, 4)) == 6
        assert op.decompose(1, (2, 4)) == (1, None)  # A active, B inactive
        assert op.decompose(4, (2, 4)) == (None, 2)  # A inactive, B active
    
    def test_stack_co_op_n_ary(self):
        """Test StackOp with 3+ states."""
        op = StackOp()
        assert op.compute_num_states((2, 3, 4)) == 9
        # state 0-1: first state (size 2) active
        assert op.decompose(0, (2, 3, 4)) == (0, None, None)
        assert op.decompose(1, (2, 3, 4)) == (1, None, None)
        # state 2-4: second state (size 3) active (offset by 2)
        assert op.decompose(2, (2, 3, 4)) == (None, 0, None)
        assert op.decompose(4, (2, 3, 4)) == (None, 2, None)
        # state 5-8: third state (size 4) active (offset by 2+3=5)
        assert op.decompose(5, (2, 3, 4)) == (None, None, 0)
        assert op.decompose(8, (2, 3, 4)) == (None, None, 3)
    
    def test_stack_co_op_inactive(self):
        """Test StackOp decompose with inactive state."""
        op = StackOp()
        assert op.decompose(None, (2, 4)) == (None, None)
        assert op.decompose(None, (2, 3, 4)) == (None, None, None)


class TestSumStates:
    """Test sum_states() function for N-ary sums."""
    
    def test_stack_three_states(self):
        """sum_states(A, B, C) creates 3-way sum."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')
            
            S = stack([A, B, C])
            assert S.num_values == 9
    
    def test_stack_iteration_three_states(self):
        """Iterate 3-way sum and check parent states with inactives."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=2, name='C')
            
            S = stack([A, B, C])
            
            results = []
            for s_state in S:
                results.append((s_state, A.value, B.value, C.value))
            
            # 7 states: A (0-1), B (2-4), C (5-6)
            expected = [
                (0, 0, None, None),   # A active
                (1, 1, None, None),   # A active
                (2, None, 0, None),   # B active
                (3, None, 1, None),   # B active
                (4, None, 2, None),   # B active
                (5, None, None, 0),   # C active
                (6, None, None, 1),   # C active
            ]
            assert results == expected
    
    def test_stack_four_states(self):
        """sum_states works with 4 states."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')
            D = State(num_values=5, name='D')
            
            S = stack([A, B, C, D])
            assert S.num_values == 14
    
    def test_stack_accepts_zero_states(self):
        """sum_states with 0 states returns State(0)."""
        with Manager():
            S = stack([])
            assert S.num_values == 0
    
    def test_stack_accepts_one_state(self):
        """sum_states with 1 state returns linked state."""
        with Manager():
            A = State(num_values=3, name='A')
            S = stack([A])
            assert S.num_values == 3
            S.value = 2
            assert A.value == 2
    
    def test_stack_requires_states(self):
        """sum_states raises for non-State arguments."""
        with Manager():
            A = State(num_values=2, name='A')
            with pytest.raises(Exception):
                stack([A, 123])
    
    def test_stack_with_name(self):
        """sum_states with name parameter."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            S = stack([A, B], name='Sum')
            assert S.name == 'Sum'
