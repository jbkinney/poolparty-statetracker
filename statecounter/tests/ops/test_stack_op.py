"""Tests for StackOp and sum_counters()."""
import pytest
from statecounter import Counter, Manager, StackOp, stack


class TestStackOperation:
    """Test sum (addition) operation."""
    
    def test_stack_num_states(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=4, name='B')
            C = stack([A,B])
            assert C.num_states == 6
    
    def test_stack_initial_state(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=4, name='B')
            C = stack([A,B])
            assert C.state == 0
    
    def test_stack_propagation_a_branch(self):
        """Sum in A's branch propagates correctly, B is inactive."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=4, name='B')
            C = stack([A,B])
            
            # State 0-1 are in A's branch
            C.state = 1
            assert A.state == 1
            assert B.state is None  # B is inactive
    
    def test_stack_propagation_b_branch(self):
        """Sum in B's branch propagates correctly, A is inactive."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=4, name='B')
            C = stack([A,B])
            
            # State 2-5 are in B's branch (offset by 2)
            C.state = 4
            assert A.state is None  # A is inactive
            assert B.state == 2  # 4 - 2 = 2
    
    def test_stack_iteration(self):
        """Iterate sum and check parent states with inactive markers."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
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
                results.append((c_state, A.state, B.state))
            
            assert results == expected


class TestStackOp:
    """Test StackOp class directly."""
    
    def test_stack_co_op(self):
        op = StackOp()
        assert op.compute_num_states((2, 4)) == 6
        assert op.decompose(1, (2, 4)) == (1, None)  # A active, B inactive
        assert op.decompose(4, (2, 4)) == (None, 2)  # A inactive, B active
    
    def test_stack_co_op_n_ary(self):
        """Test StackOp with 3+ counters."""
        op = StackOp()
        assert op.compute_num_states((2, 3, 4)) == 9
        # state 0-1: first counter (size 2) active
        assert op.decompose(0, (2, 3, 4)) == (0, None, None)
        assert op.decompose(1, (2, 3, 4)) == (1, None, None)
        # state 2-4: second counter (size 3) active (offset by 2)
        assert op.decompose(2, (2, 3, 4)) == (None, 0, None)
        assert op.decompose(4, (2, 3, 4)) == (None, 2, None)
        # state 5-8: third counter (size 4) active (offset by 2+3=5)
        assert op.decompose(5, (2, 3, 4)) == (None, None, 0)
        assert op.decompose(8, (2, 3, 4)) == (None, None, 3)
    
    def test_stack_co_op_inactive(self):
        """Test StackOp decompose with inactive state."""
        op = StackOp()
        assert op.decompose(None, (2, 4)) == (None, None)
        assert op.decompose(None, (2, 3, 4)) == (None, None, None)


class TestSumCounters:
    """Test sum_counters() function for N-ary sums."""
    
    def test_stack_three_counters(self):
        """sum_counters(A, B, C) creates 3-way sum."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            
            S = stack([A, B, C])
            assert S.num_states == 9
    
    def test_stack_iteration_three_counters(self):
        """Iterate 3-way sum and check parent states with inactives."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=2, name='C')
            
            S = stack([A, B, C])
            
            results = []
            for s_state in S:
                results.append((s_state, A.state, B.state, C.state))
            
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
    
    def test_stack_four_counters(self):
        """sum_counters works with 4 counters."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            D = Counter(num_states=5, name='D')
            
            S = stack([A, B, C, D])
            assert S.num_states == 14
    
    def test_stack_accepts_zero_counters(self):
        """sum_counters with 0 counters returns Counter(0)."""
        with Manager():
            S = stack([])
            assert S.num_states == 0
    
    def test_stack_accepts_one_counter(self):
        """sum_counters with 1 counter returns linked counter."""
        with Manager():
            A = Counter(num_states=3, name='A')
            S = stack([A])
            assert S.num_states == 3
            S.state = 2
            assert A.state == 2
    
    def test_stack_requires_counters(self):
        """sum_counters raises for non-Counter arguments."""
        with Manager():
            A = Counter(num_states=2, name='A')
            with pytest.raises(TypeError, match="Expected Counter"):
                stack([A, 123])
    
    def test_stack_with_name(self):
        """sum_counters with name parameter."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            S = stack([A, B], name='Sum')
            assert S.name == 'Sum'
