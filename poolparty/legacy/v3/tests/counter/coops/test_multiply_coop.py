"""Tests for MultiplyCoOp and multiply_counters()."""
import pytest
from poolparty.counter import Counter, CounterManager, MultiplyCoOp, multiply_counters


class TestProductOperation:
    """Test product (multiplication) operation."""
    
    def test_product_num_states(self):
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        assert C.num_states == 6
    
    def test_product_initial_state(self):
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        assert A.state == 0
        assert B.state == 0
        assert C.state == 0
    
    def test_product_state_propagation_down(self):
        """Setting product state propagates to parents."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        
        # Set C=5 -> A=5%2=1, B=5//2=2
        C.state = 5
        assert A.state == 1
        assert B.state == 2
    
    def test_product_iteration(self):
        """Iterate product and check parent states."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        
        expected = [
            (0, 0, 0),  # C=0: A=0, B=0
            (1, 1, 0),  # C=1: A=1, B=0
            (2, 0, 1),  # C=2: A=0, B=1
            (3, 1, 1),  # C=3: A=1, B=1
            (4, 0, 2),  # C=4: A=0, B=2
            (5, 1, 2),  # C=5: A=1, B=2
        ]
        
        results = []
        for c_state in C:
            results.append((c_state, A.state, B.state))
        
        assert results == expected


class TestMultiplyCoOp:
    """Test MultiplyCoOp class directly."""
    
    def test_product_co_op(self):
        op = MultiplyCoOp()
        assert op.compute_num_states((2, 3)) == 6
        assert op.decompose(5, (2, 3)) == (1, 2)
    
    def test_product_co_op_n_ary(self):
        """Test MultiplyCoOp with 3+ counters."""
        op = MultiplyCoOp()
        assert op.compute_num_states((2, 3, 4)) == 24
        # decompose(state=13, sizes=(2,3,4)):
        # 13 % 2 = 1, 13 // 2 = 6
        # 6 % 3 = 0, 6 // 3 = 2
        # 2 % 4 = 2
        assert op.decompose(13, (2, 3, 4)) == (1, 0, 2)
    
    def test_product_co_op_inactive(self):
        """Test MultiplyCoOp decompose with inactive state."""
        op = MultiplyCoOp()
        assert op.decompose(-1, (2, 3)) == (-1, -1)
        assert op.decompose(-1, (2, 3, 4)) == (-1, -1, -1)


class TestMultiplyCounters:
    """Test multiply_counters() function for N-ary products."""
    
    def test_product_three_counters(self):
        """multiply_counters(A, B, C) creates 3-way product."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = Counter(num_states=4, name='C')
        
        P = multiply_counters(A, B, C)
        assert P.num_states == 24
    
    def test_product_iteration_three_counters(self):
        """Iterate 3-way product and check parent states."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = Counter(num_states=2, name='C')
        
        P = multiply_counters(A, B, C)
        
        results = []
        for p_state in P:
            results.append((p_state, A.state, B.state, C.state))
        
        # All 12 combinations with A cycling fastest, then B, then C
        expected = [
            (0, 0, 0, 0), (1, 1, 0, 0), (2, 0, 1, 0), (3, 1, 1, 0),
            (4, 0, 2, 0), (5, 1, 2, 0), (6, 0, 0, 1), (7, 1, 0, 1),
            (8, 0, 1, 1), (9, 1, 1, 1), (10, 0, 2, 1), (11, 1, 2, 1),
        ]
        assert results == expected
    
    def test_product_four_counters(self):
        """multiply_counters works with 4 counters."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=2, name='B')
        C = Counter(num_states=2, name='C')
        D = Counter(num_states=2, name='D')
        
        P = multiply_counters(A, B, C, D)
        assert P.num_states == 16
    
    def test_product_requires_two_counters(self):
        """multiply_counters raises for less than 2 counters."""
        A = Counter(num_states=2, name='A')
        with pytest.raises(ValueError, match="requires at least 2 counters"):
            multiply_counters(A)
    
    def test_product_requires_counters(self):
        """multiply_counters raises for non-Counter arguments."""
        A = Counter(num_states=2, name='A')
        with pytest.raises(TypeError, match="Expected Counter"):
            multiply_counters(A, "not a counter")
    
    def test_product_with_name(self):
        """multiply_counters with name parameter."""
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        P = multiply_counters(A, B, name='Product')
        assert P.name == 'Product'


class TestCounterTimesCounter:
    """Test A * B still creates a product counter."""
    
    def test_counter_times_counter_still_works(self):
        """A * B still creates a product counter (not repeat)."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            
            assert C.num_states == 6  # Product, not sum
            
            # Verify it's a product (both A and B active at same time)
            C.state = 5
            assert A.state == 1  # 5 % 2 = 1
            assert B.state == 2  # 5 // 2 = 2

