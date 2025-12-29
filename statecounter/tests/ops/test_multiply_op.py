"""Tests for ProductOp and product_counters()."""
import pytest
from statecounter import Counter, Manager, ProductOp, product


class TestProductOperation:
    """Test product (multiplication) operation."""
    
    def test_product_num_states(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            assert C.num_states == 6
    
    def test_product_initial_state(self):
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            assert A.state == 0
            assert B.state == 0
            assert C.state == 0
    
    def test_product_state_propagation_down(self):
        """Setting product state propagates to parents."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            
            # Set C=5 -> A=5%2=1, B=5//2=2
            C.state = 5
            assert A.state == 1
            assert B.state == 2
    
    def test_product_iteration(self):
        """Iterate product and check parent states."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            
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


class TestProductOp:
    """Test ProductOp class directly."""
    
    def test_product_co_op(self):
        op = ProductOp()
        assert op.compute_num_states((2, 3)) == 6
        assert op.decompose(5, (2, 3)) == (1, 2)
    
    def test_product_co_op_n_ary(self):
        """Test ProductOp with 3+ counters."""
        op = ProductOp()
        assert op.compute_num_states((2, 3, 4)) == 24
        # decompose(state=13, sizes=(2,3,4)):
        # 13 % 2 = 1, 13 // 2 = 6
        # 6 % 3 = 0, 6 // 3 = 2
        # 2 % 4 = 2
        assert op.decompose(13, (2, 3, 4)) == (1, 0, 2)
    
    def test_product_co_op_inactive(self):
        """Test ProductOp decompose with inactive state."""
        op = ProductOp()
        assert op.decompose(None, (2, 3)) == (None, None)
        assert op.decompose(None, (2, 3, 4)) == (None, None, None)


class TestMultiplyCounters:
    """Test product_counters() function for N-ary products."""
    
    def test_product_three_counters(self):
        """product_counters(A, B, C) creates 3-way product."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            
            P = product([A, B, C])
            assert P.num_states == 24
    
    def test_product_iteration_three_counters(self):
        """Iterate 3-way product and check parent states."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=2, name='C')
            
            P = product([A, B, C])
            
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
        """product_counters works with 4 counters."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=2, name='B')
            C = Counter(num_states=2, name='C')
            D = Counter(num_states=2, name='D')
            
            P = product([A, B, C, D])
            assert P.num_states == 16
    
    def test_product_accepts_zero_counters(self):
        """product_counters with 0 counters returns Counter(1)."""
        with Manager():
            P = product([])
            assert P.num_states == 1
    
    def test_product_accepts_one_counter(self):
        """product_counters with 1 counter returns linked counter."""
        with Manager():
            A = Counter(num_states=3, name='A')
            P = product([A])
            assert P.num_states == 3
            P.state = 2
            assert A.state == 2
    
    def test_product_requires_counters(self):
        """product_counters raises for non-Counter arguments."""
        with Manager():
            A = Counter(num_states=2, name='A')
            with pytest.raises(TypeError, match="Expected Counter"):
                product([A, 123], "not a counter")
    
    def test_product_with_name(self):
        """product_counters with name parameter."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            P = product([A, B], name='Product')
            assert P.name == 'Product'
