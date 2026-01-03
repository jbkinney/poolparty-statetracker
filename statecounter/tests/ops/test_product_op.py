"""Tests for ProductOp and product_counters()."""
import pytest
from statecounter import Counter, Manager, ProductOp, product, ordered_product, stack

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
            assert A.state == 0  # Leaf counter defaults to 0
            assert B.state == 0  # Leaf counter defaults to 0
            assert C.state is None  # Derived counter defaults to inactive
    
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
            with pytest.raises(Exception):
                product([A, 123], "not a counter")
    
    def test_product_with_name(self):
        """product_counters with name parameter."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            P = product([A, B], name='Product')
            assert P.name == 'Product'


class TestOrderedProductFlattening:
    """Test ordered_product recursive flattening and deduplication."""

    def test_ordered_product_flattens_nested_products(self):
        """ordered_product([A*B, C]) should flatten to ordered_product([A, B, C])."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')

            # Create nested product
            AB = ordered_product([A, B])
            ABC_nested = ordered_product([AB, C])

            # Create flat product
            ABC_flat = ordered_product([A, B, C])

            # Both should have same num_states
            assert ABC_nested.num_states == ABC_flat.num_states == 24

            # Both should iterate the same way
            nested_results = []
            for state in ABC_nested:
                nested_results.append((state, A.state, B.state, C.state))

            flat_results = []
            for state in ABC_flat:
                flat_results.append((state, A.state, B.state, C.state))

            assert nested_results == flat_results

    def test_ordered_product_deduplicates_through_nested_products(self):
        """ordered_product([A*B, C, D*A]) should deduplicate A."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            D = Counter(num_states=5, name='D')

            AB = ordered_product([A, B])
            DA = ordered_product([D, A])

            # Without flattening, this would have A twice
            # With flattening, A should be deduplicated
            result = ordered_product([AB, C, DA])

            # Should be A * B * C * D = 2 * 3 * 4 * 5 = 120
            # NOT (A * B) * C * (D * A) = 6 * 4 * 10 = 240
            assert result.num_states == 120

    def test_ordered_product_diamond_pattern_no_conflict(self):
        """Diamond pattern should not cause ConflictingStateAssignmentError."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')

            # Create diamond: M depends on A, T depends on M (and thus A)
            M = ordered_product([A, B])  # M = A * B
            T = ordered_product([M])     # T = M = A * B

            # Result has M as direct parent and T (which contains M) as another parent
            # This is the diamond pattern that was causing conflicts
            result = ordered_product([M, T])

            # Should deduplicate to just A * B = 6 states
            assert result.num_states == 6

            # Should be able to iterate without ConflictingStateAssignmentError
            results = []
            for state in result:
                results.append((state, A.state, B.state))

            expected = [
                (0, 0, 0), (1, 1, 0), (2, 0, 1),
                (3, 1, 1), (4, 0, 2), (5, 1, 2),
            ]
            assert results == expected

    def test_ordered_product_does_not_flatten_non_product_ops(self):
        """Non-product operations like stack should NOT be flattened."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')

            # Create a stack (not a product)
            AB_stack = stack([A, B])  # stack has 5 states (2 + 3)

            # ordered_product should NOT flatten through the stack
            result = ordered_product([AB_stack, C])

            # Should be stack(A,B) * C = 5 * 4 = 20
            # NOT A * B * C = 24 (which would happen if we flattened through stack)
            assert result.num_states == 20

    def test_ordered_product_complex_diamond(self):
        """Complex diamond with multiple levels should work correctly."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')

            # Create: ((A*B)*C) * (A*C)
            AB = ordered_product([A, B])
            ABC = ordered_product([AB, C])
            AC = ordered_product([A, C])

            result = ordered_product([ABC, AC])

            # Should flatten to A * B * C = 24 (A and C deduplicated)
            assert result.num_states == 24

            # Should iterate without conflict
            count = 0
            for _ in result:
                count += 1
            assert count == 24

    def test_ordered_product_immediate_duplicate(self):
        """Immediate duplicates should still be deduplicated."""
        with Manager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')

            # Direct duplicate
            result = ordered_product([A, B, A])

            # Should deduplicate to A * B = 6
            assert result.num_states == 6
