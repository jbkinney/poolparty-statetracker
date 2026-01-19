"""Tests for ProductOp and product_states()."""
import pytest
from statetracker import (
    State, Manager, ProductOp, product, ordered_product, stack,
    set_product_order_mode, get_product_order_mode,
)

class TestProductOperation:
    """Test product (multiplication) operation."""
    
    def test_product_num_states(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            assert C.num_values == 6
    
    def test_product_initial_state(self):
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            assert A.value == 0  # Leaf state defaults to 0
            assert B.value == 0  # Leaf state defaults to 0
            assert C.value is None  # Derived state defaults to inactive
    
    def test_product_state_propagation_down(self):
        """Setting product state propagates to parents."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])
            
            # Set C=5 -> A=5%2=1, B=5//2=2
            C.value = 5
            assert A.value == 1
            assert B.value == 2
    
    def test_product_iteration(self):
        """Iterate product and check parent states."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
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
                results.append((c_state, A.value, B.value))
            
            assert results == expected


class TestProductOp:
    """Test ProductOp class directly."""
    
    def test_product_co_op(self):
        op = ProductOp()
        assert op.compute_num_states((2, 3)) == 6
        assert op.decompose(5, (2, 3)) == (1, 2)
    
    def test_product_co_op_n_ary(self):
        """Test ProductOp with 3+ states."""
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


class TestMultiplyStates:
    """Test product_states() function for N-ary products."""
    
    def test_product_three_states(self):
        """product_states(A, B, C) creates 3-way product."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')
            
            P = product([A, B, C])
            assert P.num_values == 24
    
    def test_product_iteration_three_states(self):
        """Iterate 3-way product and check parent states."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=2, name='C')
            
            P = product([A, B, C])
            
            results = []
            for p_state in P:
                results.append((p_state, A.value, B.value, C.value))
            
            # All 12 combinations with A cycling fastest, then B, then C
            expected = [
                (0, 0, 0, 0), (1, 1, 0, 0), (2, 0, 1, 0), (3, 1, 1, 0),
                (4, 0, 2, 0), (5, 1, 2, 0), (6, 0, 0, 1), (7, 1, 0, 1),
                (8, 0, 1, 1), (9, 1, 1, 1), (10, 0, 2, 1), (11, 1, 2, 1),
            ]
            assert results == expected
    
    def test_product_four_states(self):
        """product_states works with 4 states."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=2, name='B')
            C = State(num_values=2, name='C')
            D = State(num_values=2, name='D')
            
            P = product([A, B, C, D])
            assert P.num_values == 16
    
    def test_product_accepts_zero_states(self):
        """product_states with 0 states returns State(1)."""
        with Manager():
            P = product([])
            assert P.num_values == 1
    
    def test_product_accepts_one_state(self):
        """product_states with 1 state returns linked state."""
        with Manager():
            A = State(num_values=3, name='A')
            P = product([A])
            assert P.num_values == 3
            P.value = 2
            assert A.value == 2
    
    def test_product_requires_states(self):
        """product_states raises for non-State arguments."""
        with Manager():
            A = State(num_values=2, name='A')
            with pytest.raises(Exception):
                product([A, 123], "not a state")
    
    def test_product_with_name(self):
        """product_states with name parameter."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            P = product([A, B], name='Product')
            assert P.name == 'Product'


class TestOrderedProductFlattening:
    """Test ordered_product recursive flattening and deduplication."""

    def test_ordered_product_flattens_nested_products(self):
        """ordered_product([A*B, C]) should flatten to ordered_product([A, B, C])."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')

            # Create nested product
            AB = ordered_product([A, B])
            ABC_nested = ordered_product([AB, C])

            # Create flat product
            ABC_flat = ordered_product([A, B, C])

            # Both should have same num_states
            assert ABC_nested.num_values == ABC_flat.num_values == 24

            # Both should iterate the same way
            nested_results = []
            for state in ABC_nested:
                nested_results.append((state, A.value, B.value, C.value))

            flat_results = []
            for state in ABC_flat:
                flat_results.append((state, A.value, B.value, C.value))

            assert nested_results == flat_results

    def test_ordered_product_deduplicates_through_nested_products(self):
        """ordered_product([A*B, C, D*A]) should deduplicate A."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')
            D = State(num_values=5, name='D')

            AB = ordered_product([A, B])
            DA = ordered_product([D, A])

            # Without flattening, this would have A twice
            # With flattening, A should be deduplicated
            result = ordered_product([AB, C, DA])

            # Should be A * B * C * D = 2 * 3 * 4 * 5 = 120
            # NOT (A * B) * C * (D * A) = 6 * 4 * 10 = 240
            assert result.num_values == 120

    def test_ordered_product_diamond_pattern_no_conflict(self):
        """Diamond pattern should not cause ConflictingValueAssignmentError."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')

            # Create diamond: M depends on A, T depends on M (and thus A)
            M = ordered_product([A, B])  # M = A * B
            T = ordered_product([M])     # T = M = A * B

            # Result has M as direct parent and T (which contains M) as another parent
            # This is the diamond pattern that was causing conflicts
            result = ordered_product([M, T])

            # Should deduplicate to just A * B = 6 states
            assert result.num_values == 6

            # Should be able to iterate without ConflictingValueAssignmentError
            results = []
            for state in result:
                results.append((state, A.value, B.value))

            # With first_state_slowest default, A (lower id) cycles slowest, B cycles fastest
            expected = [
                (0, 0, 0), (1, 0, 1), (2, 0, 2),
                (3, 1, 0), (4, 1, 1), (5, 1, 2),
            ]
            assert results == expected

    def test_ordered_product_does_not_flatten_non_product_ops(self):
        """Non-product operations like stack should NOT be flattened."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')

            # Create a stack (not a product)
            AB_stack = stack([A, B])  # stack has 5 states (2 + 3)

            # ordered_product should NOT flatten through the stack
            result = ordered_product([AB_stack, C])

            # Should be stack(A,B) * C = 5 * 4 = 20
            # NOT A * B * C = 24 (which would happen if we flattened through stack)
            assert result.num_values == 20

    def test_ordered_product_complex_diamond(self):
        """Complex diamond with multiple levels should work correctly."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = State(num_values=4, name='C')

            # Create: ((A*B)*C) * (A*C)
            AB = ordered_product([A, B])
            ABC = ordered_product([AB, C])
            AC = ordered_product([A, C])

            result = ordered_product([ABC, AC])

            # Should flatten to A * B * C = 24 (A and C deduplicated)
            assert result.num_values == 24

            # Should iterate without conflict
            count = 0
            for _ in result:
                count += 1
            assert count == 24

    def test_ordered_product_immediate_duplicate(self):
        """Immediate duplicates should still be deduplicated."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')

            # Direct duplicate
            result = ordered_product([A, B, A])

            # Should deduplicate to A * B = 6
            assert result.num_values == 6


class TestProductOrderMode:
    """Test product order mode (first_state_fastest vs first_state_slowest)."""
    
    def test_get_set_product_order_mode(self):
        """Test getter/setter for product order mode."""
        original = get_product_order_mode()
        try:
            set_product_order_mode('first_state_slowest')
            assert get_product_order_mode() == 'first_state_slowest'
            set_product_order_mode('first_state_fastest')
            assert get_product_order_mode() == 'first_state_fastest'
        finally:
            set_product_order_mode(original)
    
    def test_invalid_mode_raises(self):
        """Invalid mode value should raise ValueError."""
        with pytest.raises(ValueError):
            set_product_order_mode('invalid_mode')
    
    def test_first_state_fastest_ordering(self):
        """Default mode: lower ID states cycle fastest."""
        original = get_product_order_mode()
        try:
            set_product_order_mode('first_state_fastest')
            with Manager():
                A = State(num_values=2, name='A')  # id=0
                B = State(num_values=3, name='B')  # id=1
                P = ordered_product([A, B])
                
                results = []
                for _ in P:
                    results.append((A.value, B.value))
                
                # A (lower id) cycles fastest: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
                expected = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
                assert results == expected
        finally:
            set_product_order_mode(original)
    
    def test_first_state_slowest_ordering(self):
        """first_state_slowest mode: lower ID states cycle slowest."""
        original = get_product_order_mode()
        try:
            set_product_order_mode('first_state_slowest')
            with Manager():
                A = State(num_values=2, name='A')  # id=0
                B = State(num_values=3, name='B')  # id=1
                P = ordered_product([A, B])
                
                results = []
                for _ in P:
                    results.append((A.value, B.value))
                
                # B (higher id, lower -id) cycles fastest: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
                expected = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
                assert results == expected
        finally:
            set_product_order_mode(original)
    
    def test_mode_affects_only_ordered_product(self):
        """Mode only affects ordered_product, not product()."""
        original = get_product_order_mode()
        try:
            set_product_order_mode('first_state_slowest')
            with Manager():
                A = State(num_values=2, name='A')
                B = State(num_values=3, name='B')
                # product() uses explicit order, not sorted by id
                P = product([A, B])
                
                results = []
                for _ in P:
                    results.append((A.value, B.value))
                
                # A cycles fastest because it's first in the list
                expected = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
                assert results == expected
        finally:
            set_product_order_mode(original)
