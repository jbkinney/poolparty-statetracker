"""Tests for split_counter() function."""
import pytest
from poolparty.counter import Counter, CounterManager, split_counter


class TestSplitCounterBasic:
    """Test basic split_counter() functionality."""
    
    def test_split_into_two_equal(self):
        """Split 10 states into 2 equal parts."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            parts = split_counter(A, 2)
            
            assert len(parts) == 2
            assert parts[0].num_states == 5
            assert parts[1].num_states == 5
    
    def test_split_into_three_equal(self):
        """Split 9 states into 3 equal parts."""
        with CounterManager():
            A = Counter(num_states=9, name='A')
            parts = split_counter(A, 3)
            
            assert len(parts) == 3
            assert parts[0].num_states == 3
            assert parts[1].num_states == 3
            assert parts[2].num_states == 3
    
    def test_split_uneven(self):
        """Split 10 states into 3 parts (uneven)."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            parts = split_counter(A, 3)
            
            assert len(parts) == 3
            # Larger parts first: 4, 3, 3
            assert parts[0].num_states == 4
            assert parts[1].num_states == 3
            assert parts[2].num_states == 3
            # Total should match
            assert sum(p.num_states for p in parts) == 10
    
    def test_split_propagation(self):
        """Split counters propagate state to parent correctly."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            left, right = split_counter(A, 2)
            
            # Left covers states 0-4
            left.state = 0
            assert A.state == 0
            left.state = 4
            assert A.state == 4
            
            # Right covers states 5-9
            right.state = 0
            assert A.state == 5
            right.state = 4
            assert A.state == 9
    
    def test_split_iteration(self):
        """Iterate split counters and check parent states."""
        with CounterManager():
            A = Counter(num_states=6, name='A')
            parts = split_counter(A, 2)  # [0,1,2] and [3,4,5]
            
            # First part
            results = []
            for state in parts[0]:
                results.append((state, A.state))
            assert results == [(0, 0), (1, 1), (2, 2)]
            
            # Second part
            results = []
            for state in parts[1]:
                results.append((state, A.state))
            assert results == [(0, 3), (1, 4), (2, 5)]


class TestSplitCounterProportional:
    """Test proportional splitting."""
    
    def test_proportional_simple(self):
        """Split with simple proportions (1:1)."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            parts = split_counter(A, (1.0, 1.0))
            
            assert len(parts) == 2
            assert parts[0].num_states == 5
            assert parts[1].num_states == 5
    
    def test_proportional_1_2_1(self):
        """Split with proportions 1:2:1."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            parts = split_counter(A, (1.0, 2.0, 1.0))
            
            assert len(parts) == 3
            # Proportions: 25%, 50%, 25% of 10 = 2.5, 5, 2.5
            # After rounding and adjustment: should sum to 10
            assert sum(p.num_states for p in parts) == 10
            # Middle part should be largest
            assert parts[1].num_states >= parts[0].num_states
            assert parts[1].num_states >= parts[2].num_states
    
    def test_proportional_integers(self):
        """Proportions can be integers (treated as floats)."""
        with CounterManager():
            A = Counter(num_states=12, name='A')
            parts = split_counter(A, (1, 2, 1))
            
            assert len(parts) == 3
            assert parts[0].num_states == 3  # 25% of 12
            assert parts[1].num_states == 6  # 50% of 12
            assert parts[2].num_states == 3  # 25% of 12
    
    def test_proportional_propagation(self):
        """Proportionally split counters propagate correctly."""
        with CounterManager():
            A = Counter(num_states=12, name='A')
            parts = split_counter(A, (1, 2, 1))  # Sizes: 3, 6, 3
            
            # First part: states 0-2
            parts[0].state = 0
            assert A.state == 0
            parts[0].state = 2
            assert A.state == 2
            
            # Second part: states 3-8
            parts[1].state = 0
            assert A.state == 3
            parts[1].state = 5
            assert A.state == 8
            
            # Third part: states 9-11
            parts[2].state = 0
            assert A.state == 9
            parts[2].state = 2
            assert A.state == 11


class TestSplitCounterNames:
    """Test naming of split counters."""
    
    def test_with_names(self):
        """Split with custom names."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            left, right = split_counter(A, 2, names=['left', 'right'])
            
            assert left.name == 'left'
            assert right.name == 'right'
    
    def test_without_names(self):
        """Split without names (names are auto-assigned)."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            parts = split_counter(A, 2)
            
            # Auto-assigned names like 'id_1', 'id_2'
            assert parts[0].name is not None
            assert parts[1].name is not None
    
    def test_names_with_proportions(self):
        """Names work with proportional split."""
        with CounterManager():
            A = Counter(num_states=12, name='A')
            parts = split_counter(A, (1, 2, 1), names=['small1', 'big', 'small2'])
            
            assert parts[0].name == 'small1'
            assert parts[1].name == 'big'
            assert parts[2].name == 'small2'


class TestSplitCounterEdgeCases:
    """Test edge cases."""
    
    def test_minimum_split(self):
        """Split 2 states into 2 parts."""
        with CounterManager():
            A = Counter(num_states=2, name='A')
            parts = split_counter(A, 2)
            
            assert len(parts) == 2
            assert parts[0].num_states == 1
            assert parts[1].num_states == 1
    
    def test_many_parts(self):
        """Split into many parts."""
        with CounterManager():
            A = Counter(num_states=100, name='A')
            parts = split_counter(A, 10)
            
            assert len(parts) == 10
            assert all(p.num_states == 10 for p in parts)
    
    def test_inactive_state_propagation(self):
        """Split counters handle inactive state (None)."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            left, right = split_counter(A, 2)
            
            left.state = None
            assert A.state is None
            
            right.state = None
            assert A.state is None


class TestSplitCounterValidation:
    """Test validation and error handling."""
    
    def test_split_spec_one_raises(self):
        """split_spec=1 raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="must be >= 2"):
                split_counter(A, 1)
    
    def test_split_spec_zero_raises(self):
        """split_spec=0 raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="must be >= 2"):
                split_counter(A, 0)
    
    def test_split_spec_negative_raises(self):
        """Negative split_spec raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="must be >= 2"):
                split_counter(A, -1)
    
    def test_proportions_too_short_raises(self):
        """Proportions sequence with < 2 elements raises."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="must have length >= 2"):
                split_counter(A, (1.0,))
    
    def test_proportions_negative_raises(self):
        """Negative proportion raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="positive numbers"):
                split_counter(A, (1.0, -1.0))
    
    def test_proportions_zero_raises(self):
        """Zero proportion raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="positive numbers"):
                split_counter(A, (1.0, 0.0))
    
    def test_not_counter_raises(self):
        """Passing non-Counter raises TypeError."""
        with pytest.raises(TypeError, match="Expected Counter"):
            split_counter("not a counter", 2)
    
    def test_wrong_names_length_raises(self):
        """Names with wrong length raises ValueError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="names has length 3.*2 parts"):
                split_counter(A, 2, names=['a', 'b', 'c'])
    
    def test_too_few_states_for_parts_raises(self):
        """Splitting into more parts than states raises."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            with pytest.raises(ValueError, match="Cannot split 3 states into 5 parts"):
                split_counter(A, 5)
    
    def test_invalid_split_spec_type_raises(self):
        """Invalid split_spec type raises TypeError."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(TypeError, match="must be int or Sequence"):
                split_counter(A, "invalid")


class TestSplitCounterComposition:
    """Test composition with other operations."""
    
    def test_split_product(self):
        """Split a product counter."""
        with CounterManager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B  # 6 states
            # C decomposition: A = state % 2, B = (state // 2) % 3
            # C=0: A=0, B=0 | C=1: A=1, B=0 | C=2: A=0, B=1
            # C=3: A=1, B=1 | C=4: A=0, B=2 | C=5: A=1, B=2
            
            parts = split_counter(C, 2)
            
            assert len(parts) == 2
            assert parts[0].num_states == 3
            assert parts[1].num_states == 3
            
            # First part: C states 0, 1, 2
            parts[0].state = 0
            assert C.state == 0
            assert A.state == 0
            assert B.state == 0
            
            parts[0].state = 2
            assert C.state == 2
            assert A.state == 0
            assert B.state == 1  # C=2 -> A=0, B=1
            
            # Second part: C states 3, 4, 5
            parts[1].state = 0
            assert C.state == 3
            assert A.state == 1
            assert B.state == 1  # C=3 -> A=1, B=1
    
    def test_split_sum(self):
        """Split a sum counter."""
        with CounterManager():
            A = Counter(num_states=3, name='A')
            B = Counter(num_states=3, name='B')
            C = A + B  # 6 states
            
            parts = split_counter(C, 2)
            
            # First part covers C=0,1,2 (A active)
            parts[0].state = 1
            assert C.state == 1
            assert A.state == 1
            assert B.state is None
            
            # Second part covers C=3,4,5 (B active)
            parts[1].state = 1
            assert C.state == 4
            assert A.state is None
            assert B.state == 1
    
    def test_split_then_sum(self):
        """Sum split counters back together."""
        with CounterManager():
            A = Counter(num_states=10, name='A')
            left, right = split_counter(A, 2)
            
            # Sum them back
            combined = left + right
            assert combined.num_states == 10
