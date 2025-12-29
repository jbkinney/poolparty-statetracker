"""Tests for split_counter() function."""
import pytest
from statecounter import Counter, Manager, split


class TestSplitCounterBasic:
    """Test basic split_counter() functionality."""
    
    def test_split_into_two_equal(self):
        """Split 10 states into 2 equal parts."""
        with Manager():
            A = Counter(num_states=10, name='A')
            parts = split(A, 2)
            
            assert len(parts) == 2
            assert parts[0].num_states == 5
            assert parts[1].num_states == 5
    
    def test_split_into_three_equal(self):
        """Split 9 states into 3 equal parts."""
        with Manager():
            A = Counter(num_states=9, name='A')
            parts = split(A, 3)
            
            assert len(parts) == 3
            assert parts[0].num_states == 3
            assert parts[1].num_states == 3
            assert parts[2].num_states == 3
    
    def test_split_uneven(self):
        """Split 10 states into 3 parts (uneven)."""
        with Manager():
            A = Counter(num_states=10, name='A')
            parts = split(A, 3)
            
            assert len(parts) == 3
            assert parts[0].num_states == 4
            assert parts[1].num_states == 3
            assert parts[2].num_states == 3
            assert sum(p.num_states for p in parts) == 10
    
    def test_split_propagation(self):
        """Split counters propagate state to parent correctly."""
        with Manager():
            A = Counter(num_states=10, name='A')
            left, right = split(A, 2)
            
            left.state = 0
            assert A.state == 0
            left.state = 4
            assert A.state == 4
            
            right.state = 0
            assert A.state == 5
            right.state = 4
            assert A.state == 9


class TestSplitCounterProportional:
    """Test proportional splitting."""
    
    def test_proportional_simple(self):
        """Split with simple proportions (1:1)."""
        with Manager():
            A = Counter(num_states=10, name='A')
            parts = split(A, (1.0, 1.0))
            
            assert len(parts) == 2
            assert parts[0].num_states == 5
            assert parts[1].num_states == 5
    
    def test_proportional_integers(self):
        """Proportions can be integers (treated as floats)."""
        with Manager():
            A = Counter(num_states=12, name='A')
            parts = split(A, (1, 2, 1))
            
            assert len(parts) == 3
            assert parts[0].num_states == 3
            assert parts[1].num_states == 6
            assert parts[2].num_states == 3


class TestSplitCounterNames:
    """Test naming of split counters."""
    
    def test_with_names(self):
        """Split with custom names."""
        with Manager():
            A = Counter(num_states=10, name='A')
            left, right = split(A, 2, names=['left', 'right'])
            
            assert left.name == 'left'
            assert right.name == 'right'


class TestSplitCounterValidation:
    """Test validation and error handling."""
    
    def test_split_spec_one_raises(self):
        """split_spec=1 raises ValueError."""
        with Manager():
            A = Counter(num_states=10, name='A')
            with pytest.raises(ValueError, match="must be >= 2"):
                split(A, 1)
    
    def test_not_counter_raises(self):
        """Passing non-Counter raises TypeError."""
        with pytest.raises(TypeError, match="Expected Counter"):
            split("not a counter", 2)
    
    def test_too_few_states_for_parts_raises(self):
        """Splitting into more parts than states raises."""
        with Manager():
            A = Counter(num_states=3, name='A')
            with pytest.raises(ValueError, match="Cannot split 3 states into 5 parts"):
                split(A, 5)
