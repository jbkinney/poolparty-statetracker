"""Tests for split_state() function."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, split


class TestSplitStateBasic:
    """Test basic split_state() functionality."""
    
    def test_split_into_two_equal(self):
        """Split 10 states into 2 equal parts."""
        with Manager():
            A = State(num_values=10, name='A')
            parts = split(A, 2)
            
            assert len(parts) == 2
            assert parts[0].num_values == 5
            assert parts[1].num_values == 5
    
    def test_split_into_three_equal(self):
        """Split 9 states into 3 equal parts."""
        with Manager():
            A = State(num_values=9, name='A')
            parts = split(A, 3)
            
            assert len(parts) == 3
            assert parts[0].num_values == 3
            assert parts[1].num_values == 3
            assert parts[2].num_values == 3
    
    def test_split_uneven(self):
        """Split 10 states into 3 parts (uneven)."""
        with Manager():
            A = State(num_values=10, name='A')
            parts = split(A, 3)
            
            assert len(parts) == 3
            assert parts[0].num_values == 4
            assert parts[1].num_values == 3
            assert parts[2].num_values == 3
            assert sum(p.num_values for p in parts) == 10
    
    def test_split_propagation(self):
        """Split states propagate state to parent correctly."""
        with Manager():
            A = State(num_values=10, name='A')
            left, right = split(A, 2)
            
            left.value = 0
            assert A.value == 0
            left.value = 4
            assert A.value == 4
            
            right.value = 0
            assert A.value == 5
            right.value = 4
            assert A.value == 9


class TestSplitStateProportional:
    """Test proportional splitting."""
    
    def test_proportional_simple(self):
        """Split with simple proportions (1:1)."""
        with Manager():
            A = State(num_values=10, name='A')
            parts = split(A, (1.0, 1.0))
            
            assert len(parts) == 2
            assert parts[0].num_values == 5
            assert parts[1].num_values == 5
    
    def test_proportional_integers(self):
        """Proportions can be integers (treated as floats)."""
        with Manager():
            A = State(num_values=12, name='A')
            parts = split(A, (1, 2, 1))
            
            assert len(parts) == 3
            assert parts[0].num_values == 3
            assert parts[1].num_values == 6
            assert parts[2].num_values == 3


class TestSplitStateNames:
    """Test naming of split states."""
    
    def test_with_names(self):
        """Split with custom names."""
        with Manager():
            A = State(num_values=10, name='A')
            left, right = split(A, 2, names=['left', 'right'])
            
            assert left.name == 'left'
            assert right.name == 'right'


class TestSplitStateValidation:
    """Test validation and error handling."""
    
    def test_split_spec_one_raises(self):
        """split_spec=1 raises ValueError."""
        with Manager():
            A = State(num_values=10, name='A')
            with pytest.raises(ValueError, match="must be >= 2"):
                split(A, 1)
    
    def test_not_state_raises(self):
        """Passing non-State raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            split("not a state", 2)
    
    def test_too_few_states_for_parts_raises(self):
        """Splitting into more parts than states raises."""
        with Manager():
            A = State(num_values=3, name='A')
            with pytest.raises(ValueError, match="Cannot split 3 values into 5 parts"):
                split(A, 5)
