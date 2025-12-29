"""Tests for CounterManager class."""
import pytest
import pandas as pd
from poolparty.counter import Counter, CounterManager


class TestCounterManager:
    """Test CounterManager context manager."""
    
    def test_context_manager_basic(self):
        """CounterManager works as context manager."""
        with CounterManager() as mgr:
            assert CounterManager._active_manager is mgr
        assert CounterManager._active_manager is None
    
    def test_auto_registration(self):
        """Counters created in context auto-register."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            assert len(mgr._counters) == 2
            assert A in mgr._counters
            assert B in mgr._counters
    
    def test_composite_counters_register(self):
        """Composite counters also register."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            C.name = 'C'
            # A, B, and C should all be registered
            assert len(mgr._counters) == 3
    
    def test_no_registration_outside_context(self):
        """Counters created outside context don't register."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
        
        # Counter created outside
        B = Counter(num_states=3, name='B')
        
        # B should not be in mgr
        assert B not in mgr._counters
        assert len(mgr._counters) == 1
    
    def test_get_counter_names(self):
        """get_counter_names returns list of counter names."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            C.name = 'C'
            
            names = mgr.get_all_names()
            assert names == ['A', 'B', 'C']
    
    def test_get_counter_names_with_auto_name(self):
        """Unnamed counters get auto-generated names like id_N."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3)  # No name -> auto-named 'id_1'
            
            names = mgr.get_all_names()
            assert names == ['A', 'id_1']
    
    def test_get_counter_by_name(self):
        """get_counter returns counter by name."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            
            assert mgr.get_by_name('A') is A
            assert mgr.get_by_name('B') is B
    
    def test_get_counter_not_found(self):
        """get_counter raises KeyError for unknown name."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            
            with pytest.raises(KeyError, match="No counter with name 'X' found"):
                mgr.get_by_name('X')
    
    def test_reset_counters_all(self):
        """reset_counters resets all counters when no arg given."""
        with CounterManager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 3
            B.state = 4
            
            mgr.reset_all()
            
            assert A.state == 0
            assert B.state == 0
    
    def test_reset_counters_specific(self):
        """reset_counters resets only specified counters."""
        with CounterManager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 3
            B.state = 4
            
            mgr.reset_all([A])
            
            assert A.state == 0
            assert B.state == 4  # unchanged
    
    def test_inactivate_counters_all(self):
        """inactivate_counters inactivates all counters when no arg given."""
        with CounterManager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 2
            B.state = 3
            
            mgr.inactivate_all()
            
            assert A.state == -1
            assert B.state == -1
    
    def test_inactivate_counters_specific(self):
        """inactivate_counters inactivates only specified counters."""
        with CounterManager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 2
            B.state = 3
            
            mgr.inactivate_all([A])
            
            assert A.state == -1
            assert B.state == 3  # unchanged
    
    def test_get_iteration_df_basic(self):
        """get_iteration_df returns DataFrame of states."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            C.name = 'C'
            
            df = mgr.test_iteration(C)
            
            assert isinstance(df, pd.DataFrame)
            assert df.index.name == 'C.state'
            assert list(df.columns) == ['A.state', 'B.state', 'C.state']
            assert len(df) == 6
    
    def test_get_iteration_df_values(self):
        """get_iteration_df has correct state values."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            C.name = 'C'
            
            df = mgr.test_iteration(C)
            
            # Check expected values for product counter
            # C=0: A=0, B=0; C=1: A=1, B=0; ...
            assert df['A.state'].tolist() == [0, 1, 0, 1, 0, 1]
            assert df['B.state'].tolist() == [0, 0, 1, 1, 2, 2]
            assert df['C.state'].tolist() == [0, 1, 2, 3, 4, 5]
    
    def test_get_iteration_df_specific_counters(self):
        """get_iteration_df with specific counters only shows those."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            C.name = 'C'
            
            df = mgr.test_iteration(C, counters=[A, C])
            
            assert list(df.columns) == ['A.state', 'C.state']
            assert len(df) == 6
    
    def test_get_iteration_df_sum_shows_inactive(self):
        """get_iteration_df shows -1 for inactive counters in sum."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A + B
            C.name = 'C'
            
            df = mgr.test_iteration(C)
            
            # First 2 rows: A active, B inactive (-1)
            assert df['A.state'].iloc[0] == 0
            assert df['B.state'].iloc[0] == -1
            assert df['A.state'].iloc[1] == 1
            assert df['B.state'].iloc[1] == -1
            
            # Last 3 rows: A inactive (-1), B active
            assert df['A.state'].iloc[2] == -1
            assert df['B.state'].iloc[2] == 0
            assert df['A.state'].iloc[4] == -1
            assert df['B.state'].iloc[4] == 2


class TestCounterIdAssignment:
    """Test automatic ID assignment and auto-naming."""
    
    def test_sequential_ids(self):
        """Counters get sequential IDs starting from 0."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            
            assert A.id == 0
            assert B.id == 1
            assert C.id == 2
    
    def test_composite_counters_get_ids(self):
        """Composite counters also get IDs."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B
            
            assert A.id == 0
            assert B.id == 1
            assert C.id == 2
    
    def test_auto_name_for_unnamed_counters(self):
        """Unnamed counters get auto-generated names like id_N."""
        with CounterManager() as mgr:
            A = Counter(num_states=2)  # No name
            B = Counter(num_states=3)  # No name
            C = Counter(num_states=4, name='C')  # Has name
            
            assert A.name == 'id_0'
            assert B.name == 'id_1'
            assert C.name == 'C'  # Keeps original name
    
    def test_named_counters_keep_name(self):
        """Named counters keep their original name."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='MyCounter')
            
            assert A.id == 0
            assert A.name == 'MyCounter'
    
    def test_id_none_outside_manager(self):
        """Counters created outside manager have id=None."""
        A = Counter(num_states=2, name='A')
        assert A.id is None
    
    def test_name_none_outside_manager(self):
        """Counters without name and outside manager have name=None."""
        A = Counter(num_states=2)
        assert A.id is None
        assert A.name is None
    
    def test_id_property_readonly(self):
        """ID property is read-only."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            assert A.id == 0
            # Cannot set id via property (no setter)
            with pytest.raises(AttributeError):
                A.id = 5
    
    def test_separate_managers_have_independent_ids(self):
        """Different CounterManager instances have independent ID sequences."""
        with CounterManager() as mgr1:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            assert A.id == 0
            assert B.id == 1
        
        with CounterManager() as mgr2:
            C = Counter(num_states=4, name='C')
            D = Counter(num_states=5, name='D')
            assert C.id == 0  # Starts fresh
            assert D.id == 1


class TestTreeValidation:
    """Test that computation graphs must be trees (no repeated counters)."""
    
    def test_same_counter_in_sum_raises(self):
        """A + A raises ValueError because A appears twice."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="Counter 'A' appears 2 times"):
                B = A + A
    
    def test_same_counter_in_product_raises(self):
        """A * A raises ValueError because A appears twice."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            with pytest.raises(ValueError, match="Counter 'A' appears 2 times"):
                B = A * A
    
    def test_same_counter_nested_raises(self):
        """C = A + B where B = A * X raises ValueError because A appears twice."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            X = Counter(num_states=3, name='X')
            B = A * X
            B.name = 'B'
            with pytest.raises(ValueError, match="Counter 'A' appears 2 times"):
                C = A + B
    
    def test_distinct_counters_work(self):
        """C = A + B with distinct A, B works fine."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A + B  # Should not raise
            assert C.num_states == 5
    
    def test_error_message_includes_counter_names(self):
        """Error message includes both the duplicate counter and the graph counter."""
        with CounterManager() as mgr:
            A = Counter(num_states=2, name='MyCounter')
            with pytest.raises(ValueError) as exc_info:
                B = A + A
            assert "MyCounter" in str(exc_info.value)
            assert "computation graph" in str(exc_info.value)

