"""Tests for Manager class."""
import pytest
import pandas as pd
from statecounter import Counter, Manager, stack, product


class TestManager:
    """Test Manager context manager."""
    
    def test_context_manager_basic(self):
        """Manager works as context manager."""
        with Manager() as mgr:
            assert Manager._active_manager is mgr
        assert Manager._active_manager is None
    
    def test_auto_registration(self):
        """Counters created in context auto-register."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            assert len(mgr._counters) == 2
            assert A in mgr._counters
            assert B in mgr._counters
    
    def test_composite_counters_register(self):
        """Composite counters also register."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            # A, B, and C should all be registered
            assert len(mgr._counters) == 3
    
    def test_counter_requires_manager(self):
        """Counters created outside context raise error."""
        with pytest.raises(RuntimeError, match="must be created within a Manager context"):
            Counter(num_states=2, name='A')
    
    def test_get_counter_names(self):
        """get_counter_names returns list of counter names."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            names = mgr.get_all_names()
            assert names == ['A', 'B', 'C']
    
    def test_get_counter_names_with_auto_name(self):
        """Unnamed counters get auto-generated names like id_N."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3)  # No name -> auto-named 'Counter[1]'
            
            names = mgr.get_all_names()
            assert names == ['A', 'Counter[1]']
    
    def test_get_counter_by_name(self):
        """get_counter returns counter by name."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            
            assert mgr.get_by_name('A') is A
            assert mgr.get_by_name('B') is B
    
    def test_get_counter_not_found(self):
        """get_counter raises KeyError for unknown name."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            
            with pytest.raises(KeyError, match="No counter with name 'X' found"):
                mgr.get_by_name('X')
    
    def test_reset_counters_all(self):
        """reset_counters resets all counters when no arg given."""
        with Manager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 3
            B.state = 4
            
            mgr.reset_all()
            
            assert A.state == 0
            assert B.state == 0
    
    def test_reset_counters_specific(self):
        """reset_counters resets only specified counters."""
        with Manager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 3
            B.state = 4
            
            mgr.reset_all([A])
            
            assert A.state == 0
            assert B.state == 4  # unchanged
    
    def test_inactivate_counters_all(self):
        """inactivate_counters inactivates all counters when no arg given."""
        with Manager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 2
            B.state = 3
            
            mgr.inactivate_all()
            
            assert A.state is None
            assert B.state is None
    
    def test_inactivate_counters_specific(self):
        """inactivate_counters inactivates only specified counters."""
        with Manager() as mgr:
            A = Counter(num_states=5, name='A')
            B = Counter(num_states=5, name='B')
            
            A.state = 2
            B.state = 3
            
            mgr.inactivate_all([A])
            
            assert A.state is None
            assert B.state == 3  # unchanged
    
    def test_get_iteration_df_basic(self):
        """get_iteration_df returns DataFrame of states."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            df = mgr.get_iteration_df(C)
            
            assert isinstance(df, pd.DataFrame)
            assert df.index.name == 'C'
            assert list(df.columns) == ['A', 'B']
            assert len(df) == 6
    
    def test_get_iteration_df_values(self):
        """get_iteration_df has correct state values."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            df = mgr.get_iteration_df(C)
            
            # Check expected values for product counter
            # C=0: A=0, B=0; C=1: A=1, B=0; ...
            assert df['A'].tolist() == [0, 1, 0, 1, 0, 1]
            assert df['B'].tolist() == [0, 0, 1, 1, 2, 2]
            assert df.index.tolist() == [0, 1, 2, 3, 4, 5]
    
    def test_get_iteration_df_specific_counters(self):
        """get_iteration_df with specific counters only shows those."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            C.name = 'C'
            
            df = mgr.get_iteration_df(C, counters=[A, C])
            
            assert list(df.columns) == ['A']
            assert len(df) == 6
    
    def test_get_iteration_df_stack_shows_inactive(self):
        """get_iteration_df shows NaN for inactive counters in sum."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = stack([A,B])
            C.name = 'C'
            
            df = mgr.get_iteration_df(C)
            
            # First 2 rows: A active, B inactive (NaN in DataFrame)
            assert df['A'].iloc[0] == 0
            assert pd.isna(df['B'].iloc[0])
            assert df['A'].iloc[1] == 1
            assert pd.isna(df['B'].iloc[1])
            
            # Last 3 rows: A inactive (NaN in DataFrame), B active
            assert pd.isna(df['A'].iloc[2])
            assert df['B'].iloc[2] == 0
            assert pd.isna(df['A'].iloc[4])
            assert df['B'].iloc[4] == 2


class TestCounterIdAssignment:
    """Test automatic ID assignment and auto-naming."""
    
    def test_sequential_ids(self):
        """Counters get sequential IDs starting from 0."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = Counter(num_states=4, name='C')
            
            assert A.id == 0
            assert B.id == 1
            assert C.id == 2
    
    def test_composite_counters_get_ids(self):
        """Composite counters also get IDs."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = product([A, B])
            
            assert A.id == 0
            assert B.id == 1
            assert C.id == 2
    
    def test_auto_name_for_unnamed_counters(self):
        """Unnamed counters get auto-generated names like id_N."""
        with Manager() as mgr:
            A = Counter(num_states=2)  # No name
            B = Counter(num_states=3)  # No name
            C = Counter(num_states=4, name='C')  # Has name
            
            assert A.name == 'Counter[0]'
            assert B.name == 'Counter[1]'
            assert C.name == 'C'  # Keeps original name
    
    def test_named_counters_keep_name(self):
        """Named counters keep their original name."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='MyCounter')
            
            assert A.id == 0
            assert A.name == 'MyCounter'
    
    def test_id_property_readonly(self):
        """ID property is read-only."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            assert A.id == 0
            # Cannot set id via property (no setter)
            with pytest.raises(AttributeError):
                A.id = 5
    
    def test_separate_managers_have_independent_ids(self):
        """Different Manager instances have independent ID sequences."""
        with Manager() as mgr1:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            assert A.id == 0
            assert B.id == 1
        
        with Manager() as mgr2:
            C = Counter(num_states=4, name='C')
            D = Counter(num_states=5, name='D')
            assert C.id == 0  # Starts fresh
            assert D.id == 1


class TestDAGSupport:
    """Test that computation graphs support DAGs (same counter reachable via multiple paths)."""
    
    def test_same_counter_in_stack_allowed(self):
        """stack([A,A]) is allowed - creates a DAG with A reachable twice."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = stack([A,A])  # Should not raise - DAGs are allowed, sum preserves duplicates
            assert B.num_states == 4  # Sum of 2 + 2
    
    def test_same_counter_in_product_deduplicated(self):
        """product([A, A]) is deduplicated - duplicate parents are removed."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = product([A, A])  # Deduplicated to just (A,)
            assert B.num_states == 2  # Just A after dedup
    
    def test_same_counter_nested_allowed(self):
        """C = stack([A,B]) where B = product([A, X]) is allowed - A reachable via two paths."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            X = Counter(num_states=3, name='X')
            B = product([A, X])
            B.name = 'B'
            C = stack([A,B])  # A and B are different objects, both kept
            assert C.num_states == 2 + 6  # A has 2, B has 2*3=6
    
    def test_distinct_counters_work(self):
        """C = stack([A,B]) with distinct A, B works fine."""
        with Manager() as mgr:
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = stack([A,B])  # Should not raise
            assert C.num_states == 5
    
    def test_shared_counter_across_pools_allowed(self):
        """Multiple pools sharing the same counter can be combined."""
        with Manager() as mgr:
            shared = Counter(num_states=3, name='shared')
            # Simulate two pools sharing the same counter - sum preserves duplicates
            sum_counter = stack([shared, shared])
            assert sum_counter.num_states == 6  # 3 + 3
