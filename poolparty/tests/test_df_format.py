"""Tests for DataFrame formatting utilities."""

import pytest
import pandas as pd
import poolparty as pp
from poolparty.df_format import (
    counter_col_name,
    get_pools_reverse_topo,
    organize_columns,
    finalize_generate_df,
)


class TestCounterColName:
    """Test counter_col_name function."""
    
    def test_uses_counter_name_if_present(self):
        """Test that counter name is used when available."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            pool.counter.name = "my_counter"
            
            result = counter_col_name(pool.counter, 0)
            assert result == "my_counter"
    
    def test_uses_counter_id_if_no_name(self):
        """Test fallback to id_{counter.id} when no name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            pool.counter.name = ""  # Clear name
            
            result = counter_col_name(pool.counter, 0)
            assert result == f"id_{pool.counter.id}"
    
    def test_uses_index_fallback(self):
        """Test fallback to id_{index} when no name or id."""
        # Create a mock counter with no name and no id
        class MockCounter:
            name = ""
            id = None
        
        mock_counter = MockCounter()
        result = counter_col_name(mock_counter, 5)
        assert result == "id_5"


class TestGetPoolsReverseTopo:
    """Test get_pools_reverse_topo function."""
    
    def test_single_pool(self):
        """Test with a single pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            result = get_pools_reverse_topo({pool})
            
            assert len(result) == 1
            assert result[0] is pool
    
    def test_parent_child_order(self):
        """Test that children come before parents."""
        with pp.Party() as party:
            parent = pp.from_seqs(['AAA', 'TTT'])
            child = pp.mutation_scan(parent, num_mutations=1)
            
            result = get_pools_reverse_topo({parent, child})
            
            # Child should come before parent in reverse topo order
            child_idx = result.index(child)
            parent_idx = result.index(parent)
            assert child_idx < parent_idx
    
    def test_chain_of_pools(self):
        """Test with a chain of operations."""
        with pp.Party() as party:
            p1 = pp.from_seqs(['AAA'])
            p2 = pp.mutation_scan(p1, num_mutations=1)
            p3 = pp.mutation_scan(p2, num_mutations=1)
            
            result = get_pools_reverse_topo({p1, p2, p3})
            
            # Order should be: p3, p2, p1 (most derived first)
            assert result.index(p3) < result.index(p2)
            assert result.index(p2) < result.index(p1)


class TestOrganizeColumns:
    """Test organize_columns function."""
    
    def test_organize_by_type(self):
        """Test organizing columns by type."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            pool.name = "test"
            
            df = pd.DataFrame({
                'test.seq': ['AAA', 'TTT'],
                'test.state': [0, 1],
                'extra': [1, 2],
            })
            
            result = organize_columns(df, {pool}, 'type')
            
            # seq columns should come first
            cols = list(result.columns)
            assert cols[0] == 'test.seq'
    
    def test_organize_by_pool(self):
        """Test organizing columns by pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            pool.name = "test"
            pool.operation.name = "test.op"
            
            df = pd.DataFrame({
                'test.seq': ['AAA', 'TTT'],
                'test.state': [0, 1],
                'extra': [1, 2],
            })
            
            result = organize_columns(df, {pool}, 'pool')
            
            # Columns should be grouped by pool
            cols = list(result.columns)
            # test.seq and test.state should be grouped together
            seq_idx = cols.index('test.seq')
            state_idx = cols.index('test.state')
            assert abs(seq_idx - state_idx) == 1
    
    def test_remaining_columns_appended(self):
        """Test that columns not matching any pool are appended at end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            pool.name = "test"
            
            df = pd.DataFrame({
                'test.seq': ['AAA'],
                'unrelated': [42],
            })
            
            result = organize_columns(df, {pool}, 'type')
            
            cols = list(result.columns)
            assert 'unrelated' in cols
            # 'unrelated' should be at the end
            assert cols[-1] == 'unrelated'


class TestFinalizeGenerateDf:
    """Test finalize_generate_df function."""
    
    def test_adds_seq_column_when_report_seq_true(self):
        """Test that 'seq' column is added when report_seq=True."""
        df = pd.DataFrame({
            'mypool.seq': ['AAA', 'TTT'],
            'other': [1, 2],
        })
        
        result = finalize_generate_df(df, 'mypool', report_seq=True, report_pool_seqs=True)
        
        assert 'seq' in result.columns
        assert list(result.columns)[0] == 'seq'
        assert list(result['seq']) == ['AAA', 'TTT']
    
    def test_no_seq_column_when_report_seq_false(self):
        """Test that 'seq' column is not added when report_seq=False."""
        df = pd.DataFrame({
            'mypool.seq': ['AAA', 'TTT'],
        })
        
        result = finalize_generate_df(df, 'mypool', report_seq=False, report_pool_seqs=True)
        
        assert 'seq' not in result.columns
    
    def test_drops_pool_seqs_when_report_pool_seqs_false(self):
        """Test that pool seq columns are dropped when report_pool_seqs=False."""
        df = pd.DataFrame({
            'seq': ['AAA', 'TTT'],
            'pool1.seq': ['AAA', 'TTT'],
            'pool2.seq': ['GGG', 'CCC'],
            'other': [1, 2],
        })
        
        result = finalize_generate_df(df, 'pool1', report_seq=False, report_pool_seqs=False)
        
        # All .seq columns should be dropped
        assert 'pool1.seq' not in result.columns
        assert 'pool2.seq' not in result.columns
        assert 'other' in result.columns
    
    def test_keeps_pool_seqs_when_report_pool_seqs_true(self):
        """Test that pool seq columns are kept when report_pool_seqs=True."""
        df = pd.DataFrame({
            'pool1.seq': ['AAA', 'TTT'],
            'other': [1, 2],
        })
        
        result = finalize_generate_df(df, 'pool1', report_seq=False, report_pool_seqs=True)
        
        assert 'pool1.seq' in result.columns


class TestIntegrationWithGenerate:
    """Test that df_format functions work correctly with Pool.generate_seqs()."""
    
    def test_generate_uses_df_format_functions(self):
        """Test that generate() produces correctly formatted output."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
            pool.name = "test_pool"
            
            df = pool.generate_seqs(num_complete_iterations=1)
            
            # Should have 'seq' as first column
            assert list(df.columns)[0] == 'seq'
            # Should have the sequences
            assert list(df['seq']) == ['AAA', 'TTT', 'GGG']
    
    def test_generate_with_organize_by_pool(self):
        """Test generate with organize_columns_by='pool'."""
        with pp.Party() as party:
            parent = pp.from_seqs(['AAA', 'TTT']).named('parent')
            child = pp.mutation_scan(parent, num_mutations=1).named('child')
            
            df = child.generate_seqs(
                num_seqs=5,
                organize_columns_by='pool',
            )
            
            # Should have properly organized columns
            assert 'seq' in df.columns
            assert 'child.seq' in df.columns

