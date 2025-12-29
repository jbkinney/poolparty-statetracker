"""Tests for design card functionality."""

import pytest
import pandas as pd
from poolparty import from_seqs, get_kmers, Pool
from poolparty.operations.mutation_scan_op import mutation_scan


class TestDesignCardBasics:
    """Tests for basic design card functionality."""
    
    def test_from_seqs_card_data(self):
        """Test that from_seqs stores results in _results_df."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        pool.set_sequential_op_states(0)
        seq = pool.seq
        
        # Check that the operation has results (now returns DataFrame)
        results_df = pool.operation.get_results()
        assert len(results_df) == 1
        assert 'seq' in results_df.columns
        assert 'seq_name' in results_df.columns
        assert 'seq_index' in results_df.columns
        assert results_df['seq'].iloc[0] == 'AAA'
        assert results_df['seq_index'].iloc[0] == 0
    
    def test_get_kmers_card_data(self):
        """Test that get_kmers stores results in _results_df."""
        pool = get_kmers(length=2, alphabet='dna', mode='sequential')
        pool.set_sequential_op_states(0)
        seq = pool.seq
        
        # Check that the operation has results (now returns DataFrame)
        results_df = pool.operation.get_results()
        assert len(results_df) == 1
        assert 'seq' in results_df.columns
        assert 'state' in results_df.columns
        assert results_df['seq'].iloc[0] == seq
        assert results_df['state'].iloc[0] == 0
    
    def test_mutation_scan_card_data(self):
        """Test that mutation_scan stores results in _results_df."""
        pool = mutation_scan('AAAA', num_mutations=1, mode='sequential')
        pool.set_sequential_op_states(0)
        seq = pool.seq
        
        # Get the mutation scan operation (not the from_seqs parent)
        # Now returns DataFrame
        results_df = pool.operation.get_results()
        assert len(results_df) == 1
        assert 'seq' in results_df.columns
        assert 'positions' in results_df.columns
        assert 'wt_chars' in results_df.columns
        assert 'mut_chars' in results_df.columns


class TestGenerateSequenceReturnsDataFrame:
    """Tests for generate_sequence returning DataFrame."""
    
    def test_returns_dataframe(self):
        """Test that generate_sequence returns a DataFrame."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        result = pool.generate_sequence(state=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_seq_column_first(self):
        """Test that 'seq' is the first column."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        result = pool.generate_sequence(state=0)
        assert result.columns[0] == 'seq'
        assert result['seq'].iloc[0] == 'AAA'
    
    def test_dataframe_has_design_card_columns(self):
        """Test that DataFrame has design card columns."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential', 
                         string_names=['first', 'second'])
        result = pool.generate_sequence(state=0)
        
        # Should have columns with op prefix format
        columns = list(result.columns)
        assert any('from_seqs' in col for col in columns)
        assert any('seq_name' in col for col in columns)


class TestGenerateLibraryReturnsDataFrame:
    """Tests for generate_library returning DataFrame."""
    
    def test_returns_dataframe(self):
        """Test that generate_library returns a DataFrame."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        result = pool.generate_library(num_seqs=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    
    def test_seq_column_first(self):
        """Test that 'seq' is the first column."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        result = pool.generate_library(num_seqs=2)
        assert result.columns[0] == 'seq'
    
    def test_dataframe_has_correct_columns(self):
        """Test that DataFrame has expected columns."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential',
                         string_names=['a', 'b', 'c'])
        result = pool.generate_library(num_seqs=3)
        
        # Should have columns with prefix format
        assert any('seq' in col for col in result.columns)
        assert any('seq_name' in col for col in result.columns)
        assert any('seq_index' in col for col in result.columns)
    
    def test_seq_values_correct(self):
        """Test that seq values are correct."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        result = pool.generate_library(num_seqs=2)
        
        assert list(result['seq']) == ['AAA', 'TTT']


class TestCompositePoolCards:
    """Tests for design cards with composite pools."""
    
    def test_concatenation_cards(self):
        """Test design cards with concatenated pools."""
        pool1 = from_seqs(['AA', 'BB'], mode='sequential')
        pool2 = from_seqs(['11', '22'], mode='sequential')
        concat = pool1 + pool2
        
        result = concat.generate_library(num_complete_iterations=1)
        
        # Should have columns from both from_seqs ops and the concat op
        columns = list(result.columns)
        
        # Check we have multiple ops represented
        assert len([c for c in columns if 'seq' in c]) >= 3  # seq + 2 from_seqs + concat
    
    def test_slice_cards(self):
        """Test design cards with sliced pool."""
        base = from_seqs(['ABCDE'], mode='sequential')
        sliced = base[1:4]
        
        result = sliced.generate_sequence(state=0)
        
        # Should have both from_seqs and slice ops in columns
        columns = list(result.columns)
        assert any('from_seqs' in col for col in columns)
        assert any('slice' in col for col in columns)
        
        # The main seq should be the sliced sequence
        assert result['seq'].iloc[0] == 'BCD'
    
    def test_repeat_cards(self):
        """Test design cards with repeated pool."""
        base = from_seqs(['AB'], mode='sequential')
        repeated = base * 3
        
        result = repeated.generate_sequence(state=0)
        
        # Should have both from_seqs and repeat ops in columns
        columns = list(result.columns)
        assert any('from_seqs' in col for col in columns)
        assert any('*' in col for col in columns)  # repeat uses '*' as name
        
        # The main seq should be the repeated sequence
        assert result['seq'].iloc[0] == 'ABABAB'
    
    def test_complex_dag_cards(self):
        """Test design cards with complex DAG."""
        # Build: (kmer + from_seqs)[0:4] * 2
        kmer = get_kmers(length=2, alphabet='dna', mode='sequential')
        suffix = from_seqs(['XX', 'YY'], mode='sequential')
        combined = kmer + suffix
        sliced = combined[0:3]
        final = sliced * 2
        
        result = final.generate_library(num_seqs=4)
        
        # Should have multiple operations in columns (using default function names)
        columns = list(result.columns)
        assert any('get_kmers' in c for c in columns)
        assert any('from_seqs' in c for c in columns)
        assert any('+' in c for c in columns)  # concatenate uses '+' as name
        assert any('slice' in c for c in columns)
        assert any('*' in c for c in columns)  # repeat uses '*' as name


class TestDesignCardClearCache:
    """Tests for clearing results DataFrame."""
    
    def test_clear_results(self):
        """Test that clear_results clears the _results_df."""
        pool = from_seqs(['AAA'], mode='sequential')
        pool.generate_sequence(state=0)
        
        # Should have data (now returns DataFrame)
        assert len(pool.operation.get_results()) > 0
        
        # Clear it
        pool.operation.clear_results()
        assert len(pool.operation.get_results()) == 0
