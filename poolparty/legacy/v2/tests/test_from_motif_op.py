"""Tests for from_motif operation."""

import pytest
import numpy as np
import pandas as pd
from poolparty.operations.from_motif_op import from_motif
from poolparty import Pool


def make_uniform_df(length: int, alphabet: str = 'ACGT') -> pd.DataFrame:
    """Create a uniform probability matrix."""
    n_chars = len(alphabet)
    prob = 1.0 / n_chars
    data = {char: [prob] * length for char in alphabet}
    return pd.DataFrame(data)


def make_deterministic_df(seq: str) -> pd.DataFrame:
    """Create a probability matrix that always generates the given sequence."""
    alphabet = 'ACGT'
    data = {char: [] for char in alphabet}
    for base in seq:
        for char in alphabet:
            data[char].append(1.0 if char == base else 0.0)
    return pd.DataFrame(data)


class TestFromMotif:
    """Tests for from_motif factory function."""
    
    def test_basic_creation(self):
        """Test basic from_motif pool creation."""
        df = make_uniform_df(10)
        pool = from_motif(df)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 10
        assert pool.operation.num_states == -1
    
    def test_generates_correct_length(self):
        """Test that sequences have correct length."""
        df = make_uniform_df(15)
        pool = from_motif(df)
        result_df = pool.generate_library(num_seqs=10, seed=42)
        seqs = list(result_df['seq'])
        for seq in seqs:
            assert len(seq) == 15
    
    def test_uses_alphabet(self):
        """Test that sequences only contain characters from the alphabet."""
        df = make_uniform_df(20, alphabet='ACGT')
        pool = from_motif(df)
        result_df = pool.generate_library(num_seqs=50, seed=42)
        seqs = list(result_df['seq'])
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_deterministic_sequence(self):
        """Test that deterministic probabilities generate expected sequence."""
        df = make_deterministic_df('ACGT')
        pool = from_motif(df)
        result_df = pool.generate_library(num_seqs=10, seed=42)
        seqs = list(result_df['seq'])
        # All sequences should be 'ACGT' since probabilities are 1.0
        for seq in seqs:
            assert seq == 'ACGT'
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        df = make_uniform_df(10)
        pool1 = from_motif(df)
        pool2 = from_motif(df)
        
        result_df1 = pool1.generate_library(num_seqs=20, seed=42)
        result_df2 = pool2.generate_library(num_seqs=20, seed=42)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        df = make_uniform_df(10)
        pool = from_motif(df)
        
        result_df1 = pool.generate_library(num_seqs=20, seed=42)
        result_df2 = pool.generate_library(num_seqs=20, seed=123)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        # Should have some different sequences
        assert seqs1 != seqs2
    
    def test_probability_distribution(self):
        """Test that sampling follows the probability distribution."""
        # Create a biased distribution: A has 90% probability at first position
        df = pd.DataFrame({
            'A': [0.9, 0.25, 0.25, 0.25],
            'C': [0.033, 0.25, 0.25, 0.25],
            'G': [0.033, 0.25, 0.25, 0.25],
            'T': [0.034, 0.25, 0.25, 0.25],
        })
        pool = from_motif(df)
        result_df = pool.generate_library(num_seqs=1000, seed=42)
        seqs = list(result_df['seq'])
        
        # Count first position As
        first_a_count = sum(1 for seq in seqs if seq[0] == 'A')
        
        # Should be close to 90%
        assert 0.85 < first_a_count / 1000 < 0.95


class TestFromMotifValidation:
    """Tests for input validation."""
    
    def test_sequential_mode_raises(self):
        """Test that sequential mode raises error."""
        df = make_uniform_df(5)
        with pytest.raises(ValueError, match="random"):
            from_motif(df, mode='sequential')
    
    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="non-empty"):
            from_motif(df)
    
    def test_columns_subset_of_alphabet_allowed(self):
        """Test that columns can be a subset of alphabet (missing filled with zeros)."""
        # Missing columns G, T - should work, with G and T having zero probability
        df = pd.DataFrame({'A': [0.5, 0.5], 'C': [0.5, 0.5]})
        pool = from_motif(df, alphabet='dna')
        assert pool.seq_length == 2
        
        # All generated sequences should only contain A or C
        result_df = pool.generate_library(num_seqs=100, seed=42)
        seqs = list(result_df['seq'])
        for seq in seqs:
            assert all(c in 'AC' for c in seq)
    
    def test_extra_columns_raises(self):
        """Test that extra columns not in alphabet raise error."""
        df = pd.DataFrame({'A': [0.25], 'C': [0.25], 'G': [0.25], 'T': [0.25], 'X': [0.0]})
        with pytest.raises(ValueError, match="Extra"):
            from_motif(df, alphabet='dna')
    
    def test_custom_alphabet(self):
        """Test that custom alphabet works."""
        df = pd.DataFrame({'X': [0.5, 0.5], 'Y': [0.5, 0.5]})
        pool = from_motif(df, alphabet=['X', 'Y'])
        assert pool.seq_length == 2
    
    def test_nan_values_raises(self):
        """Test that NaN values raise error."""
        df = pd.DataFrame({'A': [0.5, np.nan], 'C': [0.1, 0.1], 'G': [0.2, 0.2], 'T': [0.2, 0.2]})
        with pytest.raises(ValueError, match="NaN"):
            from_motif(df)
    
    def test_negative_probability_raises(self):
        """Test that negative probabilities raise error."""
        df = pd.DataFrame({'A': [-0.1, 0.5], 'C': [0.4, 0.1], 'G': [0.4, 0.2], 'T': [0.3, 0.2]})
        with pytest.raises(ValueError, match=">= 0"):
            from_motif(df)
    
    def test_row_sum_zero_raises(self):
        """Test that rows summing to zero raise error."""
        df = pd.DataFrame({'A': [0.0, 0.5], 'C': [0.0, 0.2], 'G': [0.0, 0.2], 'T': [0.0, 0.1]})
        with pytest.raises(ValueError, match="sum to zero"):
            from_motif(df)
    
    def test_rows_auto_normalized(self):
        """Test that rows are automatically normalized to sum to 1."""
        # Create unnormalized probabilities (sum to 2 per row)
        df = pd.DataFrame({'A': [0.5, 0.5], 'C': [0.5, 0.5], 'G': [0.5, 0.5], 'T': [0.5, 0.5]})
        pool = from_motif(df)
        # Should not raise - rows are auto-normalized
        result_df = pool.generate_library(num_seqs=10, seed=42)
        assert len(result_df) == 10


class TestFromMotifAncestors:
    """Tests for ancestor tracking in from_motif pools."""
    
    def test_no_parent_pools(self):
        """Test that from_motif has no parent pools."""
        df = make_uniform_df(5)
        pool = from_motif(df)
        assert pool.operation.parent_pools == []
