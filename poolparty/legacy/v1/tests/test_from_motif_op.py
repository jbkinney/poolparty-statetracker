"""Tests for from_motif operation."""

import pytest
import numpy as np
import pandas as pd
from poolparty.operations.from_motif_op import from_motif_op
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
        pool = from_motif_op(df)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 10
        assert pool.operation.num_states == -1
    
    def test_generates_correct_length(self):
        """Test that sequences have correct length."""
        df = make_uniform_df(15)
        pool = from_motif_op(df)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        for seq in seqs:
            assert len(seq) == 15
    
    def test_uses_alphabet(self):
        """Test that sequences only contain characters from the alphabet."""
        df = make_uniform_df(20, alphabet='ACGT')
        pool = from_motif_op(df)
        seqs = pool.generate_library(num_seqs=50, seed=42)
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_deterministic_sequence(self):
        """Test that deterministic probabilities generate expected sequence."""
        df = make_deterministic_df('ACGT')
        pool = from_motif_op(df)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        # All sequences should be 'ACGT' since probabilities are 1.0
        for seq in seqs:
            assert seq == 'ACGT'
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        df = make_uniform_df(10)
        pool1 = from_motif_op(df)
        pool2 = from_motif_op(df)
        
        seqs1 = pool1.generate_library(num_seqs=20, seed=42)
        seqs2 = pool2.generate_library(num_seqs=20, seed=42)
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        df = make_uniform_df(10)
        pool = from_motif_op(df)
        
        seqs1 = pool.generate_library(num_seqs=20, seed=42)
        seqs2 = pool.generate_library(num_seqs=20, seed=123)
        
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
        pool = from_motif_op(df)
        seqs = pool.generate_library(num_seqs=1000, seed=42)
        
        # Count first position As
        first_a_count = sum(1 for seq in seqs if seq[0] == 'A')
        
        # Should be close to 90%
        assert 0.85 < first_a_count / 1000 < 0.95


class TestFromMotifOrientation:
    """Tests for orientation handling."""
    
    def test_forward_orientation(self):
        """Test forward orientation."""
        df = make_deterministic_df('ACGT')
        pool = from_motif_op(df, orientation='forward')
        seqs = pool.generate_library(num_seqs=5, seed=42)
        
        for seq in seqs:
            assert seq == 'ACGT'
    
    def test_reverse_orientation(self):
        """Test reverse orientation (always reverse complement)."""
        df = make_deterministic_df('ACGT')
        pool = from_motif_op(df, orientation='reverse')
        seqs = pool.generate_library(num_seqs=5, seed=42)
        
        # Reverse complement of ACGT is ACGT (palindrome!)
        # Let's use a non-palindromic sequence
        df2 = make_deterministic_df('AAAA')
        pool2 = from_motif_op(df2, orientation='reverse')
        seqs2 = pool2.generate_library(num_seqs=5, seed=42)
        
        # Reverse complement of AAAA is TTTT
        for seq in seqs2:
            assert seq == 'TTTT'
    
    def test_both_orientation(self):
        """Test 'both' orientation (random choice)."""
        df = make_deterministic_df('AACC')
        pool = from_motif_op(df, orientation='both', forward_prob=0.5)
        seqs = pool.generate_library(num_seqs=100, seed=42)
        
        # Should have a mix of forward and reverse
        # Forward: AACC, Reverse: GGTT
        forward_count = sum(1 for seq in seqs if seq == 'AACC')
        reverse_count = sum(1 for seq in seqs if seq == 'GGTT')
        
        assert forward_count > 0
        assert reverse_count > 0
        assert forward_count + reverse_count == 100
    
    def test_forward_prob_biased(self):
        """Test forward_prob affects orientation ratio."""
        df = make_deterministic_df('AACC')
        
        # 90% forward probability
        pool = from_motif_op(df, orientation='both', forward_prob=0.9)
        seqs = pool.generate_library(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for seq in seqs if seq == 'AACC')
        
        # Should be close to 90%
        assert 0.85 < forward_count / 1000 < 0.95


class TestFromMotifValidation:
    """Tests for input validation."""
    
    def test_sequential_mode_raises(self):
        """Test that sequential mode raises error."""
        df = make_uniform_df(5)
        with pytest.raises(ValueError, match="random"):
            from_motif_op(df, mode='sequential')
    
    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="non-empty"):
            from_motif_op(df)
    
    def test_invalid_orientation_raises(self):
        """Test that invalid orientation raises error."""
        df = make_uniform_df(5)
        with pytest.raises(ValueError, match="orientation"):
            from_motif_op(df, orientation='invalid')
    
    def test_invalid_forward_prob_raises(self):
        """Test that invalid forward_prob raises error."""
        df = make_uniform_df(5)
        with pytest.raises(ValueError, match="forward_prob"):
            from_motif_op(df, forward_prob=1.5)
        
        with pytest.raises(ValueError, match="forward_prob"):
            from_motif_op(df, forward_prob=-0.1)
    
    def test_non_single_char_column_raises(self):
        """Test that non-single-character column names raise error."""
        df = pd.DataFrame({'AA': [0.5, 0.5], 'T': [0.5, 0.5]})
        with pytest.raises(ValueError, match="single characters"):
            from_motif_op(df)
    
    def test_duplicate_columns_raises(self):
        """Test that duplicate column names raise error."""
        df = pd.DataFrame([[0.5, 0.5]], columns=['A', 'A'])
        with pytest.raises(ValueError, match="unique"):
            from_motif_op(df)
    
    def test_nan_values_raises(self):
        """Test that NaN values raise error."""
        df = pd.DataFrame({'A': [0.5, np.nan], 'T': [0.5, 0.5]})
        with pytest.raises(ValueError, match="NaN"):
            from_motif_op(df)
    
    def test_negative_probability_raises(self):
        """Test that negative probabilities raise error."""
        df = pd.DataFrame({'A': [-0.1, 0.5], 'T': [1.1, 0.5]})
        with pytest.raises(ValueError, match="non-negative"):
            from_motif_op(df)
    
    def test_row_sum_not_one_raises(self):
        """Test that rows not summing to 1.0 raise error."""
        df = pd.DataFrame({'A': [0.3, 0.5], 'T': [0.3, 0.5]})  # Sum = 0.6
        with pytest.raises(ValueError, match="sum to approximately 1.0"):
            from_motif_op(df)


class TestFromMotifAncestors:
    """Tests for ancestor tracking in from_motif pools."""
    
    def test_no_parent_pools(self):
        """Test that from_motif has no parent pools."""
        df = make_uniform_df(5)
        pool = from_motif_op(df)
        assert pool.operation.parent_pools == []
    
    def test_ancestors_include_self(self):
        """Test that ancestors include the pool itself."""
        df = make_uniform_df(5)
        pool = from_motif_op(df)
        assert pool in pool.ancestors
        assert len(pool.ancestors) == 1
