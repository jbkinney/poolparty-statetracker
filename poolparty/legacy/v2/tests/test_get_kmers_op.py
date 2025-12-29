"""Tests for get_kmers operation."""

import pytest
from poolparty import get_kmers, Pool


class TestKmerPool:
    """Tests for get_kmers factory function."""
    
    def test_basic_creation(self):
        """Test basic kmer_pool creation."""
        pool = get_kmers(length=3, alphabet='dna')
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 64  # 4^3
        assert pool.seq_length == 3
    
    def test_num_states_calculation(self):
        """Test that num_states is calculated correctly."""
        # DNA alphabet (4 chars)
        pool = get_kmers(length=2, alphabet='dna')
        assert pool.operation.num_states == 16  # 4^2
        
        # Binary alphabet
        pool = get_kmers(length=4, alphabet=['0', '1'])
        assert pool.operation.num_states == 16  # 2^4
        
        # Larger alphabet
        pool = get_kmers(length=2, alphabet='protein')  # 20 chars
        assert pool.operation.num_states == 400  # 20^2
    
    def test_seq_length(self):
        """Test sequence length matches requested length."""
        pool = get_kmers(length=5, alphabet='dna')
        pool.set_state(0)
        assert len(pool.seq) == 5
        
        pool = get_kmers(length=10, alphabet='dna')
        pool.set_state(0)
        assert len(pool.seq) == 10
    
    def test_state_to_sequence_mapping(self):
        """Test that state maps deterministically to k-mer in sequential mode."""
        pool = get_kmers(length=2, alphabet=['A', 'C', 'G', 'T'], mode='sequential')
        
        # State 0 should be 'AA' (first char repeated)
        pool.set_state(0)
        assert pool.seq == 'AA'
        
        # State 1 should be 'AC'
        pool.set_state(1)
        assert pool.seq == 'AC'
        
        # State 4 should be 'CA' (second digit increments)
        pool.set_state(4)
        assert pool.seq == 'CA'
    
    def test_sequential_mode_enumerates_all(self):
        """Test that sequential mode enumerates all k-mers."""
        pool = get_kmers(length=2, alphabet=['A', 'C'], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Should get all 4 combinations
        assert len(result_df) == 4
        assert set(result_df['seq']) == {'AA', 'AC', 'CA', 'CC'}
    
    def test_generate_library_with_seed(self):
        """Test that same seed produces same sequences."""
        pool1 = get_kmers(length=5, alphabet='dna')
        pool2 = get_kmers(length=5, alphabet='dna')
        
        result_df1 = pool1.generate_library(num_seqs=10, seed=42)
        result_df2 = pool2.generate_library(num_seqs=10, seed=42)
        
        assert list(result_df1['seq']) == list(result_df2['seq'])
    
    def test_custom_alphabet(self):
        """Test kmer_pool with custom alphabet in sequential mode."""
        pool = get_kmers(length=3, alphabet=['X', 'Y', 'Z'], mode='sequential')
        assert pool.operation.num_states == 27  # 3^3
        
        pool.set_state(0)
        assert pool.seq == 'XXX'
        
        pool.set_state(1)
        assert pool.seq == 'XXY'
    
    def test_invalid_length_raises(self):
        """Test that invalid length raises error."""
        with pytest.raises(ValueError):
            get_kmers(length=0, alphabet='dna')
        
        with pytest.raises(ValueError):
            get_kmers(length=-1, alphabet='dna')
    
    def test_invalid_alphabet_raises(self):
        """Test that invalid alphabet raises error."""
        with pytest.raises((KeyError, ValueError, TypeError)):
            get_kmers(length=3, alphabet='invalid_name')
        
        with pytest.raises(ValueError):
            get_kmers(length=3, alphabet=[])


class TestKmerPoolAncestors:
    """Tests for ancestor tracking in kmer_pool pools."""
    
    def test_no_parent_pools(self):
        """Test that kmer_pool has no parent pools."""
        pool = get_kmers(length=3, alphabet='dna')
        assert pool.operation.parent_pools == []


class TestMultiLengthKmers:
    """Tests for multi-length k-mer generation."""
    
    def test_multi_length_num_states(self):
        """Test num_states calculation for multiple lengths."""
        # Binary alphabet for easy math
        pool = get_kmers(length=[2, 3], alphabet=['0', '1'], mode='sequential')
        # 2^2 + 2^3 = 4 + 8 = 12
        assert pool.operation.num_states == 12
        
        # DNA alphabet
        pool = get_kmers(length=[1, 2], alphabet='dna', mode='sequential')
        # 4^1 + 4^2 = 4 + 16 = 20
        assert pool.operation.num_states == 20
    
    def test_multi_length_seq_length_is_none(self):
        """Test that seq_length is None for multiple lengths."""
        pool = get_kmers(length=[2, 3], alphabet='dna')
        assert pool.seq_length is None
    
    def test_single_length_list_behaves_like_int(self):
        """Test that single-element list behaves like int."""
        pool_int = get_kmers(length=3, alphabet='dna', mode='sequential')
        pool_list = get_kmers(length=[3], alphabet='dna', mode='sequential')
        
        assert pool_int.operation.num_states == pool_list.operation.num_states
        assert pool_int.seq_length == pool_list.seq_length
        assert pool_int.seq_length == 3
    
    def test_multi_length_sequential_enumeration(self):
        """Test that sequential mode enumerates all kmers for each length in order."""
        pool = get_kmers(length=[1, 2], alphabet=['A', 'B'], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Should have 2 + 4 = 6 sequences
        assert len(result_df) == 6
        
        # First 2 should be 1-mers
        assert result_df['seq'].iloc[0] == 'A'
        assert result_df['seq'].iloc[1] == 'B'
        
        # Next 4 should be 2-mers
        assert set(result_df['seq'].iloc[2:]) == {'AA', 'AB', 'BA', 'BB'}
    
    def test_multi_length_sequential_length_tracking(self):
        """Test that design card tracks correct length for each sequence."""
        pool = get_kmers(length=[1, 2], alphabet=['A', 'B'], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Check length column exists and has correct values
        length_col = [c for c in result_df.columns if 'length' in c][0]
        lengths = result_df[length_col].tolist()
        
        # First 2 are length 1, next 4 are length 2
        assert lengths[:2] == [1, 1]
        assert lengths[2:] == [2, 2, 2, 2]
    
    def test_multi_length_random_mode(self):
        """Test random mode with multiple lengths."""
        pool = get_kmers(length=[2, 5], alphabet='dna', mode='random')
        result_df = pool.generate_library(num_seqs=100, seed=42)
        
        # All sequences should be length 2 or 5
        seq_lengths = result_df['seq'].str.len()
        assert set(seq_lengths.unique()) == {2, 5}
        
        # With uniform probs, should see both lengths (statistically)
        assert (seq_lengths == 2).sum() > 0
        assert (seq_lengths == 5).sum() > 0
    
    def test_multi_length_random_with_custom_probs(self):
        """Test that length_probs affects length distribution in random mode."""
        # Heavily bias toward length 2
        pool = get_kmers(length=[2, 10], alphabet='dna', mode='random', 
                         length_probs=[0.99, 0.01])
        result_df = pool.generate_library(num_seqs=100, seed=42)
        
        seq_lengths = result_df['seq'].str.len()
        # Most should be length 2
        assert (seq_lengths == 2).sum() > 80
    
    def test_multi_length_random_reproducible_with_seed(self):
        """Test that random mode is reproducible with same seed."""
        pool1 = get_kmers(length=[2, 3, 4], alphabet='dna', mode='random')
        pool2 = get_kmers(length=[2, 3, 4], alphabet='dna', mode='random')
        
        result_df1 = pool1.generate_library(num_seqs=50, seed=123)
        result_df2 = pool2.generate_library(num_seqs=50, seed=123)
        
        assert list(result_df1['seq']) == list(result_df2['seq'])
    
    def test_multi_length_state_mapping(self):
        """Test specific state to sequence mapping with multiple lengths."""
        pool = get_kmers(length=[1, 2], alphabet=['A', 'B'], mode='sequential')
        
        # State 0: first 1-mer = 'A'
        pool.set_state(0)
        assert pool.seq == 'A'
        
        # State 1: second 1-mer = 'B'
        pool.set_state(1)
        assert pool.seq == 'B'
        
        # State 2: first 2-mer = 'AA'
        pool.set_state(2)
        assert pool.seq == 'AA'
        
        # State 5: fourth 2-mer = 'BB'
        pool.set_state(5)
        assert pool.seq == 'BB'


class TestMultiLengthValidation:
    """Tests for validation of multi-length k-mer parameters."""
    
    def test_empty_length_list_raises(self):
        """Test that empty length list raises error."""
        with pytest.raises(ValueError):
            get_kmers(length=[], alphabet='dna')
    
    def test_invalid_length_in_list_raises(self):
        """Test that invalid length in list raises error."""
        with pytest.raises(ValueError):
            get_kmers(length=[2, 0, 3], alphabet='dna')
        
        with pytest.raises(ValueError):
            get_kmers(length=[2, -1], alphabet='dna')
    
    def test_length_probs_wrong_length_raises(self):
        """Test that length_probs with wrong length raises error."""
        with pytest.raises(ValueError):
            get_kmers(length=[2, 3], length_probs=[0.5], alphabet='dna')
        
        with pytest.raises(ValueError):
            get_kmers(length=[2, 3], length_probs=[0.3, 0.3, 0.4], alphabet='dna')
    
    def test_negative_length_probs_raises(self):
        """Test that negative probabilities raise error."""
        with pytest.raises(ValueError):
            get_kmers(length=[2, 3], length_probs=[0.5, -0.5], alphabet='dna')
    
    def test_zero_sum_length_probs_raises(self):
        """Test that all-zero probabilities raise error."""
        with pytest.raises(ValueError):
            get_kmers(length=[2, 3], length_probs=[0.0, 0.0], alphabet='dna')
    
    def test_length_probs_normalized(self):
        """Test that length_probs are normalized internally."""
        pool = get_kmers(length=[2, 3], length_probs=[2.0, 8.0], alphabet='dna')
        # Should be normalized to [0.2, 0.8]
        assert abs(pool.operation.length_probs[0] - 0.2) < 1e-10
        assert abs(pool.operation.length_probs[1] - 0.8) < 1e-10
