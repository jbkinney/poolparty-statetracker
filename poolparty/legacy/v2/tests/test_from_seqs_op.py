"""Tests for from_seqs operation."""

import pytest
from poolparty import from_seqs, Pool


class TestFromSeqs:
    """Tests for from_seqs factory function."""
    
    def test_basic_creation(self):
        """Test basic pool creation from sequences."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'])
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 3
        assert pool.seq_length == 3
    
    def test_seq_at_different_states(self):
        """Test that different states return sequences (probabilistic in random mode)."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'])
        
        # In random mode with uniform distribution, sequences are randomly sampled
        # Just verify that we get valid sequences
        pool.set_sequential_op_states(0)
        assert pool.seq in ['AAA', 'TTT', 'GGG']
        
        pool.set_sequential_op_states(1)
        assert pool.seq in ['AAA', 'TTT', 'GGG']
        
        pool.set_sequential_op_states(2)
        assert pool.seq in ['AAA', 'TTT', 'GGG']
    
    def test_state_wrapping(self):
        """Test that states wrap around correctly in sequential mode."""
        pool = from_seqs(['AAA', 'TTT'], mode='sequential')
        
        pool.set_sequential_op_states(0)
        assert pool.seq == 'AAA'
        
        pool.set_sequential_op_states(2)  # Should wrap to 0
        assert pool.seq == 'AAA'
        
        pool.set_sequential_op_states(3)  # Should wrap to 1
        assert pool.seq == 'TTT'
    
    def test_sequential_mode(self):
        """Test sequential mode generates sequences in order."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        assert list(result_df['seq']) == ['AAA', 'TTT', 'GGG']
    
    def test_generate_library_num_seqs(self):
        """Test generate_library with num_seqs parameter."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        result_df = pool.generate_library(num_seqs=5)
        assert len(result_df) == 5
        # First 3 should match the pool, then wrap
        seqs = list(result_df['seq'])
        assert seqs[:3] == ['AAA', 'TTT', 'GGG']
        assert seqs[3:5] == ['AAA', 'TTT']
    
    def test_empty_seqs_raises(self):
        """Test that empty sequences list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            from_seqs([])
    
    def test_different_lengths_allowed(self):
        """Test that sequences with different lengths are now allowed."""
        pool = from_seqs(['AAA', 'TT', 'GGGG'])
        assert pool.seq_length is None  # Variable length
    
    def test_name_attribute(self):
        """Test name attribute is on operation."""
        pool = from_seqs(['AAA'], name='test_pool')
        assert pool.operation.name == 'test_pool'
    

class TestFromSeqsAncestors:
    """Tests for ancestor tracking in from_seqs pools."""
    
    def test_no_parent_pools(self):
        """Test that from_seqs has no parent pools."""
        pool = from_seqs(['AAA', 'TTT'])
        assert pool.operation.parent_pools == []
