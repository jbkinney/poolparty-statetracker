"""Tests for reset_op_id_counter utility function."""

import pytest
from poolparty import reset_op_id_counter, Operation, from_seqs


class TestResetOpIdCounter:
    """Tests for reset_op_id_counter function."""
    
    def test_resets_counter_to_zero(self):
        """Test that reset_op_id_counter sets Operation.id_counter to 0."""
        # Set counter to a non-zero value
        Operation.id_counter = 42
        
        reset_op_id_counter()
        
        assert Operation.id_counter == 0
    
    def test_operations_start_from_zero_after_reset(self):
        """Test that new operations get IDs starting from 0 after reset."""
        # Create some operations to increment counter
        _ = from_seqs(['AAA', 'TTT'])
        _ = from_seqs(['GGG', 'CCC'])
        
        # Counter should be > 0
        assert Operation.id_counter > 0
        
        reset_op_id_counter()
        
        # New operation should have id=0
        pool = from_seqs(['ACGT'])
        assert pool.operation.id == 0
    
    def test_subsequent_operations_increment_from_zero(self):
        """Test that operations created after reset have sequential IDs from 0."""
        reset_op_id_counter()
        
        pool1 = from_seqs(['AAA'])
        pool2 = from_seqs(['TTT'])
        pool3 = from_seqs(['GGG'])
        
        assert pool1.operation.id == 0
        assert pool2.operation.id == 1
        assert pool3.operation.id == 2
    
    def test_multiple_resets(self):
        """Test that multiple consecutive resets work correctly."""
        reset_op_id_counter()
        reset_op_id_counter()
        reset_op_id_counter()
        
        assert Operation.id_counter == 0
        
        pool = from_seqs(['ACGT'])
        assert pool.operation.id == 0
    
    def test_reset_after_creating_many_operations(self):
        """Test reset works after creating many operations."""
        reset_op_id_counter()
        
        # Create many operations
        for i in range(100):
            _ = from_seqs(['A' * 10])
        
        assert Operation.id_counter == 100
        
        reset_op_id_counter()
        
        assert Operation.id_counter == 0
        pool = from_seqs(['ACGT'])
        assert pool.operation.id == 0
    
    def test_reset_returns_none(self):
        """Test that reset_op_id_counter returns None."""
        result = reset_op_id_counter()
        assert result is None
