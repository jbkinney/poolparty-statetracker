"""Tests for insertion_scan operation."""

import pytest
from poolparty.operations.insertion_scan_op import insertion_scan_op
from poolparty import Pool, from_seqs_op


class TestInsertionScan:
    """Tests for insertion_scan factory function."""
    
    def test_basic_creation_overwrite(self):
        """Test basic insertion_scan with overwrite mode."""
        pool = insertion_scan_op('AAAA', 'XX', insert_or_overwrite='overwrite')
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 3  # 4 - 2 + 1 = 3 positions
        assert pool.seq_length == 4  # Same as original
    
    def test_basic_creation_insert(self):
        """Test basic insertion_scan with insert mode."""
        pool = insertion_scan_op('AAAA', 'XX', insert_or_overwrite='insert')
        assert pool.operation.num_states == 5  # 0, 1, 2, 3, 4 positions
        assert pool.seq_length == 6  # 4 + 2 = 6
    
    def test_overwrite_sequential(self):
        """Test overwrite mode with sequential iteration."""
        pool = insertion_scan_op('AAAA', 'XX', insert_or_overwrite='overwrite', mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 3
        expected = ['XXAA', 'AXXA', 'AAXX']
        assert seqs == expected
    
    def test_insert_sequential(self):
        """Test insert mode with sequential iteration."""
        pool = insertion_scan_op('AA', 'XX', insert_or_overwrite='insert', mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 3
        expected = ['XXAA', 'AXXA', 'AAXX']
        assert seqs == expected
    
    def test_mark_changes(self):
        """Test mark_changes option."""
        pool = insertion_scan_op('AAAA', 'XX', mark_changes=True, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # 'XX' should be swapped to 'xx'
            assert 'xx' in seq
    
    def test_step_size(self):
        """Test step_size parameter."""
        pool = insertion_scan_op('AAAAAA', 'XX', step_size=2, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 0, 2, 4 (6 - 2 + 1 = 5, range(0, 5, 2) = [0, 2, 4])
        assert len(seqs) == 3
    
    def test_explicit_positions(self):
        """Test explicit positions parameter."""
        pool = insertion_scan_op('AAAA', 'XX', positions=[0, 2], mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 2
        assert seqs[0] == 'XXAA'
        assert seqs[1] == 'AAXX'
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = insertion_scan_op('AAAA', 'XX')
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_with_pool_parents(self):
        """Test insertion_scan with Pool parents."""
        background = from_seqs_op(['AAAA', 'TTTT'], mode='sequential')
        insert = from_seqs_op(['XX', 'YY'], mode='sequential')
        pool = insertion_scan_op(background, insert, mode='sequential')
        
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # 2 background × 2 insert × 3 positions = 12 total
        assert len(seqs) == 12


class TestInsertionScanValidation:
    """Tests for input validation."""
    
    def test_insert_longer_than_background_overwrite_raises(self):
        """Test that insert > background in overwrite mode raises error."""
        with pytest.raises(ValueError, match="cannot exceed"):
            insertion_scan_op('AA', 'XXXX', insert_or_overwrite='overwrite')
    
    def test_invalid_mode_raises(self):
        """Test that invalid insert_or_overwrite raises error."""
        with pytest.raises(ValueError, match="insert_or_overwrite"):
            insertion_scan_op('AAAA', 'XX', insert_or_overwrite='invalid')
    
    def test_both_interfaces_raises(self):
        """Test that using both interfaces raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            insertion_scan_op('AAAA', 'XX', start=1, positions=[0, 1])


class TestInsertionScanAncestors:
    """Tests for ancestor tracking."""
    
    def test_has_parent_pools(self):
        """Test that insertion_scan has two parent pools."""
        pool = insertion_scan_op('AAAA', 'XX')
        parents = pool.operation.parent_pools
        assert len(parents) == 2
