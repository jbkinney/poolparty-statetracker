"""Tests for shuffle_scan operation."""

import pytest
from collections import Counter
from poolparty.operations.shuffle_scan_op import shuffle_scan_op
from poolparty import Pool, from_seqs_op


class TestShuffleScan:
    """Tests for shuffle_scan factory function."""
    
    def test_basic_creation(self):
        """Test basic shuffle_scan pool creation."""
        pool = shuffle_scan_op('ACGTACGT', shuffle_size=4)
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 5  # 8 - 4 + 1 = 5 positions
    
    def test_sequential_enumerates_positions(self):
        """Test that sequential mode enumerates all positions."""
        pool = shuffle_scan_op('ABCDEFGH', shuffle_size=3, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 6  # 8 - 3 + 1 = 6 positions
    
    def test_preserves_length(self):
        """Test that shuffled sequences have same length."""
        pool = shuffle_scan_op('ACGTACGT', shuffle_size=4)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert len(seq) == 8
    
    def test_preserves_composition_in_window(self):
        """Test that character composition is preserved in shuffled window."""
        pool = shuffle_scan_op('AABBCCDD', shuffle_size=4, mode='sequential', mark_changes=False)
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # The window characters should be rearranged, not changed
        original_counts = Counter('AABBCCDD')
        for seq in seqs:
            assert Counter(seq) == original_counts
    
    def test_mark_changes(self):
        """Test mark_changes option."""
        pool = shuffle_scan_op('AAAA', shuffle_size=2, mark_changes=True, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Shuffled region should be lowercase (swapcase from uppercase)
        for seq in seqs:
            # Should have some lowercase characters
            assert any(c.islower() for c in seq)
    
    def test_num_shuffles(self):
        """Test num_shuffles parameter."""
        pool = shuffle_scan_op('ACGTACGT', shuffle_size=4, num_shuffles=3)
        # 5 positions × 3 shuffles = 15 states
        assert pool.operation.num_states == 15
    
    def test_step_size(self):
        """Test step_size parameter."""
        pool = shuffle_scan_op('ACGTACGTACGT', shuffle_size=4, step_size=2, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 0, 2, 4, 6 (12 - 4 + 1 = 9, range(0, 9, 2) = [0, 2, 4, 6])
        assert len(seqs) == 5  # 0, 2, 4, 6, 8
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = shuffle_scan_op('ACGTACGT', shuffle_size=4)
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2


class TestShuffleScanValidation:
    """Tests for input validation."""
    
    def test_shuffle_size_exceeds_length_raises(self):
        """Test that shuffle_size > length raises error."""
        with pytest.raises(ValueError, match="cannot exceed"):
            shuffle_scan_op('ACGT', shuffle_size=10)
    
    def test_shuffle_size_zero_raises(self):
        """Test that shuffle_size=0 raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            shuffle_scan_op('ACGT', shuffle_size=0)
    
    def test_num_shuffles_zero_raises(self):
        """Test that num_shuffles=0 raises error."""
        with pytest.raises(ValueError, match="num_shuffles"):
            shuffle_scan_op('ACGT', shuffle_size=2, num_shuffles=0)


class TestShuffleScanAncestors:
    """Tests for ancestor tracking."""
    
    def test_has_parent_pool(self):
        """Test that shuffle_scan has parent pool."""
        pool = shuffle_scan_op('ACGT', shuffle_size=2)
        parents = pool.operation.parent_pools
        assert len(parents) == 1
