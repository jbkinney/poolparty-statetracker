"""Tests for deletion_scan operation."""

import pytest
from poolparty.operations.deletion_scan_op import deletion_scan_op
from poolparty import Pool, from_seqs_op


class TestDeletionScan:
    """Tests for deletion_scan factory function."""
    
    def test_basic_creation(self):
        """Test basic deletion_scan pool creation."""
        pool = deletion_scan_op('ACGTACGT', deletion_size=2)
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 7  # 8 - 2 + 1 = 7 positions
    
    def test_sequential_enumerates_all(self):
        """Test that sequential mode enumerates all positions."""
        pool = deletion_scan_op('ABCDEFGH', deletion_size=2, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 7  # 8 - 2 + 1 = 7 positions
        
        # Check first and last deletions
        assert seqs[0] == '--CDEFGH'  # Deletion at position 0
        assert seqs[-1] == 'ABCDEF--'  # Deletion at position 6
    
    def test_mark_changes_true(self):
        """Test mark_changes=True keeps length same."""
        pool = deletion_scan_op('ACGTACGT', deletion_size=2, mark_changes=True)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert len(seq) == 8  # Same as original
            assert '--' in seq  # Deletion marked
    
    def test_mark_changes_false(self):
        """Test mark_changes=False shortens sequence."""
        pool = deletion_scan_op('ACGTACGT', deletion_size=2, mark_changes=False)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert len(seq) == 6  # 8 - 2 = 6
    
    def test_custom_deletion_character(self):
        """Test custom deletion_character."""
        pool = deletion_scan_op('ACGTACGT', deletion_size=2, deletion_character='X')
        seqs = pool.generate_library(num_seqs=5, seed=42)
        
        for seq in seqs:
            assert 'XX' in seq
    
    def test_step_size(self):
        """Test step_size parameter."""
        pool = deletion_scan_op('ABCDEFGHIJ', deletion_size=2, step_size=3, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 0, 3, 6 (10 - 2 + 1 = 9, range(0, 9, 3) = [0, 3, 6])
        assert len(seqs) == 3
        assert seqs[0] == '--CDEFGHIJ'  # Position 0
        assert seqs[1] == 'ABC--FGHIJ'  # Position 3
    
    def test_explicit_positions(self):
        """Test explicit positions parameter."""
        pool = deletion_scan_op('ABCDEFGH', deletion_size=2, positions=[0, 3], mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 2
        assert seqs[0] == '--CDEFGH'
        assert seqs[1] == 'ABC--FGH'
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = deletion_scan_op('ACGTACGT', deletion_size=2)
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2


class TestDeletionScanValidation:
    """Tests for input validation."""
    
    def test_deletion_size_exceeds_length_raises(self):
        """Test that deletion_size > length raises error."""
        with pytest.raises(ValueError, match="cannot be larger"):
            deletion_scan_op('ACGT', deletion_size=10)
    
    def test_deletion_size_zero_raises(self):
        """Test that deletion_size=0 raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            deletion_scan_op('ACGT', deletion_size=0)
    
    def test_both_interfaces_raises(self):
        """Test that using both interfaces raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            deletion_scan_op('ACGT', deletion_size=1, start=1, positions=[0, 1])


class TestDeletionScanAncestors:
    """Tests for ancestor tracking."""
    
    def test_has_parent_pool(self):
        """Test that deletion_scan has parent pool."""
        pool = deletion_scan_op('ACGT', deletion_size=1)
        parents = pool.operation.parent_pools
        assert len(parents) == 1
