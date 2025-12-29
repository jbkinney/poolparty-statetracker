"""Tests for multiinsertion_scan operation."""

import pytest
from poolparty.operations.multiinsertion_scan_op import multiinsertion_scan_op
from poolparty import Pool, from_seqs_op


class TestMultiinsertionScan:
    """Tests for multiinsertion_scan factory function."""
    
    def test_basic_creation(self):
        """Test basic multiinsertion_scan pool creation."""
        pool = multiinsertion_scan_op(
            'NNNNNNNNNN',
            insert_seqs=['AA', 'BB'],
            anchor_pos=5,
            insert_ranges=[(-4, 0), (1, 4)],
        )
        assert isinstance(pool, Pool)
        assert pool.operation.num_states > 0
    
    def test_overwrite_mode(self):
        """Test overwrite mode."""
        pool = multiinsertion_scan_op(
            'NNNNNNNNNN',
            insert_seqs=['XX'],
            anchor_pos=5,
            insert_ranges=[(-2, 3)],
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # All should have same length as original
        for seq in seqs:
            assert len(seq) == 10
            assert 'XX' in seq
    
    def test_insert_mode(self):
        """Test insert mode."""
        pool = multiinsertion_scan_op(
            'NNNNNNNN',
            insert_seqs=['XX'],
            anchor_pos=4,
            insert_ranges=[(-2, 3)],
            insert_or_overwrite='insert',
            mode='sequential'
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # All should be original + insert length
        for seq in seqs:
            assert len(seq) == 10  # 8 + 2
            assert 'XX' in seq
    
    def test_min_spacing(self):
        """Test min_spacing constraint."""
        pool = multiinsertion_scan_op(
            'N' * 20,
            insert_seqs=['AA', 'BB'],
            anchor_pos=10,
            insert_ranges=[(-5, 0), (0, 5)],
            min_spacing=3,  # Require 3bp gap between inserts
            mode='sequential'
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # All combinations should have proper spacing
        for seq in seqs:
            aa_pos = seq.find('AA')
            bb_pos = seq.find('BB')
            if aa_pos < bb_pos:
                # AA ends at aa_pos + 2, BB starts at bb_pos
                assert bb_pos - (aa_pos + 2) >= 3 or aa_pos == -1 or bb_pos == -1
    
    def test_mark_changes(self):
        """Test mark_changes option."""
        pool = multiinsertion_scan_op(
            'NNNNNNNNNN',
            insert_seqs=['XX'],
            anchor_pos=5,
            insert_ranges=[(-2, 3)],
            mark_changes=True,
            mode='sequential'
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # 'XX' should become 'xx'
            assert 'xx' in seq
    
    def test_enforce_order(self):
        """Test enforce_order constraint."""
        # With enforce_order=True, inserts must appear in list order
        pool = multiinsertion_scan_op(
            'N' * 20,
            insert_seqs=['AA', 'BB'],
            anchor_pos=10,
            insert_positions=[[-5, 5], [-5, 5]],  # Both can go to same positions
            enforce_order=True,
            mode='sequential'
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # AA should always appear before BB
        for seq in seqs:
            aa_pos = seq.find('AA')
            bb_pos = seq.find('BB')
            if aa_pos != -1 and bb_pos != -1:
                assert aa_pos < bb_pos


class TestMultiinsertionScanValidation:
    """Tests for input validation."""
    
    def test_empty_inserts_raises(self):
        """Test that empty insert_seqs raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            multiinsertion_scan_op('NNNN', [], anchor_pos=2, insert_ranges=[])
    
    def test_no_valid_combinations_raises(self):
        """Test that impossible constraints raise error."""
        with pytest.raises(ValueError, match="No valid"):
            multiinsertion_scan_op(
                'NNNN',
                insert_seqs=['XXXXXX'],  # Too long
                anchor_pos=2,
                insert_ranges=[(0, 1)],
                insert_or_overwrite='overwrite'
            )


class TestMultiinsertionScanAncestors:
    """Tests for ancestor tracking."""
    
    def test_has_parent_pools(self):
        """Test that multiinsertion_scan has parent pools."""
        pool = multiinsertion_scan_op(
            'NNNNNNNNNN',
            insert_seqs=['XX', 'YY'],
            anchor_pos=5,
            insert_ranges=[(-3, 0), (1, 4)],
        )
        parents = pool.operation.parent_pools
        # 1 background + 2 inserts = 3
        assert len(parents) == 3
