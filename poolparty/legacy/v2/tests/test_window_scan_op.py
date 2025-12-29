"""Tests for WindowScanOp and SlotOp."""

import pytest
import pandas as pd
from poolparty.operations import (
    window_scan, SlotOp, mutation_scan, from_seqs,
    shuffle_scan, deletion_scan, insertion_scan
)
from poolparty.pool import Pool


def get_col(df, col_suffix):
    """Get a column by suffix, handling prefixed column names."""
    for col in df.columns:
        if col == col_suffix or col.endswith(f').{col_suffix}'):
            return df[col]
    raise KeyError(f"Column ending with '{col_suffix}' not found in {list(df.columns)}")


class TestSlotOp:
    """Tests for SlotOp class."""
    
    def test_slot_op_creation(self):
        """Test basic SlotOp creation."""
        slot_op = SlotOp()
        pool = Pool(slot_op)
        assert pool is not None
        assert slot_op.num_states == 1
    
    def test_slot_op_set_contents(self):
        """Test setting contents on SlotOp."""
        slot_op = SlotOp()
        pool = Pool(slot_op)
        
        slot_op.set_contents(['ACGT', 'TGCA', 'GGCC'])
        assert slot_op.num_states == 3
        assert slot_op._contents == ['ACGT', 'TGCA', 'GGCC']
    
    def test_slot_op_generate(self):
        """Test generating from SlotOp."""
        slot_op = SlotOp()
        pool = Pool(slot_op)
        
        slot_op.set_contents(['ACGT'])
        assert pool.seq == 'ACGT'
    
    def test_slot_op_batch_generate(self):
        """Test batch generation from SlotOp."""
        slot_op = SlotOp()
        pool = Pool(slot_op)
        
        slot_op.set_contents(['ACGT', 'TGCA', 'GGCC'])
        df = pool.generate_library(num_seqs=3, init_state=0)
        
        assert len(df) == 3
        assert list(df['seq']) == ['ACGT', 'TGCA', 'GGCC']


class TestWindowScanOpStringTransform:
    """Tests for WindowScanOp with simple string transforms."""
    
    def test_basic_string_transform(self):
        """Test window_scan with a simple string transform."""
        pool = window_scan(
            'ACGTACGT', 
            window_size=4, 
            transform=lambda w: w[::-1],  # Reverse
            mode='sequential'
        )
        
        # 8-char seq with 4-char window: positions 0,1,2,3,4
        assert pool.operation.num_states == 5
        
        df = pool.generate_library(num_seqs=5, init_state=0)
        assert len(df) == 5
        
        # Position 0: reverse 'ACGT' -> 'TGCA', result: 'TGCAACGT'
        assert df.iloc[0]['seq'] == 'TGCAACGT'
        assert get_col(df, 'position').iloc[0] == 0
        assert get_col(df, 'original_window').iloc[0] == 'ACGT'
        assert get_col(df, 'transformed_window').iloc[0] == 'TGCA'
    
    def test_deletion_marked(self):
        """Test window_scan for marked deletion."""
        pool = window_scan(
            'ACGTACGT', 
            window_size=3, 
            transform=lambda w: '-' * len(w),
            mode='sequential'
        )
        
        # 8-char seq with 3-char window: positions 0,1,2,3,4,5
        assert pool.operation.num_states == 6
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        
        # Position 0: '---TACGT'
        assert df.iloc[0]['seq'] == '---TACGT'
        # Position 5: 'ACGTA---'
        assert df.iloc[5]['seq'] == 'ACGTA---'
    
    def test_deletion_actual(self):
        """Test window_scan for actual deletion (removes chars)."""
        pool = window_scan(
            'ACGTACGT', 
            window_size=3, 
            transform=lambda w: '',
            mode='sequential'
        )
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        
        # Position 0: delete 'ACG' -> 'TACGT' (5 chars)
        assert df.iloc[0]['seq'] == 'TACGT'
        assert len(df.iloc[0]['seq']) == 5
    
    def test_insertion_mode(self):
        """Test window_scan with window_size=0 (insertion mode)."""
        pool = window_scan(
            'ACGT', 
            window_size=0, 
            transform=lambda w: 'NNN',
            mode='sequential'
        )
        
        # Insert positions: 0,1,2,3,4 (can insert at end too)
        assert pool.operation.num_states == 5
        
        df = pool.generate_library(num_seqs=5, init_state=0)
        
        # Position 0: 'NNNACGT'
        assert df.iloc[0]['seq'] == 'NNNACGT'
        # Position 4: 'ACGTNNN'
        assert df.iloc[4]['seq'] == 'ACGTNNN'
    
    def test_num_transforms_per_window(self):
        """Test multiple transforms per window position."""
        transform_count = [0]
        
        def counting_transform(w):
            transform_count[0] += 1
            return w.lower()
        
        pool = window_scan(
            'ACGTACGT', 
            window_size=4, 
            transform=counting_transform,
            num_transforms_per_window=3,
            mode='sequential'
        )
        
        # 5 positions × 3 transforms = 15 states
        assert pool.operation.num_states == 15
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        positions = get_col(df, 'position')
        
        # First 3 should all be at position 0
        assert positions.iloc[0] == 0
        assert positions.iloc[1] == 0
        assert positions.iloc[2] == 0
        # Next 3 at position 1
        assert positions.iloc[3] == 1
        assert positions.iloc[4] == 1
        assert positions.iloc[5] == 1


class TestWindowScanOpPositions:
    """Tests for WindowScanOp position handling."""
    
    def test_range_based_positions(self):
        """Test range-based position specification."""
        pool = window_scan(
            'ACGTACGTACGT',  # 12 chars
            window_size=4,
            transform=lambda w: w.lower(),
            start=2,
            end=8,
            step_size=2,
            mode='sequential'
        )
        
        # Positions: 2, 4 (6 would put window end at 10 > 8)
        # Actually: range(2, 8-4+1, 2) = range(2, 5, 2) = [2, 4]
        assert pool.operation.num_states == 2
        
        df = pool.generate_library(num_seqs=2, init_state=0)
        positions = get_col(df, 'position')
        assert positions.iloc[0] == 2
        assert positions.iloc[1] == 4
    
    def test_explicit_positions(self):
        """Test explicit position list."""
        pool = window_scan(
            'ACGTACGTACGT',  # 12 chars
            window_size=4,
            transform=lambda w: w.lower(),
            positions=[0, 4, 8],
            mode='sequential'
        )
        
        assert pool.operation.num_states == 3
        
        df = pool.generate_library(num_seqs=3, init_state=0)
        positions = get_col(df, 'position')
        assert list(positions) == [0, 4, 8]
    
    def test_invalid_position_raises(self):
        """Test that invalid positions raise errors."""
        with pytest.raises(ValueError):
            window_scan(
                'ACGT',
                window_size=4,
                transform=lambda w: w,
                positions=[1],  # Window would exceed sequence
                mode='sequential'
            )
    
    def test_mixed_params_raises(self):
        """Test that mixing range and positions raises error."""
        with pytest.raises(ValueError):
            window_scan(
                'ACGTACGT',
                window_size=4,
                transform=lambda w: w,
                start=0,
                positions=[0, 2],
                mode='sequential'
            )


class TestWindowScanOpPoolTransform:
    """Tests for WindowScanOp with Pool-based transforms."""
    
    def test_pool_transform_sequential(self):
        """Test Pool-based transform with sequential mode."""
        pool = window_scan(
            'ACGTACGT',  # 8 chars
            window_size=4,
            transform=lambda slot: mutation_scan(slot, num_mutations=1, mode='sequential'),
            mode='sequential'
        )
        
        # 5 positions × (4 positions × 3 mutations = 12 states) = 60 states
        assert pool.operation.num_states == 60
        assert pool.operation._is_pool_transform
        assert pool.operation._transform_is_sequential
    
    def test_pool_transform_random(self):
        """Test Pool-based transform with random mode."""
        pool = window_scan(
            'ACGTACGT',
            window_size=4,
            transform=lambda slot: mutation_scan(slot, num_mutations=1, mode='random'),
            num_transforms_per_window=3,
            mode='sequential'
        )
        
        # 5 positions × 3 random samples = 15 states
        assert pool.operation.num_states == 15
        assert pool.operation._is_pool_transform
        assert not pool.operation._transform_is_sequential
    
    def test_pool_transform_generates_valid_sequences(self):
        """Test that Pool-based transform generates valid sequences."""
        pool = window_scan(
            'AAAAAAAA',  # Easy to verify mutations
            window_size=4,
            transform=lambda slot: mutation_scan(slot, num_mutations=1, mode='sequential'),
            mode='sequential'
        )
        
        df = pool.generate_library(num_seqs=12, init_state=0)
        
        # All sequences should have exactly 1 non-A character in first 4 positions
        for i, row in df.iterrows():
            seq = row['seq']
            # The mutation is in the window (positions 0-3)
            window_part = seq[:4]
            non_a = sum(1 for c in window_part if c != 'A')
            assert non_a == 1, f"Row {i}: expected 1 mutation, got {non_a} in '{seq}'"
    
    def test_pool_transform_design_cards(self):
        """Test that Pool-based transform propagates design cards."""
        pool = window_scan(
            'AAAAAAAA',
            window_size=4,
            transform=lambda slot: mutation_scan(slot, num_mutations=1, mode='sequential'),
            mode='sequential'
        )
        
        df = pool.generate_library(num_seqs=12, init_state=0)
        
        # Should have WindowScanOp's cards (may be prefixed)
        position_cols = [c for c in df.columns if 'position' in c.lower()]
        assert len(position_cols) > 0
        
        original_window_cols = [c for c in df.columns if 'original_window' in c]
        assert len(original_window_cols) > 0
        
        transformed_window_cols = [c for c in df.columns if 'transformed_window' in c]
        assert len(transformed_window_cols) > 0
        
        # Should have transform's cards (prefixed with 'transform.')
        transform_cols = [c for c in df.columns if 'transform.' in c]
        assert len(transform_cols) > 0


class TestWindowScanOpDesignCards:
    """Tests for WindowScanOp design card functionality."""
    
    def test_basic_design_cards(self):
        """Test that basic design cards are present."""
        pool = window_scan(
            'ACGTACGT',
            window_size=4,
            transform=lambda w: w[::-1],
            mode='sequential'
        )
        
        df = pool.generate_library(num_seqs=5, init_state=0)
        
        # Check that design card columns exist (may be prefixed)
        expected_suffixes = ['position', 'window_size', 'transform_idx', 
                             'original_window', 'transformed_window']
        for suffix in expected_suffixes:
            matching_cols = [c for c in df.columns if c == suffix or c.endswith(f').{suffix}')]
            assert len(matching_cols) > 0, f"Missing column ending with: {suffix}"
    
    def test_design_card_values(self):
        """Test that design card values are correct."""
        pool = window_scan(
            'ACGTACGT',
            window_size=4,
            transform=lambda w: w.lower(),
            mode='sequential'
        )
        
        df = pool.generate_library(num_seqs=5, init_state=0)
        
        window_sizes = get_col(df, 'window_size')
        transform_idxs = get_col(df, 'transform_idx')
        original_windows = get_col(df, 'original_window')
        transformed_windows = get_col(df, 'transformed_window')
        
        for i in range(len(df)):
            # window_size should always be 4
            assert window_sizes.iloc[i] == 4
            # transform_idx should be 0 (only 1 transform per window)
            assert transform_idxs.iloc[i] == 0
            # transformed should be lowercase of original
            assert transformed_windows.iloc[i] == original_windows.iloc[i].lower()


class TestWindowScanOpWithPoolParent:
    """Tests for WindowScanOp with Pool as background."""
    
    def test_pool_parent(self):
        """Test window_scan with Pool as background."""
        bg_pool = from_seqs(['ACGTACGT', 'TGCATGCA'], mode='sequential')
        
        pool = window_scan(
            bg_pool,
            window_size=4,
            transform=lambda w: w[::-1],
            mode='sequential'
        )
        
        # Should work with Pool parent
        df = pool.generate_library(num_seqs=5, init_state=0)
        assert len(df) == 5


class TestWindowScanOpRandomMode:
    """Tests for WindowScanOp in random mode."""
    
    def test_random_mode_basic(self):
        """Test window_scan in random mode."""
        pool = window_scan(
            'ACGTACGT',
            window_size=4,
            transform=lambda w: w[::-1],
            mode='random'
        )
        
        # Should generate valid sequences
        df = pool.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        
        # All sequences should be valid
        for seq in df['seq']:
            assert len(seq) == 8
            assert set(seq).issubset({'A', 'C', 'G', 'T'})
    
    def test_random_mode_reproducible(self):
        """Test that random mode is reproducible with seed."""
        pool = window_scan(
            'ACGTACGT',
            window_size=4,
            transform=lambda w: w[::-1],
            mode='random'
        )
        
        df1 = pool.generate_library(num_seqs=10, init_state=0, seed=42)
        df2 = pool.generate_library(num_seqs=10, init_state=0, seed=42)
        
        assert list(df1['seq']) == list(df2['seq'])


class TestShuffleScan:
    """Tests for shuffle_scan helper function."""
    
    def test_basic_shuffle_scan(self):
        """Test basic shuffle_scan functionality."""
        pool = shuffle_scan('ACGTACGT', shuffle_size=4, seed=42)
        
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5
        
        # All sequences should have same length
        for seq in df['seq']:
            assert len(seq) == 8
    
    def test_shuffle_scan_sequential(self):
        """Test shuffle_scan in sequential mode."""
        pool = shuffle_scan('ACGTACGT', shuffle_size=4, mode='sequential', seed=42)
        
        # Should have 5 positions
        assert pool.operation.num_states == 5
    
    def test_shuffle_preserves_characters(self):
        """Test that shuffle preserves the characters in the window."""
        pool = shuffle_scan('ACGT', shuffle_size=4, seed=42)
        
        df = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in df['seq']:
            # The sequence should have the same characters as the original
            assert sorted(seq) == sorted('ACGT')


class TestDeletionScan:
    """Tests for deletion_scan helper function."""
    
    def test_marked_deletion(self):
        """Test deletion_scan with marked deletions."""
        pool = deletion_scan('ACGTACGT', deletion_size=3, mark_deletions=True, mode='sequential')
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        
        # All sequences should have same length
        for seq in df['seq']:
            assert len(seq) == 8
            assert '---' in seq  # Should have deletion markers
    
    def test_actual_deletion(self):
        """Test deletion_scan with actual deletions."""
        pool = deletion_scan('ACGTACGT', deletion_size=3, mark_deletions=False, mode='sequential')
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        
        # All sequences should be shorter
        for seq in df['seq']:
            assert len(seq) == 5  # 8 - 3 = 5
    
    def test_deletion_scan_positions(self):
        """Test that deletions happen at correct positions."""
        pool = deletion_scan('ACGTACGT', deletion_size=3, mark_deletions=True, mode='sequential')
        
        df = pool.generate_library(num_seqs=1, init_state=0)
        
        # First position should delete first 3 chars
        assert df.iloc[0]['seq'] == '---TACGT'


class TestInsertionScan:
    """Tests for insertion_scan helper function."""
    
    def test_overwrite_mode(self):
        """Test insertion_scan with overwrite mode."""
        pool = insertion_scan('ACGTACGT', insert='NNN', overwrite=True, mode='sequential')
        
        df = pool.generate_library(num_seqs=6, init_state=0)
        
        # All sequences should have same length
        for seq in df['seq']:
            assert len(seq) == 8
            assert 'NNN' in seq
    
    def test_insert_mode(self):
        """Test insertion_scan with insert mode (no overwrite)."""
        pool = insertion_scan('ACGT', insert='NNN', overwrite=False, mode='sequential')
        
        df = pool.generate_library(num_seqs=5, init_state=0)
        
        # All sequences should be longer
        for seq in df['seq']:
            assert len(seq) == 7  # 4 + 3 = 7
            assert 'NNN' in seq
    
    def test_insertion_scan_positions(self):
        """Test that insertions happen at correct positions."""
        pool = insertion_scan('ACGT', insert='NNN', overwrite=True, mode='sequential')
        
        df = pool.generate_library(num_seqs=1, init_state=0)
        
        # First position should replace first 3 chars
        assert df.iloc[0]['seq'] == 'NNNT'

