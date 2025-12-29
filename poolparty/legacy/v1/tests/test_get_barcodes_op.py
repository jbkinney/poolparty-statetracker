"""Tests for get_barcodes_op operation."""

import pytest
from poolparty.operations.get_barcodes_op import get_barcodes_op
from poolparty.utils import edit_distance, hamming_distance
from poolparty import Pool


class TestBarcodePool:
    """Tests for get_barcodes_op factory function."""
    
    def test_basic_creation(self):
        """Test basic barcode_pool creation."""
        pool = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        assert isinstance(pool, Pool)
        assert pool.operation.num_states == 10
        assert pool.seq_length == 8
    
    def test_generates_correct_number(self):
        """Test that correct number of barcodes are generated."""
        pool = get_barcodes_op(num_barcodes=50, length=10, seed=42)
        seqs = pool.generate_library(num_complete_iterations=1)
        assert len(seqs) == 50
        assert len(set(seqs)) == 50  # All unique
    
    def test_correct_length(self):
        """Test that barcodes have correct length."""
        pool = get_barcodes_op(num_barcodes=20, length=12, seed=42)
        seqs = pool.generate_library(num_complete_iterations=1)
        for seq in seqs:
            assert len(seq) == 12
    
    def test_dna_alphabet(self):
        """Test that barcodes only contain DNA bases."""
        pool = get_barcodes_op(num_barcodes=30, length=8, seed=42)
        seqs = pool.generate_library(num_complete_iterations=1)
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_min_edit_distance(self):
        """Test that min_edit_distance constraint is satisfied."""
        pool = get_barcodes_op(
            num_barcodes=20,
            length=8,
            min_edit_distance=3,
            seed=42
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Check all pairs
        for i, seq1 in enumerate(seqs):
            for seq2 in seqs[i+1:]:
                dist = edit_distance(seq1, seq2)
                assert dist >= 3, f"Edit distance {dist} < 3 between {seq1} and {seq2}"
    
    def test_min_hamming_distance(self):
        """Test that min_hamming_distance constraint is satisfied."""
        pool = get_barcodes_op(
            num_barcodes=20,
            length=8,
            min_hamming_distance=3,
            seed=42
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Check all pairs
        for i, seq1 in enumerate(seqs):
            for seq2 in seqs[i+1:]:
                dist = hamming_distance(seq1, seq2)
                assert dist >= 3, f"Hamming distance {dist} < 3 between {seq1} and {seq2}"
    
    def test_gc_range_constraint(self):
        """Test that gc_range constraint is satisfied."""
        pool = get_barcodes_op(
            num_barcodes=20,
            length=10,
            gc_range=(0.4, 0.6),
            seed=42
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            gc_count = sum(1 for c in seq if c in 'GC')
            gc_frac = gc_count / len(seq)
            assert 0.4 <= gc_frac <= 0.6, f"GC content {gc_frac} not in [0.4, 0.6] for {seq}"
    
    def test_max_homopolymer_constraint(self):
        """Test that max_homopolymer constraint is satisfied."""
        pool = get_barcodes_op(
            num_barcodes=20,
            length=12,
            max_homopolymer=2,
            seed=42
        )
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # Check no runs > 2
            run_length = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    run_length += 1
                    assert run_length <= 2, f"Run of {run_length} in {seq}"
                else:
                    run_length = 1
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same barcodes."""
        pool1 = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        pool2 = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        
        seqs1 = pool1.generate_library(num_complete_iterations=1)
        seqs2 = pool2.generate_library(num_complete_iterations=1)
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_barcodes(self):
        """Test that different seeds produce different barcodes."""
        pool1 = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        pool2 = get_barcodes_op(num_barcodes=10, length=8, seed=123)
        
        seqs1 = set(pool1.generate_library(num_complete_iterations=1))
        seqs2 = set(pool2.generate_library(num_complete_iterations=1))
        
        # Should have some difference
        assert seqs1 != seqs2


class TestBarcodePoolVariableLength:
    """Tests for variable-length barcode generation."""
    
    def test_variable_length_basic(self):
        """Test variable-length barcode generation."""
        pool = get_barcodes_op(
            num_barcodes=30,
            length=[6, 8, 10],
            padding_char='-',
            seed=42
        )
        
        # Sequence length should be max length
        assert pool.seq_length == 10
        
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Check all have length 10 (with padding)
        for seq in seqs:
            assert len(seq) == 10
        
        # Check we got some of each length (unpadded)
        unpadded_lengths = [len(s.replace('-', '')) for s in seqs]
        assert 6 in unpadded_lengths
        assert 8 in unpadded_lengths
        assert 10 in unpadded_lengths
    
    def test_length_proportions(self):
        """Test length_proportions parameter."""
        pool = get_barcodes_op(
            num_barcodes=100,
            length=[6, 8, 10],
            length_proportions=[0.5, 0.3, 0.2],
            padding_char='-',
            seed=42
        )
        
        seqs = pool.generate_library(num_complete_iterations=1)
        unpadded_lengths = [len(s.replace('-', '')) for s in seqs]
        
        # Count each length
        count_6 = unpadded_lengths.count(6)
        count_8 = unpadded_lengths.count(8)
        count_10 = unpadded_lengths.count(10)
        
        # Should be approximately 50, 30, 20
        assert count_6 == 50
        assert count_8 == 30
        assert count_10 == 20
    
    def test_padding_right(self):
        """Test right padding."""
        pool = get_barcodes_op(
            num_barcodes=10,
            length=[4, 6],
            padding_char='-',
            padding_side='right',
            seed=42
        )
        
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # Padding should be on the right
            if '-' in seq:
                assert seq.endswith('-') or seq.endswith('--')
    
    def test_padding_left(self):
        """Test left padding."""
        pool = get_barcodes_op(
            num_barcodes=10,
            length=[4, 6],
            padding_char='-',
            padding_side='left',
            seed=42
        )
        
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # Padding should be on the left
            if '-' in seq:
                assert seq.startswith('-') or seq.startswith('--')


class TestBarcodePoolValidation:
    """Tests for input validation."""
    
    def test_invalid_num_barcodes(self):
        """Test that invalid num_barcodes raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            get_barcodes_op(num_barcodes=0, length=8)
        
        with pytest.raises(ValueError, match="positive integer"):
            get_barcodes_op(num_barcodes=-5, length=8)
    
    def test_invalid_length(self):
        """Test that invalid length raises error."""
        with pytest.raises(ValueError):
            get_barcodes_op(num_barcodes=10, length=[])
        
        with pytest.raises(ValueError, match="positive integers"):
            get_barcodes_op(num_barcodes=10, length=0)
        
        with pytest.raises(ValueError, match="positive integers"):
            get_barcodes_op(num_barcodes=10, length=-5)
    
    def test_hamming_with_variable_length(self):
        """Test that min_hamming_distance with variable length raises error."""
        with pytest.raises(ValueError, match="hamming"):
            get_barcodes_op(
                num_barcodes=10,
                length=[6, 8],
                min_hamming_distance=3
            )
    
    def test_avoid_without_distance(self):
        """Test that avoid_sequences without avoid_min_distance raises error."""
        with pytest.raises(ValueError, match="avoid_min_distance"):
            get_barcodes_op(
                num_barcodes=10,
                length=8,
                avoid_sequences=['AAAAAAAA']
            )
    
    def test_invalid_gc_range(self):
        """Test that invalid gc_range raises error."""
        with pytest.raises(ValueError):
            get_barcodes_op(num_barcodes=10, length=8, gc_range=(0.6, 0.4))  # min > max
        
        with pytest.raises(ValueError):
            get_barcodes_op(num_barcodes=10, length=8, gc_range=(1.5, 2.0))  # > 1
    
    def test_invalid_padding_side(self):
        """Test that invalid padding_side raises error."""
        with pytest.raises(ValueError, match="padding_side"):
            get_barcodes_op(num_barcodes=10, length=8, padding_side='center')


class TestBarcodePoolAncestors:
    """Tests for ancestor tracking in barcode_pool pools."""
    
    def test_no_parent_pools(self):
        """Test that barcode_pool has no parent pools."""
        pool = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        assert pool.operation.parent_pools == []
    
    def test_ancestors_include_self(self):
        """Test that ancestors include the pool itself."""
        pool = get_barcodes_op(num_barcodes=10, length=8, seed=42)
        assert pool in pool.ancestors
        assert len(pool.ancestors) == 1


class TestBarcodePoolHelpers:
    """Tests for helper functions."""
    
    def test_edit_distance(self):
        """Test edit distance calculation."""
        assert edit_distance('', '') == 0
        assert edit_distance('abc', 'abc') == 0
        assert edit_distance('abc', 'ab') == 1
        assert edit_distance('abc', 'abcd') == 1
        assert edit_distance('abc', 'abd') == 1
        assert edit_distance('kitten', 'sitting') == 3
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        assert hamming_distance('abc', 'abc') == 0
        assert hamming_distance('abc', 'abd') == 1
        assert hamming_distance('abc', 'xyz') == 3
        
        with pytest.raises(ValueError):
            hamming_distance('abc', 'ab')
