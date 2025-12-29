"""Tests for shuffle operation."""

import pytest
from collections import Counter
from poolparty.operations.shuffle_op import shuffle_op
from poolparty import Pool, from_seqs_op


class TestShuffle:
    """Tests for shuffle factory function."""
    
    def test_basic_creation(self):
        """Test basic shuffle pool creation."""
        pool = shuffle_op('ACGT')
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
        assert pool.operation.num_states == 24  # 4!
    
    def test_preserves_length(self):
        """Test that shuffled sequences have same length."""
        pool = shuffle_op('ACGTACGT')
        seqs = pool.generate_library(num_seqs=10, seed=42)
        for seq in seqs:
            assert len(seq) == 8
    
    def test_preserves_composition(self):
        """Test that shuffled sequences have same character composition."""
        original = 'AACCGGTT'
        pool = shuffle_op(original)
        seqs = pool.generate_library(num_seqs=20, seed=42)
        
        original_counts = Counter(original)
        for seq in seqs:
            assert Counter(seq) == original_counts
    
    def test_sequential_enumerates_all(self):
        """Test that sequential mode enumerates all permutations."""
        pool = shuffle_op('ABC', mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # 3! = 6 permutations
        assert len(seqs) == 6
        assert len(set(seqs)) == 6  # All unique
        
        # All permutations of ABC
        expected = {'ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA'}
        assert set(seqs) == expected
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = shuffle_op('ACGTACGT')
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_region_shuffle_op(self):
        """Test shuffling a specific region."""
        pool = shuffle_op('AABBCCDD', start=2, end=6, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Region BBCC = 4! = 24 permutations
        assert len(seqs) == 24
        
        # Flanks should be preserved
        for seq in seqs:
            assert seq.startswith('AA')
            assert seq.endswith('DD')


class TestShuffleWithFlanks:
    """Tests for shuffle_flanks mode."""
    
    def test_shuffle_flanks_basic(self):
        """Test basic flank shuffling."""
        pool = shuffle_op('AABBCCDD', start=2, end=6, shuffle_flanks=True, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Left flank AA = 2!, right flank DD = 2!
        # Total = 2! × 2! = 4
        assert len(seqs) == 4
        
        # Region should be preserved
        for seq in seqs:
            assert seq[2:6] == 'BBCC'
    
    def test_shuffle_flanks_preserves_region(self):
        """Test that shuffle_flanks keeps region fixed."""
        pool = shuffle_op('XXXYYYZZZ', start=3, end=6, shuffle_flanks=True)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert seq[3:6] == 'YYY'


class TestShuffleDinucleotide:
    """Tests for dinucleotide-preserving shuffle."""
    
    def test_dinucleotide_mode(self):
        """Test dinucleotide-preserving shuffle."""
        pool = shuffle_op('ACGTACGTACGT', preserve_dinucleotides=True)
        assert pool.operation.num_states == -1
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        # All sequences should preserve composition
        original_counts = Counter('ACGTACGTACGT')
        for seq in seqs:
            assert Counter(seq) == original_counts
    
    def test_dinucleotide_sequential_raises(self):
        """Test that dinucleotide mode with sequential raises error."""
        with pytest.raises(ValueError, match="random"):
            shuffle_op('ACGT', preserve_dinucleotides=True, mode='sequential')


class TestShuffleMarkChanges:
    """Tests for mark_changes option."""
    
    def test_mark_changes(self):
        """Test that mark_changes swaps case."""
        pool = shuffle_op('ACGT', mark_changes=True, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # All should be lowercase (original was uppercase, swapcase makes lowercase)
        for seq in seqs:
            assert seq.islower()
    
    def test_mark_changes_region_only(self):
        """Test mark_changes only affects shuffled region."""
        pool = shuffle_op('AABBCCDD', start=2, end=6, mark_changes=True, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            # Flanks should be unchanged (uppercase)
            assert seq[:2] == 'AA'
            assert seq[6:] == 'DD'
            # Region should be lowercase
            assert seq[2:6].islower()


class TestShuffleValidation:
    """Tests for input validation."""
    
    def test_negative_start_raises(self):
        """Test that negative start raises error."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            shuffle_op('ACGT', start=-1)
    
    def test_end_exceeds_length_raises(self):
        """Test that end > length raises error."""
        with pytest.raises(ValueError, match="cannot exceed"):
            shuffle_op('ACGT', end=10)
    
    def test_start_exceeds_end_raises(self):
        """Test that start > end raises error."""
        with pytest.raises(ValueError, match="must be <="):
            shuffle_op('ACGT', start=3, end=1)


class TestShuffleAncestors:
    """Tests for ancestor tracking in shuffle pools."""
    
    def test_has_parent_pool_from_string(self):
        """Test that shuffle from string has parent pool."""
        pool = shuffle_op('ACGT')
        parents = pool.operation.parent_pools
        assert len(parents) == 1
    
    def test_has_parent_pool_from_pool(self):
        """Test that shuffle from Pool has correct parent."""
        parent = from_seqs_op(['ACGT', 'TGCA'])
        pool = shuffle_op(parent)
        
        parents = pool.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_ancestors_include_all(self):
        """Test that ancestors include both self and parent."""
        parent = from_seqs_op(['ACGT'])
        pool = shuffle_op(parent)
        
        assert pool in pool.ancestors
        assert parent in pool.ancestors
        assert len(pool.ancestors) == 2
