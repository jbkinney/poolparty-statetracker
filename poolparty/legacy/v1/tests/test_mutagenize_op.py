"""Tests for mutagenize operation."""

import pytest
from collections import Counter
from poolparty.operations.mutagenize_op import mutagenize_op
from poolparty import Pool, from_seqs_op


class TestMutagenize:
    """Tests for mutagenize factory function."""
    
    def test_basic_creation(self):
        """Test basic mutagenize pool creation."""
        pool = mutagenize_op('ACGTACGT')
        assert isinstance(pool, Pool)
        assert pool.seq_length == 8
        assert pool.operation.num_states == -1
    
    def test_preserves_length(self):
        """Test that mutated sequences have same length."""
        pool = mutagenize_op('ACGTACGT', mutation_rate=0.5)
        seqs = pool.generate_library(num_seqs=20, seed=42)
        for seq in seqs:
            assert len(seq) == 8
    
    def test_uses_alphabet(self):
        """Test that mutations use the specified alphabet."""
        pool = mutagenize_op('AAAA', alphabet='dna', mutation_rate=0.5)
        seqs = pool.generate_library(num_seqs=100, seed=42)
        
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences from the same pool."""
        pool = mutagenize_op('ACGTACGT', mutation_rate=0.3)
        
        # Same pool, same seed should produce same sequences
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_mutation_rate_zero(self):
        """Test that mutation_rate=0 produces original sequence."""
        pool = mutagenize_op('ACGTACGT', mutation_rate=0.0)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert seq == 'ACGTACGT'
    
    def test_mutation_rate_one(self):
        """Test that mutation_rate=1 mutates all positions."""
        pool = mutagenize_op('AAAA', mutation_rate=1.0, alphabet='dna')
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            # All positions should be different from original
            assert all(c != 'A' for c in seq)
    
    def test_position_specific_rates(self):
        """Test position-specific mutation rates."""
        # High rate at position 0, zero elsewhere
        rates = [1.0, 0.0, 0.0, 0.0]
        pool = mutagenize_op('AAAA', mutation_rate=rates, alphabet='dna')
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            # Position 0 should always be mutated
            assert seq[0] != 'A'
            # Other positions should not be mutated
            assert seq[1:] == 'AAA'
    
    def test_mark_changes(self):
        """Test mark_changes option."""
        pool = mutagenize_op('AAAA', mutation_rate=1.0, mark_changes=True)
        seqs = pool.generate_library(num_seqs=5, seed=42)
        
        for seq in seqs:
            # Original was uppercase, mutated should be lowercase
            assert seq.islower()


class TestMutagenizeValidation:
    """Tests for input validation."""
    
    def test_sequential_mode_raises(self):
        """Test that sequential mode raises error."""
        with pytest.raises(ValueError, match="random"):
            mutagenize_op('ACGT', mode='sequential')
    
    def test_negative_rate_raises(self):
        """Test that negative mutation_rate raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            mutagenize_op('ACGT', mutation_rate=-0.1)
    
    def test_rate_above_one_raises(self):
        """Test that mutation_rate > 1 raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            mutagenize_op('ACGT', mutation_rate=1.5)
    
    def test_array_length_mismatch_raises(self):
        """Test that mismatched rate array length raises error."""
        with pytest.raises(ValueError, match="must match sequence length"):
            mutagenize_op('ACGT', mutation_rate=[0.1, 0.2])  # Length 2 != 4


class TestMutagenizeAncestors:
    """Tests for ancestor tracking in mutagenize pools."""
    
    def test_has_parent_pool_from_string(self):
        """Test that mutagenize from string has parent pool."""
        pool = mutagenize_op('ACGT')
        parents = pool.operation.parent_pools
        assert len(parents) == 1
    
    def test_has_parent_pool_from_pool(self):
        """Test that mutagenize from Pool has correct parent."""
        parent = from_seqs_op(['ACGT', 'TGCA'])
        pool = mutagenize_op(parent)
        
        parents = pool.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_ancestors_include_all(self):
        """Test that ancestors include both self and parent."""
        parent = from_seqs_op(['ACGT'])
        pool = mutagenize_op(parent)
        
        assert pool in pool.ancestors
        assert parent in pool.ancestors
        assert len(pool.ancestors) == 2
