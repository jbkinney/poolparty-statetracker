"""Tests for subseq operation."""

import pytest
from poolparty.operations.subseq_op import subseq_op
from poolparty import Pool, from_seqs_op


class TestSubseq:
    """Tests for subseq factory function."""
    
    def test_basic_creation(self):
        """Test basic subseq pool creation."""
        pool = subseq_op('ACGTACGTACGT', width=4)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
        assert pool.operation.num_states == 9  # 12 - 4 + 1 = 9 positions
    
    def test_extracts_correct_width(self):
        """Test that extracted subsequences have correct width."""
        pool = subseq_op('ACGTACGTACGT', width=5)
        seqs = pool.generate_library(num_seqs=10, seed=42)
        for seq in seqs:
            assert len(seq) == 5
    
    def test_sequential_enumerates_all(self):
        """Test that sequential mode enumerates all subsequences."""
        pool = subseq_op('ABCDEFGH', width=3, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # 8 - 3 + 1 = 6 positions
        assert len(seqs) == 6
        expected = ['ABC', 'BCD', 'CDE', 'DEF', 'EFG', 'FGH']
        assert seqs == expected
    
    def test_step_size(self):
        """Test step_size parameter."""
        pool = subseq_op('ABCDEFGHIJ', width=2, step_size=3, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 0, 3, 6 (10 - 2 + 1 = 9, range(0, 9, 3) = [0, 3, 6])
        assert len(seqs) == 3
        expected = ['AB', 'DE', 'GH']
        assert seqs == expected
    
    def test_start_parameter(self):
        """Test start parameter."""
        pool = subseq_op('ABCDEFGHIJ', width=2, start=3, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 3, 4, 5, 6, 7, 8 (range(3, 9))
        assert len(seqs) == 6
        assert seqs[0] == 'DE'  # Position 3
    
    def test_end_parameter(self):
        """Test end parameter."""
        pool = subseq_op('ABCDEFGHIJ', width=2, end=6, mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Positions: 0, 1, 2, 3, 4 (range(0, 6-2+1) = range(0, 5))
        assert len(seqs) == 5
        assert seqs[-1] == 'EF'  # Position 4
    
    def test_explicit_positions(self):
        """Test explicit positions parameter."""
        pool = subseq_op('ABCDEFGHIJ', width=2, positions=[0, 3, 7], mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 3
        expected = ['AB', 'DE', 'HI']
        assert seqs == expected
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = subseq_op('ACGTACGTACGT', width=4)
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_with_pool_parent(self):
        """Test subseq with Pool as parent."""
        parent = from_seqs_op(['AAAAAA', 'TTTTTT'], mode='sequential')
        pool = subseq_op(parent, width=3, mode='sequential')
        
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Parent has 2 states, subseq has 4 positions (6-3+1)
        # Total: 2 × 4 = 8
        assert len(seqs) == 8
        
        # Check we got subsequences from both parent sequences
        assert 'AAA' in seqs
        assert 'TTT' in seqs


class TestSubseqValidation:
    """Tests for input validation."""
    
    def test_width_zero_raises(self):
        """Test that width=0 raises error."""
        with pytest.raises(ValueError, match="width must be > 0"):
            subseq_op('ACGT', width=0)
    
    def test_width_negative_raises(self):
        """Test that negative width raises error."""
        with pytest.raises(ValueError, match="width must be > 0"):
            subseq_op('ACGT', width=-1)
    
    def test_width_exceeds_length_raises(self):
        """Test that width > seq length raises error."""
        with pytest.raises(ValueError, match="cannot be longer"):
            subseq_op('ACGT', width=10)
    
    def test_both_interfaces_raises(self):
        """Test that using both interfaces raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            subseq_op('ACGT', width=2, start=1, positions=[0, 1])
    
    def test_empty_positions_raises(self):
        """Test that empty positions raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            subseq_op('ACGT', width=2, positions=[])
    
    def test_invalid_position_raises(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError, match="invalid"):
            subseq_op('ACGT', width=2, positions=[3])  # Position 3 + width 2 = 5 > 4
    
    def test_duplicate_positions_raises(self):
        """Test that duplicate positions raise error."""
        with pytest.raises(ValueError, match="duplicates"):
            subseq_op('ACGTACGT', width=2, positions=[0, 1, 0])
    
    def test_negative_start_raises(self):
        """Test that negative start raises error."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            subseq_op('ACGT', width=2, start=-1)
    
    def test_zero_step_raises(self):
        """Test that step_size=0 raises error."""
        with pytest.raises(ValueError, match="step_size must be > 0"):
            subseq_op('ACGT', width=2, step_size=0)


class TestSubseqAncestors:
    """Tests for ancestor tracking in subseq pools."""
    
    def test_has_parent_pool_from_string(self):
        """Test that subseq from string has parent pool (from_seqs wrapper)."""
        pool = subseq_op('ACGTACGT', width=3)
        parents = pool.operation.parent_pools
        assert len(parents) == 1
    
    def test_has_parent_pool_from_pool(self):
        """Test that subseq from Pool has the correct parent."""
        parent = from_seqs_op(['ACGTACGT'])
        pool = subseq_op(parent, width=3)
        
        parents = pool.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_ancestors_include_all(self):
        """Test that ancestors include both self and parent."""
        parent = from_seqs_op(['ACGTACGT'])
        pool = subseq_op(parent, width=3)
        
        assert pool in pool.ancestors
        assert parent in pool.ancestors
        assert len(pool.ancestors) == 2
