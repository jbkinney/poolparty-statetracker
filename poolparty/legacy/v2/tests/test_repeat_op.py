"""Tests for repeat operation."""

import pytest
from poolparty import from_seqs, Pool
from poolparty.operations.repeat_op import repeat, RepeatOp


class TestRepeat:
    """Tests for repeat factory function."""
    
    def test_basic_repeat_two(self):
        """Test basic repetition n=2."""
        pool = from_seqs(['AB'])
        result = repeat(pool, 2)
        
        assert isinstance(result, Pool)
        assert result.seq_length == 4
        result.set_state(0)
        assert result.seq == 'ABAB'
    
    def test_repeat_three(self):
        """Test repetition n=3."""
        pool = from_seqs(['XY'])
        result = repeat(pool, 3)
        
        assert result.seq_length == 6
        result.set_state(0)
        assert result.seq == 'XYXYXY'
    
    def test_repeat_five(self):
        """Test repetition n=5."""
        pool = from_seqs(['A'])
        result = repeat(pool, 5)
        
        assert result.seq_length == 5
        result.set_state(0)
        assert result.seq == 'AAAAA'
    
    def test_large_repeat(self):
        """Test large repeat count n=10."""
        pool = from_seqs(['AB'])
        result = repeat(pool, 10)
        
        assert result.seq_length == 20
        result.set_state(0)
        assert result.seq == 'AB' * 10
    
    def test_very_large_repeat(self):
        """Test very large repeat count n=100."""
        pool = from_seqs(['X'])
        result = repeat(pool, 100)
        
        assert result.seq_length == 100
        result.set_state(0)
        assert result.seq == 'X' * 100
    
    def test_repeat_one(self):
        """Test that repeat(pool, 1) returns equivalent sequence."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, 1)
        
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'ABC'
    
    def test_repeat_zero_raises(self):
        """Test that repeat(pool, 0) raises ValueError."""
        pool = from_seqs(['ABC'])
        with pytest.raises(ValueError, match="is not positive"):
            repeat(pool, 0)
    
    def test_repeat_negative_raises(self):
        """Test that negative n raises ValueError."""
        pool = from_seqs(['ABC'])
        with pytest.raises(ValueError, match="is not positive"):
            repeat(pool, -1)
        
        with pytest.raises(ValueError, match="is not positive"):
            repeat(pool, -5)
    
    def test_seq_length_fixed(self):
        """Test seq_length calculation with fixed-length parent."""
        pool = from_seqs(['ABCD'])  # length 4
        result = repeat(pool, 3)
        assert result.seq_length == 12  # 4 * 3
    
    def test_seq_length_variable(self):
        """Test seq_length is None when parent has variable length."""
        var_pool = from_seqs(['A', 'BB', 'CCC'])
        result = repeat(var_pool, 3)
        assert result.seq_length is None
    
    def test_num_states_always_one(self):
        """Test that RepeatOp always has num_states=1."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, 5)
        assert result.operation.num_states == 1
    
    def test_name_attribute(self):
        """Test name attribute on RepeatOp."""
        pool = from_seqs(['AB'])
        op = RepeatOp(pool, n=3, name='my_repeat')
        result = Pool(operation=op)
        assert result.operation.name == 'my_repeat'
    
    def test_sequential_mode_with_varying_parent(self):
        """Test sequential mode with varying parent pool."""
        parent = from_seqs(['A', 'B', 'C'], mode='sequential')
        result = repeat(parent, 2)
        
        result_df = result.generate_library(num_complete_iterations=1)
        
        # Should get 3 sequences (one for each parent state)
        assert len(result_df) == 3
        assert set(result_df['seq']) == {'AA', 'BB', 'CC'}
    
    def test_repeat_longer_sequences(self):
        """Test repeating longer sequences."""
        pool = from_seqs(['ACGT'])
        result = repeat(pool, 3)
        
        assert result.seq_length == 12
        result.set_state(0)
        assert result.seq == 'ACGTACGTACGT'


class TestRepeatAncestors:
    """Tests for ancestor tracking in repeat pools."""
    
    def test_single_parent_pool(self):
        """Test that repeat has exactly one parent pool."""
        parent = from_seqs(['AAA'])
        result = repeat(parent, 3)
        
        parents = result.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_parent_preserved_through_operation(self):
        """Test that parent reference is preserved."""
        parent = from_seqs(['XYZ'], name='original')
        result = repeat(parent, 2)
        
        assert result.operation.parent_pools[0].operation.name == 'original'
