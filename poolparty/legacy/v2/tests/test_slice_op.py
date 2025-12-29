"""Tests for slice operation."""

import pytest
from poolparty import from_seqs, Pool
from poolparty.operations.slice_op import subseq, SliceOp


class TestSubseq:
    """Tests for subseq factory function."""
    
    def test_single_positive_index(self):
        """Test single positive index."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, 0)
        result.set_state(0)
        assert result.seq == 'A'
        assert result.seq_length == 1
        
        result = subseq(pool, 2)
        result.set_state(0)
        assert result.seq == 'C'
        
        result = subseq(pool, 4)
        result.set_state(0)
        assert result.seq == 'E'
    
    def test_single_negative_index(self):
        """Test single negative index."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, -1)
        result.set_state(0)
        assert result.seq == 'E'
        assert result.seq_length == 1
        
        result = subseq(pool, -2)
        result.set_state(0)
        assert result.seq == 'D'
        
        result = subseq(pool, -5)
        result.set_state(0)
        assert result.seq == 'A'
    
    def test_slice_range(self):
        """Test slice range [start:stop]."""
        pool = from_seqs(['ABCDEFGH'])
        
        result = subseq(pool, slice(1, 4))
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'BCD'
        
        result = subseq(pool, slice(0, 2))
        assert result.seq_length == 2
        result.set_state(0)
        assert result.seq == 'AB'
        
        result = subseq(pool, slice(5, 8))
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'FGH'
    
    def test_slice_with_step(self):
        """Test slice with step [::step]."""
        pool = from_seqs(['ABCDEF'])
        
        result = subseq(pool, slice(None, None, 2))
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'ACE'
        
        result = subseq(pool, slice(None, None, 3))
        assert result.seq_length == 2
        result.set_state(0)
        assert result.seq == 'AD'
    
    def test_slice_start_with_step(self):
        """Test slice with start and step [start::step]."""
        pool = from_seqs(['ABCDEFGH'])
        
        result = subseq(pool, slice(1, None, 2))
        assert result.seq_length == 4
        result.set_state(0)
        assert result.seq == 'BDFH'
    
    def test_negative_slice(self):
        """Test negative slice [-n:]."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, slice(-3, None))
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'CDE'
        
        result = subseq(pool, slice(-1, None))
        assert result.seq_length == 1
        result.set_state(0)
        assert result.seq == 'E'
    
    def test_full_slice(self):
        """Test full slice [:] returns copy of full sequence."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, slice(None, None))
        assert result.seq_length == 5
        result.set_state(0)
        assert result.seq == 'ABCDE'
    
    def test_empty_slice(self):
        """Test empty slice result [n:n]."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, slice(2, 2))
        assert result.seq_length == 0
        result.set_state(0)
        assert result.seq == ''
        
        result = subseq(pool, slice(5, 5))
        assert result.seq_length == 0
        result.set_state(0)
        assert result.seq == ''
    
    def test_seq_length_fixed_single_index(self):
        """Test seq_length is 1 for single index."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, 2)
        assert result.seq_length == 1
    
    def test_seq_length_fixed_slice(self):
        """Test seq_length calculation with fixed-length parent and slice."""
        pool = from_seqs(['ABCDEFGHIJ'])  # length 10
        
        result = subseq(pool, slice(2, 7))
        assert result.seq_length == 5
        
        result = subseq(pool, slice(0, 10, 2))
        assert result.seq_length == 5
    
    def test_seq_length_variable(self):
        """Test seq_length is None when parent has variable length."""
        var_pool = from_seqs(['A', 'BB', 'CCC'])
        result = subseq(var_pool, slice(0, 2))
        assert result.seq_length is None
    
    def test_num_states_always_one(self):
        """Test that SliceOp always has num_states=1."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, 2)
        assert result.operation.num_states == 1
        
        result = subseq(pool, slice(1, 4))
        assert result.operation.num_states == 1
    
    def test_name_attribute(self):
        """Test name attribute on SliceOp."""
        pool = from_seqs(['ABCDE'])
        op = SliceOp(pool, key=slice(1, 4), name='my_slice')
        result = Pool(operation=op)
        assert result.operation.name == 'my_slice'
    
    def test_sequential_mode_with_varying_parent(self):
        """Test sequential mode with varying parent pool."""
        parent = from_seqs(['AAAA', 'BBBB', 'CCCC'], mode='sequential')
        result = subseq(parent, slice(1, 3))
        
        result_df = result.generate_library(num_complete_iterations=1)
        
        assert len(result_df) == 3
        assert set(result_df['seq']) == {'AA', 'BB', 'CC'}
    
    def test_reverse_slice(self):
        """Test reverse slice [::-1]."""
        pool = from_seqs(['ABCDE'])
        
        result = subseq(pool, slice(None, None, -1))
        assert result.seq_length == 5
        result.set_state(0)
        assert result.seq == 'EDCBA'


class TestSliceAncestors:
    """Tests for ancestor tracking in slice pools."""
    
    def test_single_parent_pool(self):
        """Test that slice has exactly one parent pool."""
        parent = from_seqs(['ABCDE'])
        result = subseq(parent, slice(1, 4))
        
        parents = result.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_parent_preserved_through_operation(self):
        """Test that parent reference is preserved."""
        parent = from_seqs(['ABCDE'], name='original')
        result = subseq(parent, 2)
        
        assert result.operation.parent_pools[0].operation.name == 'original'
    
    def test_single_index_parent_tracking(self):
        """Test parent tracking with single index."""
        parent = from_seqs(['XYZ'])
        result = subseq(parent, 1)
        
        parents = result.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
