"""Tests for fixed mode operations."""

import pytest
from poolparty import from_seqs, Pool
from poolparty.operations.slice_op import subseq, SliceOp
from poolparty.operations.concatenate_op import concatenate, ConcatenateOp
from poolparty.operations.repeat_op import repeat, RepeatOp


class TestFixedModeErrors:
    """Test that fixed mode operations raise errors on state/RNG modification."""
    
    def test_slice_op_raises_on_initialize_rng(self):
        """SliceOp should raise RuntimeError if initialize_rng is called."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        
        with pytest.raises(RuntimeError, match="Cannot set RNG seed on fixed-mode operation"):
            result.operation.initialize_rng(42)
    
    def test_slice_op_raises_on_set_state(self):
        """SliceOp should raise RuntimeError if set_state is called."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        
        with pytest.raises(RuntimeError, match="Cannot set state on fixed-mode operation"):
            result.operation.set_state(5)
    
    def test_concatenate_op_raises_on_initialize_rng(self):
        """ConcatenateOp should raise RuntimeError if initialize_rng is called."""
        pool1 = from_seqs(['ABC'])
        pool2 = from_seqs(['DEF'])
        result = concatenate([pool1, pool2])
        
        with pytest.raises(RuntimeError, match="Cannot set RNG seed on fixed-mode operation"):
            result.operation.initialize_rng(42)
    
    def test_concatenate_op_raises_on_set_state(self):
        """ConcatenateOp should raise RuntimeError if set_state is called."""
        pool1 = from_seqs(['ABC'])
        pool2 = from_seqs(['DEF'])
        result = concatenate([pool1, pool2])
        
        with pytest.raises(RuntimeError, match="Cannot set state on fixed-mode operation"):
            result.operation.set_state(5)
    
    def test_repeat_op_raises_on_initialize_rng(self):
        """RepeatOp should raise RuntimeError if initialize_rng is called."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, n=3)
        
        with pytest.raises(RuntimeError, match="Cannot set RNG seed on fixed-mode operation"):
            result.operation.initialize_rng(42)
    
    def test_repeat_op_raises_on_set_state(self):
        """RepeatOp should raise RuntimeError if set_state is called."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, n=3)
        
        with pytest.raises(RuntimeError, match="Cannot set state on fixed-mode operation"):
            result.operation.set_state(5)


class TestFixedModeAttributes:
    """Test that fixed mode operations have correct mode attribute."""
    
    def test_slice_op_mode_is_fixed(self):
        """SliceOp should have mode='fixed'."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        assert result.operation.mode == 'fixed'
    
    def test_concatenate_op_mode_is_fixed(self):
        """ConcatenateOp should have mode='fixed'."""
        pool1 = from_seqs(['ABC'])
        pool2 = from_seqs(['DEF'])
        result = concatenate([pool1, pool2])
        assert result.operation.mode == 'fixed'
    
    def test_repeat_op_mode_is_fixed(self):
        """RepeatOp should have mode='fixed'."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, n=3)
        assert result.operation.mode == 'fixed'


class TestFixedOpsInPool:
    """Test that Pool correctly handles fixed mode operations."""
    
    def test_fixed_ops_list_populated(self):
        """Pool should populate fixed_ops list."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        
        assert len(result.fixed_ops) == 1
        assert result.fixed_ops[0] is result.operation
    
    def test_fixed_ops_excluded_from_random_ops(self):
        """Fixed ops should not be in random_ops list."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        
        assert result.operation not in result.random_ops
    
    def test_fixed_ops_excluded_from_sequential_ops(self):
        """Fixed ops should not be in sequential_ops list."""
        pool = from_seqs(['ABCDE'])
        result = subseq(pool, slice(1, 4))
        
        assert result.operation not in result.sequential_ops
    
    def test_mixed_dag_with_fixed_and_random(self):
        """Test DAG with both fixed and random operations."""
        # Random parent
        parent = from_seqs(['AAA', 'BBB', 'CCC'], mode='random')
        # Fixed slice
        sliced = subseq(parent, slice(0, 2))
        # Fixed repeat
        repeated = repeat(sliced, n=2)
        
        # Check operation modes
        assert parent.operation.mode == 'random'
        assert sliced.operation.mode == 'fixed'
        assert repeated.operation.mode == 'fixed'
        
        # Check op lists in final pool
        assert len(repeated.random_ops) == 1
        assert len(repeated.fixed_ops) == 2
        assert len(repeated.sequential_ops) == 0
    
    def test_mixed_dag_with_fixed_and_sequential(self):
        """Test DAG with both fixed and sequential operations."""
        # Sequential parent
        parent = from_seqs(['AAA', 'BBB', 'CCC'], mode='sequential')
        # Fixed concatenation
        result = concatenate([parent, 'XYZ'])
        
        # Check operation modes
        assert parent.operation.mode == 'sequential'
        assert result.operation.mode == 'fixed'
        
        # Check op lists
        assert len(result.sequential_ops) == 1
        assert len(result.fixed_ops) >= 1  # concatenate + potentially from_seqs for 'XYZ'


class TestFixedOpsIntegration:
    """Integration tests for fixed mode operations in sequence generation."""
    
    def test_slice_produces_correct_sequences(self):
        """Slicing should work correctly with fixed mode."""
        pool = from_seqs(['ABCDEFGH'])
        result = subseq(pool, slice(2, 6))
        
        df = result.generate_library(num_seqs=3, seed=42)
        assert len(df) == 3
        assert all(seq == 'CDEF' for seq in df['seq'])
    
    def test_concatenate_produces_correct_sequences(self):
        """Concatenation should work correctly with fixed mode."""
        pool1 = from_seqs(['ABC'])
        pool2 = from_seqs(['DEF'])
        result = concatenate([pool1, pool2])
        
        df = result.generate_library(num_seqs=3, seed=42)
        assert len(df) == 3
        assert all(seq == 'ABCDEF' for seq in df['seq'])
    
    def test_repeat_produces_correct_sequences(self):
        """Repeat should work correctly with fixed mode."""
        pool = from_seqs(['ABC'])
        result = repeat(pool, n=3)
        
        df = result.generate_library(num_seqs=3, seed=42)
        assert len(df) == 3
        assert all(seq == 'ABCABCABC' for seq in df['seq'])
    
    def test_chained_fixed_ops(self):
        """Multiple chained fixed operations should work."""
        pool = from_seqs(['ABCDEFGHIJ'])
        sliced = subseq(pool, slice(2, 8))  # CDEFGH
        repeated = repeat(sliced, n=2)  # CDEFGHCDEFGH
        
        df = repeated.generate_library(num_seqs=1, seed=0)
        assert df['seq'].iloc[0] == 'CDEFGHCDEFGH'
    
    def test_pool_set_state_skips_fixed_ops(self):
        """Pool.set_state should not error when fixed ops are in DAG."""
        parent = from_seqs(['AAA', 'BBB', 'CCC'], mode='sequential')
        result = subseq(parent, slice(0, 2))
        
        # This should not raise - fixed ops should be skipped
        result.set_state(1)
        
        # Verify parent state was set
        assert parent.operation.state == 1


class TestDunderOpsUseFixedMode:
    """Test that dunder operations (__add__, __mul__, __getitem__) use fixed mode."""
    
    def test_add_uses_fixed_mode(self):
        """Pool.__add__ should create fixed mode ConcatenateOp."""
        pool1 = from_seqs(['ABC'])
        pool2 = from_seqs(['DEF'])
        result = pool1 + pool2
        
        assert result.operation.mode == 'fixed'
    
    def test_mul_uses_fixed_mode(self):
        """Pool.__mul__ should create fixed mode RepeatOp."""
        pool = from_seqs(['ABC'])
        result = pool * 3
        
        assert result.operation.mode == 'fixed'
    
    def test_getitem_uses_fixed_mode(self):
        """Pool.__getitem__ should create fixed mode SliceOp."""
        pool = from_seqs(['ABCDE'])
        result = pool[1:4]
        
        assert result.operation.mode == 'fixed'

