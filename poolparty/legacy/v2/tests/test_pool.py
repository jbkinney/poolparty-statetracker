"""Tests for Pool class and composite operations."""

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from poolparty import Pool, from_seqs, get_kmers


class TestPoolBasics:
    """Tests for basic Pool functionality."""
    
    def test_pool_str(self):
        """Test that str(pool) returns current sequence."""
        pool = from_seqs(['HELLO'])
        assert str(pool) == 'HELLO'
    
    def test_pool_repr(self):
        """Test pool repr."""
        pool = from_seqs(['AAA'], name='test')
        repr_str = repr(pool)
        assert 'Pool' in repr_str
        assert 'test' in repr_str
    
    def test_mode_validation(self):
        """Test that invalid mode raises error."""
        with pytest.raises(BeartypeCallHintParamViolation, match="violates type hint"):
            from_seqs(['AAA'], mode='invalid')
    
    def test_set_mode(self):
        """Test set_mode method on operation."""
        pool = from_seqs(['AAA', 'TTT'])
        assert pool.operation.mode == 'random'
        
        pool.operation.set_mode('sequential')
        assert pool.operation.mode == 'sequential'
        
        # beartype enforces Literal type hint, raising BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            pool.operation.set_mode('invalid')


class TestConcatOperation:
    """Tests for concatenation (+) operator."""
    
    def test_pool_plus_string(self):
        """Test Pool + string concatenation."""
        pool = from_seqs(['AAA'])
        result = pool + 'TTT'
        
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_sequential_op_states(0)
        assert result.seq == 'AAATTT'
    
    def test_string_plus_pool(self):
        """Test string + Pool concatenation."""
        pool = from_seqs(['AAA'])
        result = 'TTT' + pool
        
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_sequential_op_states(0)
        assert result.seq == 'TTTAAA'
    
    def test_pool_plus_pool(self):
        """Test Pool + Pool concatenation."""
        pool1 = from_seqs(['AAA'])
        pool2 = from_seqs(['TTT'])
        result = pool1 + pool2
        
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_sequential_op_states(0)
        assert result.seq == 'AAATTT'
    
    def test_concat_with_varying_pools(self):
        """Test concatenation with pools that have multiple states."""
        pool1 = from_seqs(['AA', 'CC'], mode='sequential')
        pool2 = from_seqs(['TT', 'GG'], mode='sequential')
        result = pool1 + pool2
        
        # Both pools in sequential mode
        result_df = result.generate_library(num_complete_iterations=1)
        
        # Should get all 4 combinations
        assert len(result_df) == 4
        assert set(result_df['seq']) == {'AATT', 'AAGG', 'CCTT', 'CCGG'}
    
    def test_concat_num_states(self):
        """Test that concat pool has correct num_internal_states."""
        pool1 = from_seqs(['A', 'C'])
        pool2 = from_seqs(['T', 'G'])
        result = pool1 + pool2
        
        # Concat itself has 1 internal state
        assert result.operation.num_states == 1
    
    def test_multiple_concat(self):
        """Test chaining multiple concatenations."""
        pool = from_seqs(['A'])
        result = pool + 'B' + 'C' + 'D'
        
        assert result.seq_length == 4
        result.set_state(0)
        assert result.seq == 'ABCD'


class TestRepeatOperation:
    """Tests for repetition (*) operator."""
    
    def test_pool_times_int(self):
        """Test Pool * int repetition."""
        pool = from_seqs(['AB'])
        result = pool * 3
        
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_sequential_op_states(0)
        assert result.seq == 'ABABAB'
    
    def test_int_times_pool(self):
        """Test int * Pool repetition."""
        pool = from_seqs(['AB'])
        result = 3 * pool
        
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_sequential_op_states(0)
        assert result.seq == 'ABABAB'
    
    def test_repeat_one(self):
        """Test repetition by 1."""
        pool = from_seqs(['ABC'])
        result = pool * 1
        
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'ABC'
    
    def test_repeat_zero(self):
        """Test repetition by 0."""
        pool = from_seqs(['ABC'])
        with pytest.raises(ValueError, match="is not positive"):
            result = pool * 0


class TestSliceOperation:
    """Tests for slicing ([]) operator."""
    
    def test_single_index(self):
        """Test single index slicing."""
        pool = from_seqs(['ABCDE'])
        
        result = pool[0]
        result.set_state(0)
        assert result.seq == 'A'
        
        result = pool[2]
        result.set_state(0)
        assert result.seq == 'C'
        
        result = pool[-1]
        result.set_state(0)
        assert result.seq == 'E'
    
    def test_slice_range(self):
        """Test slice range."""
        pool = from_seqs(['ABCDE'])
        
        result = pool[1:4]
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'BCD'
    
    def test_slice_with_step(self):
        """Test slice with step."""
        pool = from_seqs(['ABCDEF'])
        
        result = pool[::2]
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'ACE'
    
    def test_slice_negative(self):
        """Test negative slicing."""
        pool = from_seqs(['ABCDE'])
        
        result = pool[-3:]
        assert result.seq_length == 3
        result.set_state(0)
        assert result.seq == 'CDE'


class TestComplexCompositions:
    """Tests for complex compositions of operations."""
    
    def test_kmer_pool_plus_from_seqs(self):
        """Test combining kmer_pool and from_seqs."""
        barcode = get_kmers(length=4, alphabet='dna')
        promoter = from_seqs(['TATA'])
        result = promoter + '...' + barcode
        
        assert result.seq_length == 4 + 3 + 4  # 11
    
    def test_complex_sequential_iteration(self):
        """Test complex composition with sequential iteration."""
        left = from_seqs(['A', 'B'], mode='sequential')
        right = from_seqs(['1', '2'], mode='sequential')
        result = left + right
        
        result_df = result.generate_library(num_complete_iterations=1)
        assert len(result_df) == 4
        assert set(result_df['seq']) == {'A1', 'A2', 'B1', 'B2'}
    
    def test_nested_operations(self):
        """Test nested operations."""
        base = from_seqs(['XY'])
        repeated = base * 2
        sliced = repeated[1:3]
        
        assert sliced.seq_length == 2
        sliced.set_state(0)
        assert sliced.seq == 'YX'
