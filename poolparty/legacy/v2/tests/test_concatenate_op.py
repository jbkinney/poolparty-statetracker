"""Tests for concatenate operation."""

import pytest
from poolparty import from_seqs, Pool
from poolparty.operations.concatenate_op import concatenate, ConcatenateOp


class TestConcatenate:
    """Tests for concatenate factory function."""
    
    def test_basic_two_sequences(self):
        """Test basic concatenation of two sequences."""
        result = concatenate(['AAA', 'TTT'])
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_state(0)
        assert result.seq == 'AAATTT'
    
    def test_three_sequences(self):
        """Test concatenation of three sequences."""
        result = concatenate(['AA', 'CC', 'GG'])
        assert isinstance(result, Pool)
        assert result.seq_length == 6
        result.set_state(0)
        assert result.seq == 'AACCGG'
    
    def test_four_sequences(self):
        """Test concatenation of four sequences."""
        result = concatenate(['A', 'C', 'G', 'T'])
        assert isinstance(result, Pool)
        assert result.seq_length == 4
        result.set_state(0)
        assert result.seq == 'ACGT'
    
    def test_five_sequences(self):
        """Test concatenation of five sequences."""
        result = concatenate(['AA', 'BB', 'CC', 'DD', 'EE'])
        assert isinstance(result, Pool)
        assert result.seq_length == 10
        result.set_state(0)
        assert result.seq == 'AABBCCDDEE'
    
    def test_many_sequences(self):
        """Test concatenation of many sequences."""
        seqs = [chr(65 + i) for i in range(10)]  # A, B, C, ..., J
        result = concatenate(seqs)
        assert result.seq_length == 10
        result.set_state(0)
        assert result.seq == 'ABCDEFGHIJ'
    
    def test_mixed_pool_and_string(self):
        """Test concatenation with mixed Pool and string inputs."""
        pool1 = from_seqs(['AAA'])
        pool2 = from_seqs(['GGG'])
        result = concatenate([pool1, 'TTT', pool2, 'CCC'])
        
        assert isinstance(result, Pool)
        assert result.seq_length == 12
        result.set_state(0)
        assert result.seq == 'AAATTTGGGCCC'
    
    def test_all_pools(self):
        """Test concatenation of all Pool inputs."""
        pool1 = from_seqs(['AA'])
        pool2 = from_seqs(['BB'])
        pool3 = from_seqs(['CC'])
        result = concatenate([pool1, pool2, pool3])
        
        assert result.seq_length == 6
        result.set_state(0)
        assert result.seq == 'AABBCC'
    
    def test_seq_length_fixed(self):
        """Test seq_length calculation with fixed-length parents."""
        result = concatenate(['ABC', 'DE', 'FGHI'])
        assert result.seq_length == 9  # 3 + 2 + 4
    
    def test_seq_length_variable(self):
        """Test seq_length is None when parent has variable length."""
        # Create a pool with variable-length sequences
        var_pool = from_seqs(['A', 'BB', 'CCC'])
        result = concatenate([var_pool, 'XX'])
        assert result.seq_length is None
    
    def test_num_states_always_one(self):
        """Test that ConcatenateOp always has num_states=1."""
        result = concatenate(['AAA', 'TTT'])
        assert result.operation.num_states == 1
        
        # Even with multiple inputs
        result = concatenate(['A', 'B', 'C', 'D', 'E'])
        assert result.operation.num_states == 1
    
    def test_name_attribute(self):
        """Test name attribute on ConcatenateOp."""
        op = ConcatenateOp(['AAA', 'TTT'], name='my_concat')
        pool = Pool(operation=op)
        assert pool.operation.name == 'my_concat'
    
    def test_sequential_mode_with_varying_parents(self):
        """Test sequential mode enumeration with multiple varying parents."""
        pool1 = from_seqs(['A', 'B'], mode='sequential')
        pool2 = from_seqs(['1', '2'], mode='sequential')
        pool3 = from_seqs(['X', 'Y'], mode='sequential')
        result = concatenate([pool1, pool2, pool3])
        
        result_df = result.generate_library(num_complete_iterations=1)
        
        # Should get all 8 combinations (2 × 2 × 2)
        assert len(result_df) == 8
        expected = {'A1X', 'A1Y', 'A2X', 'A2Y', 'B1X', 'B1Y', 'B2X', 'B2Y'}
        assert set(result_df['seq']) == expected
    
    def test_sequential_mode_three_varying_pools(self):
        """Test sequential mode with three varying pools of different sizes."""
        pool1 = from_seqs(['A', 'B', 'C'], mode='sequential')
        pool2 = from_seqs(['1', '2'], mode='sequential')
        result = concatenate([pool1, '-', pool2])
        
        result_df = result.generate_library(num_complete_iterations=1)
        
        # Should get 3 × 2 = 6 combinations
        assert len(result_df) == 6
        expected = {'A-1', 'A-2', 'B-1', 'B-2', 'C-1', 'C-2'}
        assert set(result_df['seq']) == expected


class TestConcatenateAncestors:
    """Tests for ancestor tracking in concatenate pools."""
    
    def test_parent_pools_from_pools(self):
        """Test that parent_pools contains all input pools."""
        pool1 = from_seqs(['AAA'])
        pool2 = from_seqs(['TTT'])
        pool3 = from_seqs(['GGG'])
        result = concatenate([pool1, pool2, pool3])
        
        parents = result.operation.parent_pools
        assert len(parents) == 3
        assert parents[0] is pool1
        assert parents[1] is pool2
        assert parents[2] is pool3
    
    def test_strings_converted_to_pools(self):
        """Test that string inputs are converted to from_seqs pools."""
        result = concatenate(['AAA', 'TTT'])
        
        parents = result.operation.parent_pools
        assert len(parents) == 2
        # Each parent should be a Pool (from the from_seqs conversion)
        assert all(isinstance(p, Pool) for p in parents)
    
    def test_mixed_inputs_parent_tracking(self):
        """Test parent tracking with mixed Pool and string inputs."""
        pool1 = from_seqs(['AAA'])
        result = concatenate([pool1, 'TTT', 'GGG'])
        
        parents = result.operation.parent_pools
        assert len(parents) == 3
        assert parents[0] is pool1
        # The string inputs should be converted to pools
        assert isinstance(parents[1], Pool)
        assert isinstance(parents[2], Pool)
