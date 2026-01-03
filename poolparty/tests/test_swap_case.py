"""Tests for the swap_case operation."""

import pytest
import poolparty as pp
from poolparty.fixed_ops.swap_case import swap_case
from poolparty.fixed_ops.fixed import FixedOp


class TestSwapCaseBasics:
    """Test basic swap_case functionality."""
    
    def test_returns_pool(self):
        """Test that swap_case returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT')
            result = swap_case(pool)
            assert hasattr(result, 'operation')
            assert isinstance(result.operation, FixedOp)
    
    def test_swaps_uppercase_to_lowercase(self):
        """Test that uppercase letters become lowercase."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT')
            result = swap_case(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'acgt'
    
    def test_swaps_lowercase_to_uppercase(self):
        """Test that lowercase letters become uppercase."""
        with pp.Party() as party:
            pool = pp.from_seq('acgt')
            result = swap_case(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_swaps_mixed_case(self):
        """Test that mixed case sequences are properly swapped."""
        with pp.Party() as party:
            pool = pp.from_seq('AcGt')
            result = swap_case(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aCgT'


class TestSwapCaseWithString:
    """Test swap_case with string input."""
    
    def test_string_input(self):
        """Test that string input is converted and swapped."""
        with pp.Party() as party:
            result = swap_case('ACGT').named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'acgt'


class TestSwapCaseWithMultipleSeqs:
    """Test swap_case with pools containing multiple sequences."""
    
    def test_multiple_sequences(self):
        """Test swap_case with multiple sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAAA', 'CCCC', 'GGGG'], mode='sequential')
            result = swap_case(pool).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 3
        assert set(df['seq']) == {'aaaa', 'cccc', 'gggg'}


class TestSwapCaseNaming:
    """Test naming parameters."""
    
    def test_pool_name(self):
        """Test name parameter."""
        with pp.Party() as party:
            result = swap_case('ACGT', name='my_pool')
        
        assert result.name == 'my_pool'
    
    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party() as party:
            result = swap_case('ACGT', op_name='my_op')
        
        assert result.operation.name == 'my_op'


class TestSwapCasePreservesSeqLength:
    """Test that swap_case preserves sequence length."""
    
    def test_preserves_seq_length(self):
        """Test that seq_length is preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGTACGT')
            result = swap_case(pool)
        
        assert result.seq_length == 8
        assert result.seq_length == pool.seq_length
