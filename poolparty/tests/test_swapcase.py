"""Tests for the swapcase operation."""

import pytest
import poolparty as pp
from poolparty.fixed_ops.swapcase import swapcase
from poolparty.fixed_ops.fixed import FixedOp


class TestSwapCaseBasics:
    """Test basic swapcase functionality."""
    
    def test_returns_pool(self):
        """Test that swapcase returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT')
            result = swapcase(pool)
            assert hasattr(result, 'operation')
            assert isinstance(result.operation, FixedOp)
    
    def test_swaps_uppercase_to_lowercase(self):
        """Test that uppercase letters become lowercase."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'acgt'
    
    def test_swaps_lowercase_to_uppercase(self):
        """Test that lowercase letters become uppercase."""
        with pp.Party() as party:
            pool = pp.from_seq('acgt')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_swaps_mixed_case(self):
        """Test that mixed case sequences are properly swapped."""
        with pp.Party() as party:
            pool = pp.from_seq('AcGt')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aCgT'


class TestSwapCaseWithString:
    """Test swapcase with string input."""
    
    def test_string_input(self):
        """Test that string input is converted and swapped."""
        with pp.Party() as party:
            result = swapcase('ACGT').named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'acgt'


class TestSwapCaseWithMultipleSeqs:
    """Test swapcase with pools containing multiple sequences."""
    
    def test_multiple_sequences(self):
        """Test swapcase with multiple sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAAA', 'CCCC', 'GGGG'], mode='sequential')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 3
        assert set(df['seq']) == {'aaaa', 'cccc', 'gggg'}


class TestSwapCaseNaming:
    """Test naming parameters."""
    
    def test_pool_name(self):
        """Test name parameter."""
        with pp.Party() as party:
            result = swapcase('ACGT', name='my_pool')
        
        assert result.name == 'my_pool'
    
    def test_default_op_name(self):
        """Test default operation name is 'swapcase'."""
        with pp.Party() as party:
            result = swapcase('ACGT')
        
        assert result.operation.name.endswith(':swapcase')


class TestSwapCasePreservesSeqLength:
    """Test that swapcase preserves sequence length."""
    
    def test_preserves_seq_length(self):
        """Test that seq_length is preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGTACGT')
            result = swapcase(pool)
        
        assert result.seq_length == 8
        assert result.seq_length == pool.seq_length


class TestSwapCasePreservesMarkers:
    """Test that swapcase preserves XML marker tags."""
    
    def test_preserves_region_marker(self):
        """Test that region markers are preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<region>TT</region>CC')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<region>tt</region>cc'
    
    def test_preserves_marker_with_attributes(self):
        """Test that marker attributes are preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<m strand="-">BB</m>CC')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<m strand="-">bb</m>cc'
    
    def test_preserves_self_closing_marker(self):
        """Test that self-closing markers are preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<ins/>BB')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<ins/>bb'
    
    def test_preserves_nested_markers(self):
        """Test that nested markers are preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<outer>BB<inner>CC</inner>DD</outer>EE')
            result = swapcase(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<outer>bb<inner>cc</inner>dd</outer>ee'


class TestUpperPreservesMarkers:
    """Test that upper() preserves XML marker tags."""
    
    def test_upper_preserves_region_marker(self):
        """Test that upper preserves region markers."""
        with pp.Party() as party:
            pool = pp.from_seq('aa<region>tt</region>cc')
            result = pp.upper(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AA<region>TT</region>CC'
    
    def test_upper_preserves_marker_attributes(self):
        """Test that upper preserves marker attributes."""
        with pp.Party() as party:
            pool = pp.from_seq('aa<m strand="-">bb</m>cc')
            result = pp.upper(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AA<m strand="-">BB</m>CC'


class TestLowerPreservesMarkers:
    """Test that lower() preserves XML marker tags."""
    
    def test_lower_preserves_region_marker(self):
        """Test that lower preserves region markers."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<region>TT</region>CC')
            result = pp.lower(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<region>tt</region>cc'
    
    def test_lower_preserves_marker_attributes(self):
        """Test that lower preserves marker attributes."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<m strand="-">BB</m>CC')
            result = pp.lower(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'aa<m strand="-">bb</m>cc'


class TestReverseComplementPreservesMarkers:
    """Test that reverse_complement() preserves and repositions XML marker tags."""
    
    def test_rc_region_marker(self):
        """Test that region markers are repositioned correctly."""
        with pp.Party() as party:
            pool = pp.from_seq('ACG<region>TT</region>AA')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'TT<region>AA</region>CGT'
    
    def test_rc_marker_with_strand_attribute(self):
        """Test that marker strand attribute is preserved."""
        with pp.Party() as party:
            pool = pp.from_seq('AA<m strand="-">CC</m>GG')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == "CC<m strand='-'>GG</m>TT"
    
    def test_rc_self_closing_marker(self):
        """Test that self-closing markers are repositioned correctly."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT<ins/>AAAA')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'TTTT<ins/>ACGT'
    
    def test_rc_marker_at_start(self):
        """Test marker at sequence start."""
        with pp.Party() as party:
            pool = pp.from_seq('<region>ACGT</region>AAAA')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        # Content: ACGTAAAA (8), marker [0,4) -> [4,8)
        assert df['seq'].iloc[0] == 'TTTT<region>ACGT</region>'
    
    def test_rc_marker_at_end(self):
        """Test marker at sequence end."""
        with pp.Party() as party:
            pool = pp.from_seq('AAAA<region>ACGT</region>')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        # Content: AAAAACGT (8), marker [4,8) -> [0,4)
        assert df['seq'].iloc[0] == '<region>ACGT</region>TTTT'
    
    def test_rc_multiple_markers(self):
        """Test multiple markers are repositioned correctly."""
        with pp.Party() as party:
            pool = pp.from_seq('A<m1>T</m1>G<m2>C</m2>')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        # Content: ATGC (4), m1 [1,2) -> [2,3), m2 [3,4) -> [0,1)
        # Reversed: GCAT, complemented: CGTA? Wait...
        # reverse(ATGC) = CGTA, complement(CGTA) = GCAT
        # So result: <m2>G</m2>CA<m1>T</m1>
        assert df['seq'].iloc[0] == '<m2>G</m2>C<m1>A</m1>T'
    
    def test_rc_nested_markers(self):
        """Test nested markers are repositioned correctly."""
        with pp.Party() as party:
            pool = pp.from_seq('A<outer>C<inner>G</inner>T</outer>A')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        # Content: ACGTA (5), outer [1,4) -> [1,4), inner [2,3) -> [2,3)
        # Reversed: ATGCA, complemented: TACGT
        assert df['seq'].iloc[0] == 'T<outer>A<inner>C</inner>G</outer>T'
    
    def test_rc_no_markers(self):
        """Test that reverse_complement still works without markers."""
        with pp.Party() as party:
            pool = pp.from_seq('ACGT')
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_rc_with_seq_length_attribute(self):
        """Test that seq_length attribute is preserved."""
        with pp.Party() as party:
            pool = pp.from_seq("AA<m seq_length='2'>CC</m>GG")
            result = pp.reverse_complement(pool).named('result')
        
        df = result.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == "CC<m seq_length='2'>GG</m>TT"
