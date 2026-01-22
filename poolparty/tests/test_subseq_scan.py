"""Tests for the subseq_scan function."""

import pytest
import poolparty as pp
from poolparty.scan_ops import subseq_scan


class TestSubseqScanBasics:
    """Test basic subseq_scan functionality."""
    
    def test_returns_pool(self):
        """Test that subseq_scan returns a Pool."""
        with pp.Party() as party:
            result = subseq_scan('ACGTACGTACGT', seq_length=4)
            assert hasattr(result, 'operation')
    
    def test_extracts_correct_length(self):
        """Test that extracted subsequences have correct length."""
        with pp.Party() as party:
            result = subseq_scan('ACGTACGTACGT', seq_length=4, mode='sequential').named('result')
        
        df = result.generate_library(num_seqs=20, seed=42)
        for seq in df['seq']:
            assert len(seq) == 4
    
    def test_sequential_mode_all_positions(self):
        """Test sequential mode extracts at all positions."""
        with pp.Party() as party:
            # 12-char sequence, 4-char extraction = 9 positions (0-8)
            result = subseq_scan('ACGTACGTACGT', seq_length=4, mode='sequential').named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 9
        
        expected_seqs = [
            'ACGT', 'CGTA', 'GTAC', 'TACG',
            'ACGT', 'CGTA', 'GTAC', 'TACG', 'ACGT'
        ]
        assert list(df['seq']) == expected_seqs
    
    def test_extracts_from_pool_input(self):
        """Test subseq_scan with Pool input."""
        with pp.Party() as party:
            bg = pp.from_seqs(['ACGTACGTACGT'])
            result = subseq_scan(bg, seq_length=4, mode='sequential').named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 9


class TestSubseqScanPositions:
    """Test positions parameter."""
    
    def test_specific_positions(self):
        """Test extraction at specific positions."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTACGTACGT',
                seq_length=4,
                positions=[0, 4, 8],
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 3
        assert list(df['seq']) == ['ACGT', 'ACGT', 'ACGT']
    
    def test_single_position(self):
        """Test extraction at a single position."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTTTTT',
                seq_length=4,
                positions=[0],
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_slice_positions(self):
        """Test extraction with slice positions."""
        with pp.Party() as party:
            # 12-char sequence, 4-char extraction = 9 positions (0-8)
            # slice(0, 6, 2) gives positions 0, 2, 4
            result = subseq_scan(
                'ACGTACGTACGT',
                seq_length=4,
                positions=slice(0, 6, 2),
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 3


class TestSubseqScanStrand:
    """Test strand parameter."""
    
    def test_plus_strand_default(self):
        """Test that default strand is '+'."""
        with pp.Party() as party:
            result = subseq_scan(
                'AAAATTTT',
                seq_length=4,
                positions=[0],
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'AAAA'
    
    def test_minus_strand_reverse_complements(self):
        """Test that minus strand returns reverse complement."""
        with pp.Party() as party:
            # AAAA reverse complement is TTTT
            result = subseq_scan(
                'AAAATTTT',
                seq_length=4,
                positions=[0],
                strand='-',
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'TTTT'
    
    def test_both_strands_doubles_states(self):
        """Test that strand='both' creates 2x states."""
        with pp.Party() as party:
            result = subseq_scan(
                'AAAATTTT',
                seq_length=4,
                positions=[0],
                strand='both',
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert len(df) == 2
        # One is AAAA (+ strand), one is TTTT (- strand reverse complement)
        assert set(df['seq']) == {'AAAA', 'TTTT'}
    
    def test_both_strands_multiple_positions(self):
        """Test strand='both' with multiple positions."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTACGTACGT',
                seq_length=4,
                positions=[0, 4],
                strand='both',
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        # 2 positions * 2 strands = 4 states
        assert len(df) == 4


class TestSubseqScanModes:
    """Test different modes."""
    
    def test_random_mode(self):
        """Test random mode samples different positions."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTACGTACGT',
                seq_length=4,
                mode='random'
            ).named('result')
        
        df = result.generate_library(num_seqs=50, seed=42)
        assert len(df) == 50
        
        # Should have variability in extracted sequences
        unique_seqs = set(df['seq'])
        assert len(unique_seqs) > 1
    
    def test_hybrid_mode(self):
        """Test hybrid mode."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTACGTACGT',
                seq_length=4,
                mode='random',
                num_states=3
            ).named('result')
        
        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20
        
        for seq in df['seq']:
            assert len(seq) == 4


class TestSubseqScanNaming:
    """Test naming parameters."""
    
    def test_pool_name(self):
        """Test name parameter."""
        with pp.Party() as party:
            result = subseq_scan('ACGTACGT', seq_length=4, name='my_subseqs')
        
        assert result.name == 'my_subseqs'
    
    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party() as party:
            result = subseq_scan('ACGTACGT', seq_length=4, op_name='my_extract')
        
        # op_name is passed to the extract operation
        assert result.operation.name == 'my_extract'


class TestSubseqScanValidation:
    """Test input validation."""
    
    def test_pool_requires_seq_length(self):
        """Test error when pool has no seq_length."""
        with pp.Party() as party:
            # breakpoint_scan outputs have seq_length=None (variable length)
            left, right = pp.breakpoint_scan('AAAAAAAAAA', num_breakpoints=1)
            
            with pytest.raises(ValueError, match="pool must have a defined seq_length"):
                subseq_scan(left, seq_length=3)
    
    def test_seq_length_must_be_positive(self):
        """Test error when seq_length <= 0."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="seq_length must be > 0"):
                subseq_scan('ACGTACGT', seq_length=0)
            
            with pytest.raises(ValueError, match="seq_length must be > 0"):
                subseq_scan('ACGTACGT', seq_length=-1)
    
    def test_seq_length_must_not_exceed_pool_length(self):
        """Test error when seq_length > pool length."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="seq_length .* must be <= pool.seq_length"):
                subseq_scan('ACGT', seq_length=5)
    
    def test_position_exceeds_maximum(self):
        """Test error when position exceeds maximum allowed value."""
        with pp.Party() as party:
            # 8-char sequence, 4-char extraction = max position is 4
            with pytest.raises(ValueError, match="out of range"):
                subseq_scan('ACGTACGT', seq_length=4, positions=[5])


class TestSubseqScanEdgeCases:
    """Test edge cases."""
    
    def test_extract_at_start(self):
        """Test extraction at position 0."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGTTTTT',
                seq_length=4,
                positions=[0],
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_extract_at_end(self):
        """Test extraction at maximum position."""
        with pp.Party() as party:
            # 8-char sequence, 4-char extraction = max position is 4
            result = subseq_scan(
                'TTTTACGT',
                seq_length=4,
                positions=[4],
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_extract_single_char(self):
        """Test extracting single character."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGT',
                seq_length=1,
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        # 4 positions
        assert len(df) == 4
        assert list(df['seq']) == ['A', 'C', 'G', 'T']
    
    def test_extract_entire_sequence(self):
        """Test extracting the entire sequence."""
        with pp.Party() as party:
            result = subseq_scan(
                'ACGT',
                seq_length=4,
                mode='sequential'
            ).named('result')
        
        df = result.generate_library(num_cycles=1)
        # Only 1 position (position 0)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'ACGT'


class TestSubseqScanWithMultipleSeqs:
    """Test subseq_scan with pools containing multiple sequences."""
    
    def test_multiple_backgrounds(self):
        """Test with multiple background sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAA', 'CCCCCCCC'], mode='sequential')
            result = subseq_scan(bg, seq_length=4, mode='sequential').named('result')
        
        df = result.generate_library(num_cycles=1)
        # 2 backgrounds * 5 positions = 10 sequences
        assert len(df) == 10
        
        # All should be length 4
        for seq in df['seq']:
            assert len(seq) == 4


class TestSubseqScanSeqLengthOutput:
    """Test that output pool has correct seq_length."""
    
    def test_output_seq_length(self):
        """Test that output pool has seq_length equal to input seq_length."""
        with pp.Party() as party:
            result = subseq_scan('ACGTACGTACGT', seq_length=4)
        
        assert result.seq_length == 4
    
    def test_output_seq_length_various_sizes(self):
        """Test output seq_length for various extraction sizes."""
        for extract_len in [1, 3, 5, 8]:
            with pp.Party() as party:
                result = subseq_scan('ACGTACGTACGT', seq_length=extract_len)
            
            assert result.seq_length == extract_len
