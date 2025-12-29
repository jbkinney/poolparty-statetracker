"""Tests for the replacement_scan wrapper function."""

import pytest
import poolparty as pp
from poolparty.operations.replacement_scan import replacement_scan


class TestReplacementScanBasics:
    """Test basic replacement_scan functionality."""
    
    def test_returns_pool(self):
        """Test that replacement_scan returns a Pool."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins)
            assert hasattr(result, 'operation')
    
    def test_sequential_mode_default(self):
        """Test replacement_scan defaults to sequential mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins).named('result')
        
        # Default: start=0, end=7, step_size=1 => 8 positions
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 8
    
    def test_preserves_total_length(self):
        """Test that output length equals background length."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            assert len(seq) == 10
    
    def test_insert_appears_in_output(self):
        """Test that insert sequence appears in output."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            assert 'TTT' in seq


class TestReplacementScanStringInputs:
    """Test replacement_scan with string inputs."""
    
    def test_bg_pool_as_string(self):
        """Test background as string."""
        with pp.Party() as party:
            result = replacement_scan('AAAAAAAAAA', pp.from_seqs(['TTT'])).named('result')
        
        df = result.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            assert 'TTT' in seq
            assert len(seq) == 10
    
    def test_ins_pool_as_string(self):
        """Test insert as string."""
        with pp.Party() as party:
            result = replacement_scan(pp.from_seqs(['AAAAAAAAAA']), 'TTT').named('result')
        
        df = result.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            assert 'TTT' in seq
            assert len(seq) == 10
    
    def test_both_as_strings(self):
        """Test both background and insert as strings."""
        with pp.Party() as party:
            result = replacement_scan('AAAAAAAAAA', 'TTT').named('result')
        
        df = result.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            assert 'TTT' in seq
            assert len(seq) == 10


class TestReplacementScanStartEndStep:
    """Test start, end, and step_size parameters."""
    
    def test_custom_start(self):
        """Test custom start position."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins, start=3).named('result')
        
        # start=3, end=7 (default), step_size=1 => 5 positions
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 5
        
        # All inserts should start at position 3 or later
        for seq in df['seq']:
            idx = seq.index('TTT')
            assert idx >= 3
    
    def test_custom_end(self):
        """Test custom end position."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins, end=4).named('result')
        
        # start=0 (default), end=4, step_size=1 => 5 positions
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 5
        
        # All inserts should start at position 4 or earlier
        for seq in df['seq']:
            idx = seq.index('TTT')
            assert idx <= 4
    
    def test_custom_step_size(self):
        """Test custom step_size."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins, step_size=2).named('result')
        
        # start=0, end=7, step_size=2 => positions 0, 2, 4, 6 = 4 positions
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 4
    
    def test_combined_start_end_step(self):
        """Test combining start, end, and step_size."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = replacement_scan(bg, ins, start=2, end=6, step_size=2).named('result')
        
        # positions 2, 4, 6 = 3 positions
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 3


class TestReplacementScanModes:
    """Test different modes."""
    
    def test_random_mode(self):
        """Test random mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, mode='random').named('result')
        
        df = result.generate_seqs(num_seqs=50, seed=42)
        assert len(df) == 50
        
        # Should have variability in insert positions
        positions = [seq.index('TTT') for seq in df['seq']]
        assert len(set(positions)) > 1
    
    def test_hybrid_mode(self):
        """Test hybrid mode."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, mode='hybrid', 
                                      hybrid_mode_num_states=5).named('result')
        
        df = result.generate_seqs(num_seqs=20, seed=42)
        assert len(df) == 20
        
        for seq in df['seq']:
            assert 'TTT' in seq
            assert len(seq) == 10


class TestReplacementScanSpacerStr:
    """Test spacer_str parameter."""
    
    def test_spacer_str(self):
        """Test custom spacer string."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, spacer_str='.').named('result')
        
        df = result.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            # Should have dots around the insert
            assert '.TTT.' in seq
    
    def test_empty_spacer_default(self):
        """Test that default spacer is empty."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            # Should NOT have dots
            assert '.' not in seq


class TestReplacementScanNaming:
    """Test naming parameters."""
    
    def test_pool_name(self):
        """Test pool_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, pool_name='my_result')
        
        assert result.name == 'my_result'
    
    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, op_name='my_join')
        
        # op_name is passed to the join operation
        assert result.operation.name == 'my_join'


class TestReplacementScanValidation:
    """Test input validation."""
    
    def test_bg_pool_requires_seq_length(self):
        """Test error when bg_pool has no seq_length."""
        with pp.Party() as party:
            # breakpoint_scan outputs have seq_length=None (variable length)
            left, right = pp.breakpoint_scan('AAAAAAAAAA', num_breakpoints=1)
            ins = pp.from_seqs(['TTT'])
            
            with pytest.raises(ValueError, match="bg_pool must have a defined seq_length"):
                replacement_scan(left, ins)
    
    def test_ins_pool_requires_seq_length(self):
        """Test error when ins_pool has no seq_length."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            # breakpoint_scan outputs have seq_length=None (variable length)
            left, right = pp.breakpoint_scan('TTT', num_breakpoints=1)
            
            with pytest.raises(ValueError, match="ins_pool must have a defined seq_length"):
                replacement_scan(bg, left)
    
    def test_end_exceeds_maximum(self):
        """Test error when end exceeds maximum allowed value."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            # max_end = 10 - 3 = 7
            
            with pytest.raises(ValueError, match="end .* exceeds maximum allowed value"):
                replacement_scan(bg, ins, end=8)


class TestReplacementScanWithMultipleSeqs:
    """Test replacement_scan with pools containing multiple sequences."""
    
    def test_multiple_backgrounds(self):
        """Test with multiple background sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA', 'CCCCCCCCCC'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        # 2 backgrounds * 8 positions = 16 sequences
        assert len(df) == 16
        
        for seq in df['seq']:
            assert 'TTT' in seq
            assert len(seq) == 10
    
    def test_multiple_inserts(self):
        """Test with multiple insert sequences."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT', 'GGG'])
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        # 8 positions * 2 inserts = 16 sequences
        assert len(df) == 16
        
        for seq in df['seq']:
            assert 'TTT' in seq or 'GGG' in seq
            assert len(seq) == 10
    
    def test_multiple_backgrounds_and_inserts(self):
        """Test with multiple backgrounds and inserts."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA', 'CCCCCCCCCC'])
            ins = pp.from_seqs(['TTT', 'GGG'])
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        # 2 backgrounds * 8 positions * 2 inserts = 32 sequences
        assert len(df) == 32


class TestReplacementScanEdgeCases:
    """Test edge cases."""
    
    def test_start_at_zero(self):
        """Test replacement at position 0."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = replacement_scan(bg, ins, start=0, end=0).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'TTTAAAAAAA'
    
    def test_replace_at_end(self):
        """Test replacement at maximum position."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            # max_end = 7
            result = replacement_scan(bg, ins, start=7, end=7).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'AAAAAAATTT'
    
    def test_replace_same_length_as_background(self):
        """Test when insert length equals background length."""
        with pp.Party() as party:
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTTTTTTTTT'])  # 10 chars
            # max_end = 10 - 10 = 0
            result = replacement_scan(bg, ins).named('result')
        
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        # Only position 0, so insert replaces everything
        assert df['seq'].iloc[0] == 'TTTTTTTTTT'

