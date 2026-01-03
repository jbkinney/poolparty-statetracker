"""Tests for the unified scan() function."""

import pytest
from beartype.roar import BeartypeCallHintParamViolation
import poolparty as pp
from poolparty.operations.scan import scan


class TestScanInsertAction:
    """Test scan() with action='insert'."""

    def test_insert_returns_pool(self):
        """Test that scan('insert', ...) returns a Pool."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = scan('insert', bg, ins_pool=ins)
            assert hasattr(result, 'operation')

    def test_insert_output_length_is_sum(self):
        """Test that output length equals bg_length + ins_length."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = scan('insert', bg, ins_pool=ins).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            assert len(seq) == 13  # 10 + 3

    def test_insert_at_specific_position(self):
        """Test insertion at specific position."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool='TTT',
                         positions=[5], mode='sequential').named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'AAAAATTTAAAAA'  # 5 A's + TTT + 5 A's

    def test_insert_requires_ins_pool(self):
        """Test that ins_pool is required for insert action."""
        with pp.Party():
            with pytest.raises(ValueError, match="ins_pool is required"):
                scan('insert', 'AAAAAAAAAA')


class TestScanReplaceAction:
    """Test scan() with action='replace'."""

    def test_replace_returns_pool(self):
        """Test that scan('replace', ...) returns a Pool."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])
            ins = pp.from_seqs(['TTT'])
            result = scan('replace', bg, ins_pool=ins)
            assert hasattr(result, 'operation')

    def test_replace_preserves_length(self):
        """Test that output length equals background length."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])  # 10 chars
            ins = pp.from_seqs(['TTT'])  # 3 chars
            result = scan('replace', bg, ins_pool=ins).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            assert len(seq) == 10

    def test_replace_at_specific_position(self):
        """Test replacement at specific position."""
        with pp.Party():
            result = scan('replace', 'AAAAAAAAAA', ins_pool='TTT',
                         positions=[5], mode='sequential').named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'AAAAATTTAA'  # 5 A's + TTT + 2 A's

    def test_replace_requires_ins_pool(self):
        """Test that ins_pool is required for replace action."""
        with pp.Party():
            with pytest.raises(ValueError, match="ins_pool is required"):
                scan('replace', 'AAAAAAAAAA')

    def test_replace_mark_changes_applies_swap_case(self):
        """Test that mark_changes=True applies swap_case to insert."""
        with pp.Party():
            result = scan('replace', 'AAAAAAAAAA', ins_pool='TTT',
                         positions=[0], mode='sequential',
                         mark_changes=True).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        # TTT should become ttt (lowercase)
        assert df['seq'].iloc[0] == 'tttAAAAAAA'


class TestScanDeleteAction:
    """Test scan() with action='delete'."""

    def test_delete_returns_pool(self):
        """Test that scan('delete', ...) returns a Pool."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])
            result = scan('delete', bg, del_length=3)
            assert hasattr(result, 'operation')

    def test_delete_with_mark_changes_fills_gap(self):
        """Test that mark_changes=True fills gap with del_char."""
        with pp.Party():
            result = scan('delete', 'AAAAAAAAAA', del_length=3,
                         positions=[0], mode='sequential',
                         mark_changes=True).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == '---AAAAAAA'
        assert len(df['seq'].iloc[0]) == 10  # Length preserved

    def test_delete_without_mark_changes_removes_segment(self):
        """Test that mark_changes=False removes segment entirely."""
        with pp.Party():
            result = scan('delete', 'AAAAAAAAAA', del_length=3,
                         positions=[0], mode='sequential',
                         mark_changes=False).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 1
        assert df['seq'].iloc[0] == 'AAAAAAA'
        assert len(df['seq'].iloc[0]) == 7  # Length reduced

    def test_delete_custom_del_char(self):
        """Test custom deletion character."""
        with pp.Party():
            result = scan('delete', 'AAAAAAAAAA', del_length=3,
                         del_char='X', positions=[0], mode='sequential',
                         mark_changes=True).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        assert df['seq'].iloc[0] == 'XXXAAAAAAA'

    def test_delete_requires_del_length(self):
        """Test that del_length is required for delete action."""
        with pp.Party():
            with pytest.raises(ValueError, match="del_length is required"):
                scan('delete', 'AAAAAAAAAA')

    def test_delete_length_must_be_positive(self):
        """Test that del_length must be > 0."""
        with pp.Party():
            with pytest.raises(ValueError, match="del_length must be > 0"):
                scan('delete', 'AAAAAAAAAA', del_length=0)

    def test_delete_length_must_be_less_than_bg(self):
        """Test that del_length must be < bg_length."""
        with pp.Party():
            with pytest.raises(ValueError, match="del_length .* must be <"):
                scan('delete', 'AAAAAAAAAA', del_length=10)


class TestScanModes:
    """Test different selection modes."""

    def test_sequential_mode(self):
        """Test sequential mode generates all positions."""
        with pp.Party():
            result = scan('insert', 'AAAA', ins_pool='T',
                         mode='sequential').named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        # 5 positions (0, 1, 2, 3, 4) for inserting into 4-char sequence
        assert len(df) == 5

    def test_random_mode(self):
        """Test random mode generates variability."""
        with pp.Party():
            result = scan('replace', 'AAAAAAAAAA', ins_pool='TTT',
                         mode='random').named('result')

        df = result.generate_seqs(num_seqs=50, seed=42)
        assert len(df) == 50

        # Should have variability in positions
        positions = [seq.index('TTT') for seq in df['seq']]
        assert len(set(positions)) > 1

    def test_hybrid_mode(self):
        """Test hybrid mode with specified states."""
        with pp.Party():
            result = scan('replace', 'AAAAAAAAAA', ins_pool='TTT',
                         mode='hybrid', num_hybrid_states=5).named('result')

        df = result.generate_seqs(num_seqs=20, seed=42)
        assert len(df) == 20


class TestScanSpacerStr:
    """Test spacer_str parameter."""

    def test_spacer_str_with_insert(self):
        """Test spacer string with insert action."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool='TTT',
                         spacer_str='.', positions=[5], mode='sequential').named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        # Should have dots around the insert
        assert '.TTT.' in df['seq'].iloc[0]

    def test_spacer_str_with_delete_marked(self):
        """Test spacer string with marked deletion."""
        with pp.Party():
            result = scan('delete', 'AAAAAAAAAA', del_length=3,
                         spacer_str='.', positions=[3], mode='sequential',
                         mark_changes=True).named('result')

        df = result.generate_seqs(num_complete_iterations=1)
        # Should have dots around the deletion marker
        assert '.---.' in df['seq'].iloc[0]


class TestScanValidation:
    """Test input validation."""

    def test_invalid_action(self):
        """Test error for invalid action (caught by beartype)."""
        with pp.Party():
            with pytest.raises(BeartypeCallHintParamViolation):
                scan('invalid', 'AAAAAAAAAA')

    def test_bg_pool_requires_seq_length(self):
        """Test error when bg_pool has no seq_length."""
        with pp.Party():
            # breakpoint_scan outputs have seq_length=None
            left, right = pp.breakpoint_scan('AAAAAAAAAA', num_breakpoints=1)

            with pytest.raises(ValueError, match="bg_pool must have a defined seq_length"):
                scan('insert', left, ins_pool='TTT')

    def test_ins_pool_requires_seq_length(self):
        """Test error when ins_pool has no seq_length."""
        with pp.Party():
            bg = pp.from_seqs(['AAAAAAAAAA'])
            left, right = pp.breakpoint_scan('TTT', num_breakpoints=1)

            with pytest.raises(ValueError, match="ins_pool must have a defined seq_length"):
                scan('insert', bg, ins_pool=left)

    def test_position_out_of_range(self):
        """Test error when position exceeds maximum."""
        with pp.Party():
            with pytest.raises(ValueError, match="out of range"):
                scan('replace', 'AAAAAAAAAA', ins_pool='TTT', positions=[10])


class TestScanStringInputs:
    """Test scan with string inputs."""

    def test_bg_as_string(self):
        """Test background as string."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool=pp.from_seqs(['TTT'])).named('result')

        df = result.generate_seqs(num_seqs=1)
        assert 'TTT' in df['seq'].iloc[0]

    def test_ins_as_string(self):
        """Test insert as string."""
        with pp.Party():
            result = scan('insert', pp.from_seqs(['AAAAAAAAAA']), ins_pool='TTT').named('result')

        df = result.generate_seqs(num_seqs=1)
        assert 'TTT' in df['seq'].iloc[0]

    def test_both_as_strings(self):
        """Test both as strings."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool='TTT').named('result')

        df = result.generate_seqs(num_seqs=1)
        assert 'TTT' in df['seq'].iloc[0]


class TestScanNaming:
    """Test naming parameters."""

    def test_pool_name(self):
        """Test name parameter."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool='TTT', name='my_result')

        assert result.name == 'my_result'

    def test_op_name(self):
        """Test op_name parameter."""
        with pp.Party():
            result = scan('insert', 'AAAAAAAAAA', ins_pool='TTT', op_name='my_op')

        assert result.operation.name == 'my_op'
