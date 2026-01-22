"""Tests for the FromIupac operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.from_iupac import FromIupacOp, from_iupac


class TestFromIupacFactory:
    """Test from_iupac factory function."""
    
    def test_returns_pool(self):
        """from_iupac returns a Pool object."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_from_iupac_op(self):
        """Pool's operation is FromIupacOp."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert isinstance(pool.operation, FromIupacOp)
    
    def test_works_with_default_party(self):
        """from_iupac works with default party context (no explicit context needed)."""
        pool = from_iupac('ACGT')
        assert pool is not None
        assert hasattr(pool, 'operation')


class TestFromIupacValidation:
    """Test parameter validation."""
    
    def test_empty_string_error(self):
        """Empty iupac_seq raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="non-empty string"):
                from_iupac('')
    
    def test_invalid_char_error(self):
        """Invalid IUPAC character raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="invalid IUPAC character"):
                from_iupac('ACGTX')
    


class TestFromIupacSequentialMode:
    """Test sequential mode."""
    
    def test_sequential_enumeration(self):
        """Sequential mode enumerates all possibilities."""
        with pp.Party() as party:
            pool = from_iupac('RY', mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        # R = A|G, Y = C|T -> 2*2 = 4 sequences
        assert len(df) == 4
        seqs = set(df['seq'])
        assert seqs == {'AC', 'AT', 'GC', 'GT'}
    
    def test_num_states_computation(self):
        """num_states equals product of possibilities."""
        with pp.Party() as party:
            # N = 4 options, so NN = 16 states
            pool = from_iupac('NN', mode='sequential')
            assert pool.operation.num_values == 16


class TestFromIupacRandomMode:
    """Test random mode."""
    
    def test_random_num_states_is_one(self):
        """Random mode has num_values=1."""
        with pp.Party() as party:
            pool = from_iupac('ACGT', mode='random')
            assert pool.operation.num_values == 1
    
    def test_random_sampling(self):
        """Random mode produces valid DNA sequences."""
        with pp.Party() as party:
            pool = from_iupac('NNNN', mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=100, seed=42)
        assert len(df) == 100
        for seq in df['seq']:
            assert len(seq) == 4
            assert all(c in 'ACGT' for c in seq)
    
    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        with pp.Party() as party:
            pool1 = from_iupac('NNNN', mode='random').named('iupac')
        df1 = pool1.generate_library(num_seqs=10, seed=42)
        
        with pp.Party() as party:
            pool2 = from_iupac('NNNN', mode='random').named('iupac')
        df2 = pool2.generate_library(num_seqs=10, seed=42)
        
        assert list(df1['seq']) == list(df2['seq'])


class TestFromIupacMarkChanges:
    """Test mark_changes parameter."""
    
    def test_mark_changes_false_by_default(self):
        """Default mark_changes is False."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert pool.operation.mark_changes is False
    
    def test_mark_changes_true_no_effect_without_region(self):
        """mark_changes=True has no effect without region."""
        with pp.Party() as party:
            # A and T are fixed, N is degenerate (4 options)
            pool = from_iupac('ANT', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df['seq']:
            # Without region, mark_changes has no effect - all uppercase
            assert seq == seq.upper()
    
    def test_mark_changes_false_preserves_uppercase(self):
        """mark_changes=False preserves uppercase input."""
        with pp.Party() as party:
            pool = from_iupac('ANT', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq == seq.upper()
    
    def test_mark_changes_false_preserves_lowercase(self):
        """mark_changes=False preserves lowercase input."""
        with pp.Party() as party:
            pool = from_iupac('ant', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq == seq.lower()
    
    def test_mark_changes_false_preserves_mixed_case(self):
        """mark_changes=False preserves mixed case input."""
        with pp.Party() as party:
            pool = from_iupac('AnT', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq[0] == 'A'
            assert seq[1].islower()
            assert seq[2] == 'T'
    
    def test_mark_changes_no_effect_without_region(self):
        """mark_changes has no effect when region is not specified."""
        with pp.Party() as party:
            # R = A|G (degenerate), Y = C|T (degenerate)
            pool = from_iupac('ARYT', mark_changes=True, mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        for seq in df['seq']:
            # Without region, mark_changes has no effect - sequence stays uppercase
            assert seq == seq.upper()
    
    def test_mark_changes_with_region(self):
        """mark_changes swaps entire sequence when region is specified."""
        with pp.Party() as party:
            bg = 'AAA<region>XXX</region>TTT'
            pool = from_iupac('NN', bg_pool=bg, region='region',
                                     mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df['seq']:
            # With region and mark_changes, the generated insert should be lowercase
            # The result format is: AAA + lowercase_insert + TTT
            assert seq.startswith('AAA')
            assert seq.endswith('TTT')
            insert = seq[3:5]  # The 2-char insert
            assert insert == insert.lower()
    
    def test_mark_changes_in_copy_params(self):
        """mark_changes is included in _get_copy_params."""
        with pp.Party() as party:
            pool = from_iupac('ACGT', mark_changes=True)
            params = pool.operation._get_copy_params()
        assert params['mark_changes'] is True


class TestFromIupacCustomName:
    """Test name parameters."""
    
    def test_default_operation_name(self):
        """Default operation name contains from_iupac."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert pool.operation.name.startswith('op[')
            assert ':from_iupac' in pool.operation.name
    
    def test_custom_operation_name(self):
        """Custom operation name."""
        with pp.Party() as party:
            pool = from_iupac('ACGT', op_name='my_motif')
            assert pool.operation.name == 'my_motif'
    
    def test_custom_pool_name(self):
        """Custom pool name."""
        with pp.Party() as party:
            pool = from_iupac('ACGT', name='my_pool')
            assert pool.name == 'my_pool'


class TestFromIupacDesignCards:
    """Test design card output."""
    
    def test_iupac_state_in_output(self):
        """Design card contains iupac_state."""
        with pp.Party() as party:
            pool = from_iupac('ACGT', op_name='motif').named('mypool')
        
        df = pool.generate_library(num_seqs=1, seed=42, report_design_cards=True)
        assert 'motif.key.iupac_state' in df.columns
    
    def test_design_card_keys_defined(self):
        """design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert 'iupac_state' in pool.operation.design_card_keys


class TestFromIupacSeqLength:
    """Test sequence length computation."""
    
    def test_seq_length_simple(self):
        """seq_length equals IUPAC sequence length."""
        with pp.Party() as party:
            pool = from_iupac('ACGT')
            assert pool.operation.seq_length == 4
    
    def test_seq_length_with_degenerate(self):
        """seq_length works with degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac('ANNN')
            assert pool.operation.seq_length == 4


class TestFromIupacIgnoreChars:
    """Test handling of ignore characters."""
    
    def test_dot_separator_preserved(self):
        """Dot separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac('ACG.TNN', mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Dot should be at position 3
            assert seq[3] == '.'
            assert len(seq) == 7
    
    def test_multiple_separators(self):
        """Multiple dot separators are preserved."""
        with pp.Party() as party:
            pool = from_iupac('A.C.G.T', mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df['seq']:
            assert seq == 'A.C.G.T'
    
    def test_dash_separator_preserved(self):
        """Dash separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac('ACG-TNN', mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df['seq']:
            assert seq[3] == '-'
    
    def test_ignore_chars_not_degenerate(self):
        """Ignore characters are not treated as degenerate."""
        with pp.Party() as party:
            pool = from_iupac('A.N', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Without region, mark_changes has no effect - all stay uppercase
            assert seq[0] == 'A'
            assert seq[1] == '.'
            assert seq[2].isupper()
    
    def test_num_states_ignores_separators(self):
        """Separators don't affect num_states calculation."""
        with pp.Party() as party:
            # N = 4 options, separators = 1 option each
            pool = from_iupac('N.N', mode='sequential')
            # 4 * 1 * 4 = 16 states
            assert pool.operation.num_values == 16


class TestFromIupacIgnoreCharsExtended:
    """Test handling of ignore characters."""
    
    def test_dot_separator_allowed(self):
        """Dot separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac('AC.GT', mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1  # Only one state since all positions are fixed
        assert df['seq'].iloc[0] == 'AC.GT'
    
    def test_hyphen_separator_allowed(self):
        """Hyphen separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac('AC-GT', mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'AC-GT'
    
    def test_space_separator_allowed(self):
        """Space separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac('AC GT', mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        assert df['seq'].iloc[0] == 'AC GT'
    
    def test_ignore_chars_with_degenerate(self):
        """Ignore characters work alongside degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac('A.N.T', mode='sequential').named('iupac')
        
        df = pool.generate_library(num_cycles=1)
        # N has 4 options, so 4 states total
        assert len(df) == 4
        # Check that dots are preserved
        for seq in df['seq']:
            assert seq[1] == '.'
            assert seq[3] == '.'
    
    def test_mark_changes_no_effect_on_separators_without_region(self):
        """mark_changes has no effect without region, separators preserved."""
        with pp.Party() as party:
            pool = from_iupac('A.N.T', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df['seq']:
            # Separators should remain unchanged
            assert seq[1] == '.'
            assert seq[3] == '.'
            # Without region, mark_changes has no effect - stays uppercase
            assert seq[0] == 'A'
            assert seq[4] == 'T'
