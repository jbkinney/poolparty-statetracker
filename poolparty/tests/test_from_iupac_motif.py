"""Tests for the FromIupacMotif operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.from_iupac_motif import FromIupacMotifOp, from_iupac_motif


class TestFromIupacMotifFactory:
    """Test from_iupac_motif factory function."""
    
    def test_returns_pool(self):
        """from_iupac_motif returns a Pool object."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_from_iupac_motif_op(self):
        """Pool's operation is FromIupacMotifOp."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert isinstance(pool.operation, FromIupacMotifOp)
    
    def test_works_with_default_party(self):
        """from_iupac_motif works with default party context (no explicit context needed)."""
        pool = from_iupac_motif('ACGT')
        assert pool is not None
        assert hasattr(pool, 'operation')


class TestFromIupacMotifValidation:
    """Test parameter validation."""
    
    def test_empty_string_error(self):
        """Empty iupac_seq raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="non-empty string"):
                from_iupac_motif('')
    
    def test_invalid_char_error(self):
        """Invalid IUPAC character raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="invalid IUPAC character"):
                from_iupac_motif('ACGTX')
    
    def test_hybrid_requires_num_hybrid_states(self):
        """Hybrid mode requires num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                from_iupac_motif('ACGT', mode='hybrid')


class TestFromIupacMotifSequentialMode:
    """Test sequential mode."""
    
    def test_sequential_enumeration(self):
        """Sequential mode enumerates all possibilities."""
        with pp.Party() as party:
            pool = from_iupac_motif('RY', mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        # R = A|G, Y = C|T -> 2*2 = 4 sequences
        assert len(df) == 4
        seqs = set(df['seq'])
        assert seqs == {'AC', 'AT', 'GC', 'GT'}
    
    def test_num_states_computation(self):
        """num_states equals product of possibilities."""
        with pp.Party() as party:
            # N = 4 options, so NN = 16 states
            pool = from_iupac_motif('NN', mode='sequential')
            assert pool.operation.num_states == 16


class TestFromIupacMotifRandomMode:
    """Test random mode."""
    
    def test_random_num_states_is_one(self):
        """Random mode has num_states=1."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', mode='random')
            assert pool.operation.num_states == 1
    
    def test_random_sampling(self):
        """Random mode produces valid DNA sequences."""
        with pp.Party() as party:
            pool = from_iupac_motif('NNNN', mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        assert len(df) == 100
        for seq in df['seq']:
            assert len(seq) == 4
            assert all(c in 'ACGT' for c in seq)
    
    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        with pp.Party() as party:
            pool1 = from_iupac_motif('NNNN', mode='random').named('iupac')
        df1 = pool1.generate_seqs(num_seqs=10, seed=42)
        
        with pp.Party() as party:
            pool2 = from_iupac_motif('NNNN', mode='random').named('iupac')
        df2 = pool2.generate_seqs(num_seqs=10, seed=42)
        
        assert list(df1['seq']) == list(df2['seq'])


class TestFromIupacMotifMarkChanges:
    """Test mark_changes parameter."""
    
    def test_mark_changes_false_by_default(self):
        """Default mark_changes is False."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert pool.operation.mark_changes is False
    
    def test_mark_changes_true_swaps_degenerate_positions(self):
        """mark_changes=True swaps case at degenerate positions."""
        with pp.Party() as party:
            # A and T are fixed, N is degenerate (4 options)
            pool = from_iupac_motif('ANT', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            # Position 0 (A) and 2 (T) should be uppercase
            # Position 1 (N -> any base) should be lowercase
            assert seq[0] == 'A'
            assert seq[2] == 'T'
            assert seq[1].islower()
    
    def test_mark_changes_false_preserves_uppercase(self):
        """mark_changes=False preserves uppercase input."""
        with pp.Party() as party:
            pool = from_iupac_motif('ANT', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq == seq.upper()
    
    def test_mark_changes_false_preserves_lowercase(self):
        """mark_changes=False preserves lowercase input."""
        with pp.Party() as party:
            pool = from_iupac_motif('ant', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq == seq.lower()
    
    def test_mark_changes_false_preserves_mixed_case(self):
        """mark_changes=False preserves mixed case input."""
        with pp.Party() as party:
            pool = from_iupac_motif('AnT', mark_changes=False, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq[0] == 'A'
            assert seq[1].islower()
            assert seq[2] == 'T'
    
    def test_mark_changes_multiple_degenerate(self):
        """mark_changes works with multiple degenerate positions."""
        with pp.Party() as party:
            # R = A|G (degenerate), Y = C|T (degenerate)
            pool = from_iupac_motif('ARYT', mark_changes=True, mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        for seq in df['seq']:
            # Position 0 (A) and 3 (T) are fixed -> uppercase
            # Positions 1 (R) and 2 (Y) are degenerate -> lowercase
            assert seq[0] == 'A'
            assert seq[3] == 'T'
            assert seq[1].islower()
            assert seq[2].islower()
    
    def test_mark_changes_all_degenerate(self):
        """mark_changes with all degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac_motif('NN', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            # All positions are degenerate -> all lowercase
            assert seq == seq.lower()
    
    def test_mark_changes_no_degenerate(self):
        """mark_changes with no degenerate positions does nothing."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            # No degenerate positions -> all uppercase
            assert seq == seq.upper()
    
    def test_mark_changes_in_copy_params(self):
        """mark_changes is included in _get_copy_params."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', mark_changes=True)
            params = pool.operation._get_copy_params()
        assert params['mark_changes'] is True


class TestFromIupacMotifCustomName:
    """Test name parameters."""
    
    def test_default_operation_name(self):
        """Default operation name contains from_iupac_motif."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert pool.operation.name.startswith('op[')
            assert ':from_iupac_motif' in pool.operation.name
    
    def test_custom_operation_name(self):
        """Custom operation name."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', op_name='my_motif')
            assert pool.operation.name == 'my_motif'
    
    def test_custom_pool_name(self):
        """Custom pool name."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', name='my_pool')
            assert pool.name == 'my_pool'


class TestFromIupacMotifDesignCards:
    """Test design card output."""
    
    def test_iupac_state_in_output(self):
        """Design card contains iupac_state."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT', op_name='motif').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1, seed=42)
        assert 'mypool.op.key.iupac_state' in df.columns
    
    def test_design_card_keys_defined(self):
        """design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert 'iupac_state' in pool.operation.design_card_keys


class TestFromIupacMotifSeqLength:
    """Test sequence length computation."""
    
    def test_seq_length_simple(self):
        """seq_length equals IUPAC sequence length."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACGT')
            assert pool.operation.seq_length == 4
    
    def test_seq_length_with_degenerate(self):
        """seq_length works with degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac_motif('ANNN')
            assert pool.operation.seq_length == 4


class TestFromIupacMotifIgnoreChars:
    """Test handling of ignore characters."""
    
    def test_dot_separator_preserved(self):
        """Dot separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACG.TNN', mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Dot should be at position 3
            assert seq[3] == '.'
            assert len(seq) == 7
    
    def test_multiple_separators(self):
        """Multiple dot separators are preserved."""
        with pp.Party() as party:
            pool = from_iupac_motif('A.C.G.T', mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            assert seq == 'A.C.G.T'
    
    def test_dash_separator_preserved(self):
        """Dash separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac_motif('ACG-TNN', mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            assert seq[3] == '-'
    
    def test_ignore_chars_not_degenerate(self):
        """Ignore characters are not treated as degenerate."""
        with pp.Party() as party:
            pool = from_iupac_motif('A.N', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            # A (fixed) -> uppercase, . (ignore) -> unchanged, N (degenerate) -> lowercase
            assert seq[0] == 'A'
            assert seq[1] == '.'
            assert seq[2].islower()
    
    def test_num_states_ignores_separators(self):
        """Separators don't affect num_states calculation."""
        with pp.Party() as party:
            # N = 4 options, separators = 1 option each
            pool = from_iupac_motif('N.N', mode='sequential')
            # 4 * 1 * 4 = 16 states
            assert pool.operation.num_states == 16


class TestFromIupacMotifIgnoreChars:
    """Test handling of ignore characters."""
    
    def test_dot_separator_allowed(self):
        """Dot separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac_motif('AC.GT', mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        assert len(df) == 1  # Only one state since all positions are fixed
        assert df['seq'].iloc[0] == 'AC.GT'
    
    def test_hyphen_separator_allowed(self):
        """Hyphen separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac_motif('AC-GT', mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        assert df['seq'].iloc[0] == 'AC-GT'
    
    def test_space_separator_allowed(self):
        """Space separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac_motif('AC GT', mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        assert df['seq'].iloc[0] == 'AC GT'
    
    def test_ignore_chars_with_degenerate(self):
        """Ignore characters work alongside degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac_motif('A.N.T', mode='sequential').named('iupac')
        
        df = pool.generate_seqs(num_cycles=1)
        # N has 4 options, so 4 states total
        assert len(df) == 4
        # Check that dots are preserved
        for seq in df['seq']:
            assert seq[1] == '.'
            assert seq[3] == '.'
    
    def test_mark_changes_ignores_separators(self):
        """mark_changes does not affect separator positions."""
        with pp.Party() as party:
            pool = from_iupac_motif('A.N.T', mark_changes=True, mode='random').named('iupac')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            # Separators should remain unchanged
            assert seq[1] == '.'
            assert seq[3] == '.'
            # Fixed positions stay uppercase
            assert seq[0] == 'A'
            assert seq[4] == 'T'
            # Degenerate position should be lowercase
            assert seq[2].islower()
