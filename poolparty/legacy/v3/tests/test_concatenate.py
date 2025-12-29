"""Tests for the Concatenate operation."""

import pytest
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.concatenate import ConcatenateOp, concatenate


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestConcatenateFactory:
    """Test concatenate factory function."""
    
    def test_returns_pool(self):
        """Test that concatenate returns a Pool."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert combined is not None
            assert hasattr(combined, 'operation')
    
    def test_creates_concatenate_op(self):
        """Test that concatenate creates a ConcatenateOp."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert isinstance(combined.operation, ConcatenateOp)


class TestConcatenatePools:
    """Test concatenating multiple pools."""
    
    def test_two_pools(self):
        """Test concatenating two pools."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = concatenate([left, right])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTT'
    
    def test_three_pools(self):
        """Test concatenating three pools."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            c = pp.from_seqs(['GGG'])
            combined = concatenate([a, b, c])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTTGGG'
    
    def test_many_pools(self):
        """Test concatenating many pools."""
        with pp.Party() as party:
            pools = [pp.from_seqs([c]) for c in 'ABCDE']
            combined = concatenate(pools)
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCDE'


class TestConcatenateWithStrings:
    """Test concatenating pools with string literals."""
    
    def test_pool_and_string(self):
        """Test concatenating pool with string."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            combined = concatenate([pool, '...'])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...'
    
    def test_string_and_pool(self):
        """Test concatenating string with pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['TTT'])
            combined = concatenate(['...', pool])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == '...TTT'
    
    def test_pool_string_pool(self):
        """Test pool-string-pool pattern."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = concatenate([left, '...', right])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...TTT'
    
    def test_multiple_strings(self):
        """Test multiple string literals."""
        with pp.Party() as party:
            pool = pp.from_seqs(['X'])
            combined = concatenate(['[', pool, ']'])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == '[X]'
    
    def test_only_strings(self):
        """Test concatenating only strings (converts to pools)."""
        with pp.Party() as party:
            combined = concatenate(['A', 'B', 'C'])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABC'


class TestConcatenateOperatorEquivalence:
    """Test that operator syntax equals function syntax."""
    
    def test_add_operator_equivalent(self):
        """Test Pool + Pool equals concatenate([Pool, Pool])."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            
            via_operator = left + right
            party.output(via_operator, name='via_operator')
        
        df1 = party.generate(num_seqs=1)
        
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            
            via_function = concatenate([left, right])
            party.output(via_function, name='via_function')
        
        df2 = party.generate(num_seqs=1)
        
        assert df1['via_operator'].iloc[0] == df2['via_function'].iloc[0]
    
    def test_add_with_string_equivalent(self):
        """Test Pool + str equals concatenate([Pool, str])."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            
            via_operator = pool + '...'
            party.output(via_operator, name='seq')
        
        df1 = party.generate(num_seqs=1)
        
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            
            via_function = concatenate([pool, '...'])
            party.output(via_function, name='seq')
        
        df2 = party.generate(num_seqs=1)
        
        assert df1['seq'].iloc[0] == df2['seq'].iloc[0]


class TestConcatenateFixedMode:
    """Test that ConcatenateOp is always fixed mode."""
    
    def test_mode_is_fixed(self):
        """Test that concatenate creates fixed mode operation."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert combined.operation.mode == 'fixed'
    
    def test_num_states_is_one(self):
        """Test that concatenate has num_states=1."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert combined.operation.num_states == 1
    
    def test_variability_from_parents(self):
        """Test that variability comes from parent pools."""
        with pp.Party() as party:
            # Each parent has 2 states = 4 combinations
            left = pp.from_seqs(['A', 'B'], mode='sequential')
            right = pp.from_seqs(['X', 'Y'], mode='sequential')
            combined = concatenate([left, right])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=4)
        # Should get all 4 combinations
        expected = {'AX', 'BY', 'AY', 'BX'}
        assert set(df['seq']) == expected or len(set(df['seq'])) == 4


class TestConcatenateDesignCards:
    """Test ConcatenateOp design cards."""
    
    def test_no_design_card_keys(self):
        """Test that concatenate has no design card keys."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert len(combined.operation.design_card_keys) == 0
    
    def test_parent_design_cards_preserved(self):
        """Test that parent design cards are still included."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'], names=['seq_a'])
            b = pp.from_seqs(['TTT'], names=['seq_b'])
            combined = concatenate([a, b])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        # Parent design cards should be present
        assert 'from_seqs.seq_name' in df.columns or len([c for c in df.columns if 'seq_name' in c]) > 0


class TestConcatenateCompute:
    """Test ConcatenateOp compute method directly."""
    
    def test_compute_joins_sequences(self):
        """Test compute method joins parent sequences."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
        
        result = combined.operation.compute(['AAA', 'TTT'], 0, None)
        assert result['seq_0'] == 'AAATTT'
    
    def test_compute_empty_string(self):
        """Test compute with empty string parent."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs([''])
            combined = concatenate([a, b])
        
        result = combined.operation.compute(['AAA', ''], 0, None)
        assert result['seq_0'] == 'AAA'
    
    def test_compute_many_parents(self):
        """Test compute with many parent sequences."""
        with pp.Party() as party:
            pools = [pp.from_seqs([c]) for c in 'ABCDE']
            combined = concatenate(pools)
        
        result = combined.operation.compute(['A', 'B', 'C', 'D', 'E'], 0, None)
        assert result['seq_0'] == 'ABCDE'


class TestConcatenateChaining:
    """Test chaining concatenation operations."""
    
    def test_nested_concatenation(self):
        """Test nested concatenation operations."""
        with pp.Party() as party:
            a = pp.from_seqs(['A'])
            b = pp.from_seqs(['B'])
            c = pp.from_seqs(['C'])
            
            ab = concatenate([a, b])
            abc = concatenate([ab, c])
            party.output(abc, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABC'
    
    def test_chained_operators(self):
        """Test chained + operators."""
        with pp.Party() as party:
            a = pp.from_seqs(['A'])
            b = pp.from_seqs(['B'])
            c = pp.from_seqs(['C'])
            d = pp.from_seqs(['D'])
            
            combined = a + b + c + d
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCD'


class TestConcatenateWithOtherOperations:
    """Test concatenation with other operation types."""
    
    def test_with_mutation_scan(self):
        """Test concatenating with mutation scan output."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutation_scan(seq, k=1, mode='sequential')
            barcode = pp.from_seqs(['NNNN'])
            combined = concatenate([mutants, '.', barcode])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=3)
        for seq in df['seq']:
            assert seq.endswith('.NNNN')
            assert len(seq) == 9  # 4 + 1 + 4
    
    def test_with_get_kmers(self):
        """Test concatenating with k-mers."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            combined = concatenate([seq, '...', barcode])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert seq.startswith('ACGT...')
            assert len(seq) == 11  # 4 + 3 + 4
    
    def test_with_breakpoint_scan(self):
        """Test concatenating breakpoint scan outputs."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            combined = concatenate([left, '---', right])
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=3)
        for seq in df['seq']:
            assert '---' in seq


class TestConcatenateCustomName:
    """Test ConcatenateOp name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b])
            assert combined.operation.name == 'concat'
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = concatenate([a, b], name='my_concat')
            assert combined.operation.name == 'my_concat'

