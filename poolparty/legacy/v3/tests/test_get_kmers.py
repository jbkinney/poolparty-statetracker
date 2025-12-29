"""Tests for the GetKmers operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.get_kmers import GetKmersOp, get_kmers


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestGetKmersFactory:
    """Test get_kmers factory function."""
    
    def test_returns_pool(self):
        """Test that get_kmers returns a Pool."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_get_kmers_op(self):
        """Test that get_kmers creates a GetKmersOp."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert isinstance(pool.operation, GetKmersOp)


class TestGetKmersSequentialMode:
    """Test GetKmers in sequential mode."""
    
    def test_sequential_all_kmers(self):
        """Test sequential mode generates all k-mers in order."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='AB', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 4  # 2^2 = 4
        assert list(df['kmer']) == ['AA', 'AB', 'BA', 'BB']
    
    def test_sequential_dna_2mers(self):
        """Test sequential mode with DNA 2-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='dna', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 16  # 4^2 = 16
        # First should be 'AA', last should be 'TT'
        assert df['kmer'].iloc[0] == 'AA'
        assert df['kmer'].iloc[-1] == 'TT'
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = get_kmers(length=1, alphabet='AB', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=5)
        assert list(df['kmer']) == ['A', 'B', 'A', 'B', 'A']
    
    def test_sequential_num_states(self):
        """Test num_states calculation."""
        with pp.Party() as party:
            pool = get_kmers(length=3, alphabet='dna', mode='sequential')
            assert pool.operation.num_states == 64  # 4^3


class TestGetKmersRandomMode:
    """Test GetKmers in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=5, alphabet='dna', mode='random')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=100, seed=42)
        assert len(df) == 100
        # All should be valid 5-mers
        for kmer in df['kmer']:
            assert len(kmer) == 5
            assert all(c in 'ACGT' for c in kmer)
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            pool = get_kmers(length=4, alphabet='dna', mode='random')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=100, seed=42)
        unique_kmers = df['kmer'].nunique()
        assert unique_kmers > 50  # Should be quite varied
    
    def test_random_reproducible(self):
        """Test that random mode is reproducible with seed."""
        with pp.Party() as party:
            pool = get_kmers(length=4, alphabet='dna', mode='random')
            party.output(pool, name='kmer')
        
        df1 = party.generate(num_seqs=10, seed=42, init_state=0)
        party.reset()
        df2 = party.generate(num_seqs=10, seed=42, init_state=0)
        
        assert list(df1['kmer']) == list(df2['kmer'])
    
    def test_random_requires_rng(self):
        """Test that random mode requires RNG."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='random')
        
        with pytest.raises(RuntimeError, match="Random mode requires RNG"):
            pool.operation.compute([], 0, None)


class TestGetKmersAlphabets:
    """Test GetKmers with different alphabets."""
    
    def test_dna_alphabet(self):
        """Test DNA alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=3, alphabet='dna', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=10)
        for kmer in df['kmer']:
            assert all(c in 'ACGT' for c in kmer)
    
    def test_rna_alphabet(self):
        """Test RNA alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=3, alphabet='rna', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=10)
        for kmer in df['kmer']:
            assert all(c in 'ACGU' for c in kmer)
    
    def test_protein_alphabet(self):
        """Test protein alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='protein', mode='random')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=10, seed=42)
        for kmer in df['kmer']:
            assert len(kmer) == 2
            # Just check they're valid amino acids
            assert all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in kmer)
    
    def test_binary_alphabet(self):
        """Test binary alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=4, alphabet='binary', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 16  # 2^4
        assert all(all(c in '01' for c in kmer) for kmer in df['kmer'])
    
    def test_custom_alphabet_string(self):
        """Test custom alphabet from string."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='XY', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert list(df['kmer']) == ['XX', 'XY', 'YX', 'YY']
    
    def test_custom_alphabet_list(self):
        """Test custom alphabet from list."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet=['A', 'B', 'C'], mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 9  # 3^2


class TestGetKmersStateToKmer:
    """Test the _state_to_kmer conversion method."""
    
    def test_state_to_kmer_binary(self):
        """Test state to k-mer conversion for binary alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=3, alphabet='01', mode='sequential')
            op = pool.operation
            
            assert op._state_to_kmer(0) == '000'
            assert op._state_to_kmer(1) == '001'
            assert op._state_to_kmer(2) == '010'
            assert op._state_to_kmer(7) == '111'
    
    def test_state_to_kmer_ab(self):
        """Test state to k-mer conversion for AB alphabet."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='AB', mode='sequential')
            op = pool.operation
            
            assert op._state_to_kmer(0) == 'AA'
            assert op._state_to_kmer(1) == 'AB'
            assert op._state_to_kmer(2) == 'BA'
            assert op._state_to_kmer(3) == 'BB'


class TestGetKmersDesignCards:
    """Test GetKmers design card output."""
    
    def test_kmer_index_in_output(self):
        """Test kmer_index is in output."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='AB', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=4)
        assert 'get_kmers.kmer_index' in df.columns
        assert list(df['get_kmers.kmer_index']) == [0, 1, 2, 3]
    
    def test_design_card_keys_defined(self):
        """Test design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert 'kmer_index' in pool.operation.design_card_keys


class TestGetKmersErrors:
    """Test GetKmers error handling."""
    
    def test_length_zero_error(self):
        """Test error for length=0."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            get_kmers(length=0)
    
    def test_length_negative_error(self):
        """Test error for negative length."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            get_kmers(length=-1)


class TestGetKmersLargeSpace:
    """Test GetKmers with large state spaces."""
    
    def test_large_kmer_random_mode(self):
        """Test large k-mer space works in random mode."""
        with pp.Party() as party:
            # 4^20 = huge number, but random mode should work
            pool = get_kmers(length=20, alphabet='dna', mode='random')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=10, seed=42)
        assert len(df) == 10
        for kmer in df['kmer']:
            assert len(kmer) == 20
    
    def test_large_kmer_returns_minus_one_states(self):
        """Test that large k-mer space returns -1 for num_states."""
        with pp.Party() as party:
            # Should exceed max_num_sequential_states
            pool = get_kmers(length=20, alphabet='dna', mode='random')
            assert pool.operation.num_states == -1


class TestGetKmersCompute:
    """Test GetKmers compute method directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = get_kmers(length=2, alphabet='AB', mode='sequential')
        
        result = pool.operation.compute([], 0, None)
        assert result['seq_0'] == 'AA'
        assert result['kmer_index'] == 0
        
        result = pool.operation.compute([], 1, None)
        assert result['seq_0'] == 'AB'
        assert result['kmer_index'] == 1
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            pool = get_kmers(length=4, alphabet='dna', mode='random')
        
        rng = np.random.default_rng(42)
        result = pool.operation.compute([], 0, rng)
        assert len(result['seq_0']) == 4
        assert all(c in 'ACGT' for c in result['seq_0'])


class TestGetKmersCustomName:
    """Test GetKmers operation name parameter."""
    
    def test_default_operation_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = get_kmers(length=4)
            assert pool.operation.name == 'get_kmers'
    
    def test_custom_operation_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = get_kmers(length=4, name='barcode')
            assert pool.operation.name == 'barcode'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns."""
        with pp.Party() as party:
            pool = get_kmers(length=4, name='my_barcode')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=1, seed=42)
        assert 'my_barcode.kmer_index' in df.columns

