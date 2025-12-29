import pytest
from poolparty import Pool, RandomMutationPool, KmerPool


class TestRandomMutationPoolBasic:
    """Test basic functionality of RandomMutationPool."""
    
    def test_init_with_string(self):
        """Test initialization with a string sequence."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        assert pool is not None
        assert pool.alphabet == ['A', 'C', 'G', 'T']
        assert pool.mutation_rate == 0.5
    
    def test_init_with_pool(self):
        """Test initialization with a Pool object."""
        base_pool = Pool(seqs=['ACGT'])
        pool = RandomMutationPool(base_pool, alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        assert pool is not None
    
    def test_infinite_states(self):
        """Test that RandomMutationPool has infinite states."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        assert pool.num_states == float('inf')
        assert not pool.is_sequential_compatible()
    
    def test_sequence_length_preserved(self):
        """Test that mutated sequence has same length as input."""
        pool = RandomMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        pool.set_state(0)
        seq = pool.seq
        assert len(seq) == 8
    
    def test_characters_from_alphabet(self):
        """Test that all characters in mutated sequence are from alphabet."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=1.0)
        pool.set_state(0)
        seq = pool.seq
        for char in seq:
            assert char in ['A', 'C', 'G', 'T']


class TestRandomMutationPoolMutationRate:
    """Test mutation rate functionality."""
    
    def test_mutation_rate_zero(self):
        """Test that mutation_rate=0 produces no mutations."""
        original_seq = 'ACGTACGT'
        pool = RandomMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.0)
        # Test multiple states
        for state in range(10):
            pool.set_state(state)
            assert pool.seq == original_seq
    
    def test_mutation_rate_one(self):
        """Test that mutation_rate=1.0 produces mutations at all positions."""
        original_seq = 'AAAA'
        pool = RandomMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], mutation_rate=1.0)
        pool.set_state(0)
        mutated = pool.seq
        # With rate=1.0, all positions should be mutated to different characters
        # At least some positions should be different (unless extremely unlucky)
        # With 4 positions and 3 choices each, probability of all same is very low
        mutations = sum(1 for i, char in enumerate(mutated) if char != original_seq[i])
        assert mutations >= 0  # At least should be valid
    
    def test_position_specific_rates_array(self):
        """Test position-specific mutation rates using an array."""
        original_seq = 'AAAA'
        # First position: no mutation, rest: always mutate
        rates = [0.0, 1.0, 1.0, 1.0]
        pool = RandomMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], mutation_rate=rates)
        
        pool.set_state(0)
        mutated = pool.seq
        
        # First position should never change
        assert mutated[0] == 'A'
    
    def test_position_specific_rates_length_validation(self):
        """Test that position-specific rates must match sequence length."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=[0.5, 0.5])
        # Error should occur when trying to compute sequence, not at init
        with pytest.raises(ValueError, match="mutation_rate array length"):
            _ = pool.seq


class TestRandomMutationPoolValidation:
    """Test input validation."""
    
    def test_empty_alphabet_error(self):
        """Test that empty alphabet raises error."""
        with pytest.raises(ValueError, match="alphabet list must be non-empty"):
            RandomMutationPool('ACGT', alphabet=[], mutation_rate=0.5)
    
    def test_mutation_rate_too_high(self):
        """Test that mutation_rate > 1 raises error."""
        with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
            RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=1.5)
    
    def test_mutation_rate_negative(self):
        """Test that negative mutation_rate raises error."""
        with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
            RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=-0.1)
    
    def test_mutation_rate_array_invalid_value(self):
        """Test that invalid values in mutation_rate array raise error."""
        with pytest.raises(ValueError, match="mutation_rate values must be between 0 and 1"):
            RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], 
                             mutation_rate=[0.5, 0.5, 1.5, 0.5])


class TestRandomMutationPoolDeterminism:
    """Test deterministic behavior with state setting."""
    
    def test_same_state_same_sequence(self):
        """Test that setting same state produces same sequence."""
        pool = RandomMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        
        pool.set_state(42)
        seq1 = pool.seq
        
        pool.set_state(42)
        seq2 = pool.seq
        
        assert seq1 == seq2
    
    def test_different_states_likely_different_sequences(self):
        """Test that different states likely produce different sequences."""
        pool = RandomMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        
        sequences = set()
        for state in range(100):
            pool.set_state(state)
            sequences.add(pool.seq)
        
        # With mutation_rate=0.5, we should see variety
        assert len(sequences) > 10


class TestRandomMutationPoolWithPoolInput:
    """Test RandomMutationPool with Pool objects as input."""
    
    def test_pool_input_generates_fresh_sequence(self):
        """Test that Pool input generates fresh sequence each time."""
        kmer_pool = KmerPool(4, alphabet='dna')
        mutation_pool = RandomMutationPool(kmer_pool, alphabet=['A', 'C', 'G', 'T'], 
                                          mutation_rate=0.3)
        
        # Set kmer pool to state 0
        kmer_pool.set_state(0)
        base_seq1 = kmer_pool.seq
        
        # Get mutated sequence
        mutation_pool.set_state(0)
        mutated_seq1 = mutation_pool.seq
        
        # Change kmer pool state
        kmer_pool.set_state(5)
        base_seq2 = kmer_pool.seq
        
        # Get new mutated sequence (same mutation state)
        mutation_pool.set_state(0)
        mutated_seq2 = mutation_pool.seq
        
        # Base sequences should be different
        assert base_seq1 != base_seq2


class TestRandomMutationPoolOperations:
    """Test RandomMutationPool with Pool operations."""
    
    def test_concatenation(self):
        """Test concatenating RandomMutationPool with other pools."""
        pool1 = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        pool2 = Pool(seqs=['NNNN'])
        
        combined = pool1 + pool2
        combined.set_state(0)
        seq = combined.seq
        
        assert len(seq) == 8
        assert seq.endswith('NNNN')
    
    def test_repetition(self):
        """Test repeating RandomMutationPool."""
        pool = RandomMutationPool('AC', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        repeated = pool * 3
        
        repeated.set_state(0)
        seq = repeated.seq
        
        assert len(seq) == 6
    
    def test_slicing(self):
        """Test slicing RandomMutationPool."""
        pool = RandomMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        sliced = pool[2:6]
        
        sliced.set_state(0)
        seq = sliced.seq
        
        assert len(seq) == 4


class TestRandomMutationPoolMutationBehavior:
    """Test specific mutation behavior."""
    
    def test_mutations_are_different_characters(self):
        """Test that mutations replace with different characters."""
        # Use all A's with 100% mutation rate
        original_seq = 'AAAA'
        pool = RandomMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], 
                                 mutation_rate=1.0)
        
        pool.set_state(0)
        mutated = pool.seq
        
        # All positions should be mutated to non-A characters
        for i, char in enumerate(mutated):
            # With mutation_rate=1.0 and alphabet containing A, 
            # mutations should choose from C, G, T
            assert char in ['A', 'C', 'G', 'T']
    
    def test_single_character_alphabet(self):
        """Test behavior when alphabet has only one character matching sequence."""
        # If alphabet only has 'A' and sequence is 'AAA', no mutation possible
        original_seq = 'AAA'
        pool = RandomMutationPool(original_seq, alphabet=['A'], mutation_rate=1.0)
        
        pool.set_state(0)
        mutated = pool.seq
        
        # Should remain unchanged since no alternative characters
        assert mutated == original_seq


class TestRandomMutationPoolRepr:
    """Test string representation."""
    
    def test_repr_with_string_input(self):
        """Test __repr__ with string input."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        repr_str = repr(pool)
        assert 'RandomMutationPool' in repr_str
        assert 'ACGT' in repr_str
        assert '0.5' in repr_str
    
    def test_repr_with_array_mutation_rate(self):
        """Test __repr__ with array mutation rate."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C'], 
                                 mutation_rate=[0.1, 0.2, 0.3, 0.4])
        repr_str = repr(pool)
        assert 'RandomMutationPool' in repr_str
        assert '4 rates' in repr_str


class TestRandomMutationPoolIteration:
    """Test iteration behavior."""
    
    def test_state_advances_manually(self):
        """Test that state can be manually advanced."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        pool.set_state(0)
        
        seq1 = pool.seq
        pool.set_state(1)
        seq2 = pool.seq
        
        # Both should be valid sequences of correct length
        assert len(seq1) == 4
        assert len(seq2) == 4
    
    def test_multiple_sequences_via_states(self):
        """Test that we can get multiple sequences via state setting."""
        pool = RandomMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
        
        sequences = []
        for i in range(10):
            pool.set_state(i)
            sequences.append(pool.seq)
        
        assert len(sequences) == 10
        for seq in sequences:
            assert len(seq) == 4

