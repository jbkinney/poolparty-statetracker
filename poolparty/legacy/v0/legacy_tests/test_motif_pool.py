import pytest
import pandas as pd
import numpy as np
from poolparty import Pool, MotifPool, KmerPool


class TestMotifPoolBasic:
    """Test basic functionality of MotifPool."""
    
    def test_init_with_valid_matrix(self):
        """Test initialization with a valid probability matrix."""
        df = pd.DataFrame({
            'A': [0.25, 0.5, 0.1],
            'C': [0.25, 0.25, 0.2],
            'G': [0.25, 0.15, 0.3],
            'T': [0.25, 0.1, 0.4]
        })
        pool = MotifPool(df)
        assert pool is not None
        assert pool.alphabet == ['A', 'C', 'G', 'T']
        assert len(pool.probability_df) == 3
    
    def test_infinite_states(self):
        """Test that MotifPool has infinite states."""
        df = pd.DataFrame({
            'A': [0.5, 0.5],
            'C': [0.5, 0.5]
        })
        pool = MotifPool(df)
        assert pool.num_states == float('inf')
        assert not pool.is_sequential_compatible()
    
    def test_sequence_length(self):
        """Test that sequence length equals number of rows in probability_df."""
        df = pd.DataFrame({
            'A': [0.25, 0.5, 0.1, 0.7, 0.3],
            'C': [0.25, 0.25, 0.2, 0.1, 0.3],
            'G': [0.25, 0.15, 0.3, 0.1, 0.2],
            'T': [0.25, 0.1, 0.4, 0.1, 0.2]
        })
        pool = MotifPool(df)
        assert pool.seq_length == 5
        pool.set_state(0)
        assert len(pool.seq) == 5
    
    def test_characters_from_alphabet(self):
        """Test that all characters in generated sequence are from alphabet."""
        df = pd.DataFrame({
            'A': [0.25, 0.25, 0.25, 0.25],
            'C': [0.25, 0.25, 0.25, 0.25],
            'G': [0.25, 0.25, 0.25, 0.25],
            'T': [0.25, 0.25, 0.25, 0.25]
        })
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        for char in seq:
            assert char in ['A', 'C', 'G', 'T']


class TestMotifPoolValidation:
    """Test input validation."""
    
    def test_empty_dataframe_error(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="probability_df must be a non-empty DataFrame"):
            MotifPool(df)
    
    def test_no_rows_error(self):
        """Test that DataFrame with no rows raises error."""
        df = pd.DataFrame(columns=['A', 'C', 'G', 'T'])
        with pytest.raises(ValueError, match="probability_df must be a non-empty DataFrame"):
            MotifPool(df)
    
    def test_no_columns_error(self):
        """Test that DataFrame with no columns raises error."""
        # This is a bit tricky - need to create a DataFrame with rows but no columns
        df = pd.DataFrame(index=[0, 1, 2])
        with pytest.raises(ValueError, match="probability_df must be a non-empty DataFrame"):
            MotifPool(df)
    
    def test_column_not_string_error(self):
        """Test that non-string column names raise error."""
        df = pd.DataFrame({
            1: [0.5, 0.5],
            2: [0.5, 0.5]
        })
        with pytest.raises(ValueError, match="All column names must be strings"):
            MotifPool(df)
    
    def test_column_not_single_character_error(self):
        """Test that multi-character column names raise error."""
        df = pd.DataFrame({
            'AA': [0.5, 0.5],
            'C': [0.5, 0.5]
        })
        with pytest.raises(ValueError, match="All column names must be single characters"):
            MotifPool(df)
    
    def test_duplicate_columns_error(self):
        """Test that duplicate column names raise error."""
        # Create DataFrame with duplicate columns
        df = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=['A', 'A'])
        with pytest.raises(ValueError, match="All column names must be unique"):
            MotifPool(df)
    
    def test_row_sum_not_one_error(self):
        """Test that rows not summing to 1.0 raise error."""
        df = pd.DataFrame({
            'A': [0.25, 0.5],
            'C': [0.25, 0.25],
            'G': [0.25, 0.15],
            'T': [0.25, 0.05]  # Second row sums to 0.95
        })
        with pytest.raises(ValueError, match="All rows in probability_df must sum to approximately 1.0"):
            MotifPool(df)
    
    def test_row_sum_close_to_one_accepted(self):
        """Test that rows very close to 1.0 are accepted."""
        df = pd.DataFrame({
            'A': [0.25, 0.5],
            'C': [0.25, 0.25],
            'G': [0.25, 0.15],
            'T': [0.25, 0.1 + 1e-10]  # Close to 1.0
        })
        pool = MotifPool(df)
        assert pool is not None
    
    def test_sequential_mode_error(self):
        """Test that sequential mode raises error."""
        df = pd.DataFrame({
            'A': [0.5, 0.5],
            'C': [0.5, 0.5]
        })
        with pytest.raises(ValueError, match="MotifPool only supports mode='random'"):
            MotifPool(df, mode='sequential')


class TestMotifPoolDeterminism:
    """Test deterministic behavior with state setting."""
    
    def test_same_state_same_sequence(self):
        """Test that setting same state produces same sequence."""
        df = pd.DataFrame({
            'A': [0.25, 0.5, 0.1],
            'C': [0.25, 0.25, 0.2],
            'G': [0.25, 0.15, 0.3],
            'T': [0.25, 0.1, 0.4]
        })
        pool = MotifPool(df)
        
        pool.set_state(42)
        seq1 = pool.seq
        
        pool.set_state(42)
        seq2 = pool.seq
        
        assert seq1 == seq2
    
    def test_different_states_produce_different_sequences(self):
        """Test that different states produce different sequences."""
        df = pd.DataFrame({
            'A': [0.25, 0.25, 0.25],
            'C': [0.25, 0.25, 0.25],
            'G': [0.25, 0.25, 0.25],
            'T': [0.25, 0.25, 0.25]
        })
        pool = MotifPool(df)
        
        sequences = set()
        for state in range(100):
            pool.set_state(state)
            sequences.add(pool.seq)
        
        # With uniform probabilities, we should see variety
        assert len(sequences) > 10


class TestMotifPoolProbabilityDistributions:
    """Test that sampling respects probability distributions."""
    
    def test_deterministic_position(self):
        """Test position with probability 1.0 for one character."""
        df = pd.DataFrame({
            'A': [1.0, 0.25, 0.25],
            'C': [0.0, 0.25, 0.25],
            'G': [0.0, 0.25, 0.25],
            'T': [0.0, 0.25, 0.25]
        })
        pool = MotifPool(df)
        
        # Generate many sequences - first position should always be 'A'
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            assert seq[0] == 'A', f"Expected first position to be 'A', got '{seq[0]}'"
    
    def test_impossible_character(self):
        """Test that character with probability 0.0 never appears."""
        df = pd.DataFrame({
            'A': [0.0, 0.5, 0.5],
            'C': [1.0, 0.5, 0.5]
        })
        pool = MotifPool(df)
        
        # Generate many sequences - first position should never be 'A'
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            assert seq[0] != 'A', f"Expected first position to not be 'A', got '{seq[0]}'"
            assert seq[0] == 'C', f"Expected first position to be 'C', got '{seq[0]}'"
    
    def test_probability_distribution_sampling(self):
        """Test that sampling approximately follows probability distribution."""
        # Create a highly biased distribution
        df = pd.DataFrame({
            'A': [0.9],  # 90% chance of A
            'C': [0.1]   # 10% chance of C
        })
        pool = MotifPool(df)
        
        # Generate many sequences and count occurrences
        counts = {'A': 0, 'C': 0}
        n_samples = 1000
        for state in range(n_samples):
            pool.set_state(state)
            seq = pool.seq
            counts[seq[0]] += 1
        
        # Check that A appears approximately 90% of the time
        # Using a loose tolerance since this is probabilistic
        a_fraction = counts['A'] / n_samples
        assert 0.85 < a_fraction < 0.95, f"Expected ~90% A, got {a_fraction*100:.1f}%"


class TestMotifPoolOperations:
    """Test MotifPool with Pool operations."""
    
    def test_concatenation(self):
        """Test concatenating MotifPool with other pools."""
        df = pd.DataFrame({
            'A': [0.5, 0.5],
            'C': [0.5, 0.5]
        })
        pool1 = MotifPool(df)
        pool2 = Pool(seqs=['NNNN'])
        
        combined = pool1 + pool2
        combined.set_state(0)
        seq = combined.seq
        
        assert len(seq) == 6
        assert seq.endswith('NNNN')
    
    def test_repetition(self):
        """Test repeating MotifPool."""
        df = pd.DataFrame({
            'A': [0.5],
            'C': [0.5]
        })
        pool = MotifPool(df)
        repeated = pool * 3
        
        repeated.set_state(0)
        seq = repeated.seq
        
        assert len(seq) == 3
    
    def test_slicing(self):
        """Test slicing MotifPool."""
        df = pd.DataFrame({
            'A': [0.25] * 8,
            'C': [0.25] * 8,
            'G': [0.25] * 8,
            'T': [0.25] * 8
        })
        pool = MotifPool(df)
        sliced = pool[2:6]
        
        sliced.set_state(0)
        seq = sliced.seq
        
        assert len(seq) == 4
    
    def test_concatenation_with_kmer_pool(self):
        """Test concatenating MotifPool with KmerPool."""
        df = pd.DataFrame({
            'A': [1.0, 0.0],
            'T': [0.0, 1.0]
        })
        motif_pool = MotifPool(df)
        kmer_pool = KmerPool(3, alphabet='dna')
        
        combined = motif_pool + kmer_pool
        combined.set_state(0)
        seq = combined.seq
        
        assert len(seq) == 5
        assert seq[:2] == 'AT'  # From motif (deterministic)


class TestMotifPoolRepr:
    """Test string representation."""
    
    def test_repr_basic(self):
        """Test __repr__ includes key information."""
        df = pd.DataFrame({
            'A': [0.25, 0.5, 0.1],
            'C': [0.25, 0.25, 0.2],
            'G': [0.25, 0.15, 0.3],
            'T': [0.25, 0.1, 0.4]
        })
        pool = MotifPool(df)
        repr_str = repr(pool)
        
        assert 'MotifPool' in repr_str
        assert '3x4' in repr_str  # 3 rows, 4 columns
        assert "['A', 'C', 'G', 'T']" in repr_str


class TestMotifPoolIteration:
    """Test iteration behavior."""
    
    def test_state_advances_manually(self):
        """Test that state can be manually advanced."""
        df = pd.DataFrame({
            'A': [0.5],
            'C': [0.5]
        })
        pool = MotifPool(df)
        pool.set_state(0)
        
        seq1 = pool.seq
        pool.set_state(1)
        seq2 = pool.seq
        
        # Different states should generally produce different sequences (probabilistically)
        # Both should be valid sequences
        assert len(seq1) == 1
        assert len(seq2) == 1
    
    def test_multiple_sequences_via_states(self):
        """Test that we can get multiple sequences via state setting."""
        df = pd.DataFrame({
            'A': [0.25, 0.25],
            'C': [0.25, 0.25],
            'G': [0.25, 0.25],
            'T': [0.25, 0.25]
        })
        pool = MotifPool(df)
        
        sequences = []
        for i in range(10):
            pool.set_state(i)
            sequences.append(pool.seq)
        
        assert len(sequences) == 10
        for seq in sequences:
            assert len(seq) == 2


class TestMotifPoolEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_position_matrix(self):
        """Test with a 1-position matrix."""
        df = pd.DataFrame({
            'A': [0.5],
            'T': [0.5]
        })
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        
        assert len(seq) == 1
        assert seq in ['A', 'T']
    
    def test_single_character_alphabet(self):
        """Test with only one character in alphabet."""
        df = pd.DataFrame({
            'X': [1.0, 1.0, 1.0]
        })
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        
        assert seq == 'XXX'
    
    def test_non_standard_characters(self):
        """Test with non-standard characters in alphabet."""
        df = pd.DataFrame({
            'X': [0.33, 0.33],
            'Y': [0.33, 0.33],
            'Z': [0.34, 0.34]
        })
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        
        assert len(seq) == 2
        for char in seq:
            assert char in ['X', 'Y', 'Z']
    
    def test_dataframe_not_modified(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'A': [0.5, 0.5],
            'C': [0.5, 0.5]
        })
        df_copy = df.copy()
        
        pool = MotifPool(df)
        pool.set_state(0)
        _ = pool.seq
        
        # Original DataFrame should not be modified
        pd.testing.assert_frame_equal(df, df_copy)


class TestMotifPoolLogomakerCompatibility:
    """Test compatibility with Logomaker-style probability matrices."""
    
    def test_logomaker_style_matrix(self):
        """Test with a matrix in Logomaker's style (positions as index)."""
        # Logomaker typically uses integer indices for positions
        df = pd.DataFrame({
            'A': [0.1, 0.7, 0.1],
            'C': [0.2, 0.1, 0.2],
            'G': [0.3, 0.1, 0.3],
            'T': [0.4, 0.1, 0.4]
        }, index=[0, 1, 2])
        
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        
        assert len(seq) == 3
        assert all(c in 'ACGT' for c in seq)
    
    def test_dna_alphabet(self):
        """Test with standard DNA alphabet."""
        df = pd.DataFrame({
            'A': [0.25, 0.25, 0.25],
            'C': [0.25, 0.25, 0.25],
            'G': [0.25, 0.25, 0.25],
            'T': [0.25, 0.25, 0.25]
        })
        pool = MotifPool(df)
        
        assert set(pool.alphabet) == {'A', 'C', 'G', 'T'}
    
    def test_protein_alphabet(self):
        """Test with a subset of protein alphabet."""
        df = pd.DataFrame({
            'A': [0.2] * 5,
            'C': [0.2] * 5,
            'D': [0.2] * 5,
            'E': [0.2] * 5,
            'F': [0.2] * 5
        })
        pool = MotifPool(df)
        pool.set_state(0)
        seq = pool.seq
        
        assert len(seq) == 5
        assert all(c in 'ACDEF' for c in seq)
