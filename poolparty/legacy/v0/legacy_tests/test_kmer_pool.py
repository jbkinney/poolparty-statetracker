"""Tests for the KmerPool class."""

import pytest
from poolparty import KmerPool


def test_kmer_pool_creation():
    """Test KmerPool creation."""
    pool = KmerPool(length=10)
    assert pool.length == 10
    assert pool.alphabet == ['A', 'C', 'G', 'T']
    assert pool.num_states == 4**10  # 1,048,576 states
    # Note: 4^10 > DEFAULT_MAX_NUM_STATES (1,000,000), so not sequential-compatible by default
    assert not pool.is_sequential_compatible()
    
    # Can make it sequential-compatible by increasing max_num_states
    pool.set_max_num_states(2_000_000)
    assert pool.is_sequential_compatible()


def test_kmer_pool_custom_alphabet():
    """Test KmerPool with custom alphabet."""
    pool = KmerPool(length=5, alphabet=['A', 'B'])
    assert pool.alphabet == ['A', 'B']
    assert pool.num_states == 2**5  # 32
    seq = pool.seq
    assert len(seq) == 5
    assert all(c in 'AB' for c in seq)


def test_kmer_pool_sequence_length():
    """Test that KmerPool generates sequences of correct length."""
    pool = KmerPool(length=20, alphabet='dna')
    assert len(pool.seq) == 20


def test_kmer_pool_deterministic_with_state():
    """Test that KmerPool is deterministic when state is set."""
    pool = KmerPool(length=10)
    pool.set_state(42)
    seq1 = pool.seq
    pool.set_state(42)
    seq2 = pool.seq
    assert seq1 == seq2


def test_kmer_pool_different_states_give_different_sequences():
    """Test that different states produce different sequences in sequential mode."""
    pool = KmerPool(length=2, alphabet=['A', 'B'], mode='sequential')
    pool.set_state(0)
    seq0 = pool.seq
    pool.set_state(1)
    seq1 = pool.seq
    pool.set_state(2)
    seq2 = pool.seq
    pool.set_state(3)
    seq3 = pool.seq
    
    # All should be different for small alphabet
    sequences = [seq0, seq1, seq2, seq3]
    assert len(set(sequences)) == 4
    assert set(sequences) == {'AA', 'AB', 'BA', 'BB'}


def test_kmer_pool_repr():
    """Test KmerPool __repr__."""
    pool = KmerPool(length=15, alphabet='dna')
    assert repr(pool) == "KmerPool(L=15, alphabet='['A', 'C', 'G', 'T']')"


def test_kmer_pool_concatenation():
    """Test KmerPool concatenation with strings."""
    pool = KmerPool(length=5)
    combined = pool + "AAA" + "TTT"
    seq = combined.seq
    assert seq.endswith("AAATTT")
    assert len(seq) == 11  # 5 + 3 + 3


def test_kmer_pool_all_sequences():
    """Test that KmerPool can generate all possible k-mers in sequential mode."""
    pool = KmerPool(length=2, alphabet=['A', 'B'], mode='sequential')
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == 4
    assert set(sequences) == {'AA', 'AB', 'BA', 'BB'}


def test_kmer_pool_correct_count():
    """Test that KmerPool has correct number of states."""
    pool = KmerPool(length=3, alphabet='dna')
    assert pool.num_states == 64  # 4^3


def test_kmer_pool_length_3():
    """Test KmerPool with length 3."""
    pool = KmerPool(length=3, alphabet=['X', 'Y', 'Z'], mode='sequential')
    assert pool.num_states == 27  # 3^3
    
    # Generate all sequences
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    # Check all sequences have correct length
    assert all(len(seq) == 3 for seq in sequences)
    # Check all sequences use only the alphabet
    assert all(all(c in 'XYZ' for c in seq) for seq in sequences)
    # Check all are unique
    assert len(set(sequences)) == 27


def test_kmer_pool_default_alphabet():
    """Test KmerPool with default ACGT alphabet in sequential mode."""
    pool = KmerPool(length=1, mode='sequential')
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == 4
    assert set(sequences) == {'A', 'C', 'G', 'T'}


def test_kmer_pool_concatenation_with_another():
    """Test KmerPool concatenation with another KmerPool."""
    pool1 = KmerPool(length=1, alphabet=['A', 'B'])
    pool2 = KmerPool(length=1, alphabet=['X', 'Y'])
    combined = pool1 + pool2
    
    # Should have 2 * 2 = 4 states
    assert combined.num_states == 4
    
    combined.set_mode('sequential')
    sequences = []
    for state in range(combined.num_states):
        combined.set_state(state)
        sequences.append(combined.seq)
    
    assert len(sequences) == 4
    # Check format: one from {A,B} followed by one from {X,Y}
    assert all(seq[0] in 'AB' and seq[1] in 'XY' for seq in sequences)
