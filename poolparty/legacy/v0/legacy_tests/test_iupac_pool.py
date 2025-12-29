"""Tests for the IUPACPool class."""

import pytest
from poolparty import IUPACPool


def test_iupac_pool_creation():
    """Test IUPACPool creation with valid IUPAC sequences."""
    pool = IUPACPool("ACGT")
    assert pool.iupac_seq == "ACGT"
    assert pool.num_states == 1  # All fixed bases
    assert pool.seq_length == 4
    
    pool2 = IUPACPool("RN")
    assert pool2.iupac_seq == "RN"
    assert pool2.num_states == 8  # R=2 options * N=4 options


def test_iupac_pool_invalid_characters():
    """Test that invalid IUPAC characters raise ValueError."""
    with pytest.raises(ValueError, match="invalid IUPAC character"):
        IUPACPool("ACGTX")
    
    with pytest.raises(ValueError, match="invalid IUPAC character"):
        IUPACPool("123")
    
    with pytest.raises(ValueError, match="invalid IUPAC character"):
        IUPACPool("acgt")  # lowercase not valid


def test_iupac_pool_empty_string():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        IUPACPool("")


def test_iupac_pool_num_states_calculation():
    """Test that num_states is calculated correctly."""
    # Fixed bases only
    pool1 = IUPACPool("ACGT")
    assert pool1.num_states == 1
    
    # Single ambiguous base
    pool2 = IUPACPool("N")
    assert pool2.num_states == 4  # N = A/C/G/T
    
    # R = A/G (2 options)
    pool3 = IUPACPool("R")
    assert pool3.num_states == 2
    
    # RN = 2 * 4 = 8
    pool4 = IUPACPool("RN")
    assert pool4.num_states == 8
    
    # YYY = 2 * 2 * 2 = 8 (Y = C/T)
    pool5 = IUPACPool("YYY")
    assert pool5.num_states == 8
    
    # NNNN = 4^4 = 256
    pool6 = IUPACPool("NNNN")
    assert pool6.num_states == 256


def test_iupac_pool_sequential_mode():
    """Test sequential mode generates all combinations in order."""
    pool = IUPACPool("RY", mode='sequential')  # R=A/G, Y=C/T -> 4 combinations
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == 4
    assert len(set(sequences)) == 4  # All unique
    # Should generate all combinations
    assert set(sequences) == {'AC', 'AT', 'GC', 'GT'}


def test_iupac_pool_sequential_single_position():
    """Test sequential mode with single position."""
    pool = IUPACPool("N", mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == 4
    assert set(sequences) == {'A', 'C', 'G', 'T'}


def test_iupac_pool_deterministic_same_state():
    """Test that same state produces same sequence."""
    pool = IUPACPool("NNNN")
    pool.set_state(123)
    seq1 = pool.seq
    pool.set_state(123)
    seq2 = pool.seq
    assert seq1 == seq2


def test_iupac_pool_seq_length():
    """Test that seq_length property is correct."""
    pool1 = IUPACPool("ACGT")
    assert pool1.seq_length == 4
    
    pool2 = IUPACPool("RYN")
    assert pool2.seq_length == 3
    
    pool3 = IUPACPool("N" * 20)
    assert pool3.seq_length == 20


def test_iupac_pool_repr():
    """Test IUPACPool __repr__."""
    pool = IUPACPool("RYN")
    assert repr(pool) == "IUPACPool(iupac_seq='RYN')"


def test_iupac_pool_concatenation():
    """Test IUPACPool concatenation with strings."""
    pool = IUPACPool("RY")
    combined = pool + "AAA"
    seq = combined.seq
    assert seq.endswith("AAA")
    assert len(seq) == 5  # 2 + 3


def test_iupac_pool_concatenation_with_another():
    """Test IUPACPool concatenation with another IUPACPool."""
    pool1 = IUPACPool("R")  # 2 states
    pool2 = IUPACPool("Y")  # 2 states
    combined = pool1 + pool2
    
    # Should have 2 * 2 = 4 states
    assert combined.num_states == 4
    assert combined.seq_length == 2


def test_iupac_pool_all_iupac_codes():
    """Test that all IUPAC codes work correctly."""
    # Test each IUPAC code
    test_cases = {
        "A": 1,  # A only
        "C": 1,  # C only
        "G": 1,  # G only
        "T": 1,  # T only
        "U": 1,  # U treated as T
        "R": 2,  # A or G
        "Y": 2,  # C or T
        "S": 2,  # G or C
        "W": 2,  # A or T
        "K": 2,  # G or T
        "M": 2,  # A or C
        "B": 3,  # C, G, or T (not A)
        "D": 3,  # A, G, or T (not C)
        "H": 3,  # A, C, or T (not G)
        "V": 3,  # A, C, or G (not T)
        "N": 4,  # Any base
    }
    
    for code, expected_states in test_cases.items():
        pool = IUPACPool(code)
        assert pool.num_states == expected_states, f"Failed for IUPAC code {code}"


def test_iupac_pool_generates_valid_dna():
    """Test that generated sequences only contain valid DNA bases."""
    pool = IUPACPool("RYNSWKMBDHV", mode='sequential')
    for state in range(min(20, pool.num_states)):
        pool.set_state(state)
        seq = pool.seq
        assert all(base in 'ACGT' for base in seq)


def test_iupac_pool_sequential_compatibility():
    """Test sequential compatibility based on max_num_states."""
    # Small number of states - should be sequential compatible
    pool1 = IUPACPool("RY")  # 4 states
    assert pool1.is_sequential_compatible()
    
    # Large number of states - might not be sequential compatible
    pool2 = IUPACPool("N" * 10)  # 4^10 = 1,048,576 states
    assert not pool2.is_sequential_compatible()  # > DEFAULT_MAX_NUM_STATES
    
    # Can make it sequential-compatible by increasing max_num_states
    pool2.set_max_num_states(2_000_000)
    assert pool2.is_sequential_compatible()


def test_iupac_pool_fixed_sequence():
    """Test that fixed IUPAC sequence always returns same sequence."""
    pool = IUPACPool("ACGT", mode='sequential')
    assert pool.num_states == 1
    
    # All states should return the same sequence
    pool.set_state(0)
    seq1 = pool.seq
    pool.set_state(1)
    seq2 = pool.seq
    pool.set_state(100)
    seq3 = pool.seq
    
    assert seq1 == seq2 == seq3 == "ACGT"


def test_iupac_pool_mixed_fixed_and_ambiguous():
    """Test sequences with both fixed and ambiguous positions."""
    pool = IUPACPool("ARCT", mode='sequential')  # R=A/G -> 2 states
    assert pool.num_states == 2
    
    pool.set_state(0)
    seq1 = pool.seq
    pool.set_state(1)
    seq2 = pool.seq
    
    assert len({seq1, seq2}) == 2  # Two different sequences
    assert seq1[0] == 'A'  # First position always A
    assert seq1[2] == 'C'  # Third position always C
    assert seq1[3] == 'T'  # Fourth position always T
    assert seq2[0] == 'A'
    assert seq2[2] == 'C'
    assert seq2[3] == 'T'
    # Second position should be A or G
    assert seq1[1] in 'AG'
    assert seq2[1] in 'AG'
    assert seq1[1] != seq2[1]  # Should be different


def test_iupac_pool_iteration():
    """Test that IUPACPool can be iterated in sequential mode."""
    pool = IUPACPool("RYN", mode='sequential')  # R=2, Y=2, N=4 -> 16 states
    sequences = []
    for state in range(min(10, pool.num_states)):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == 10
    assert all(len(seq) == 3 for seq in sequences)
    assert all(all(base in 'ACGT' for base in seq) for seq in sequences)
