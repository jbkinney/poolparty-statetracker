"""Tests for the InsertionScanPool class."""

import pytest
from poolparty import InsertionScanPool, Pool, KmerPool


# Realistic DNA test sequences (>= 15nt)
TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"  # 24nt
TEST_SEQ_18 = "ACGTACGTACGTACGTAC"  # 18nt
TEST_SEQ_20 = "ACGTACGTACGTACGTACGT"  # 20nt
INSERT_6 = "TTTTTT"  # 6nt insertion


def test_insertion_scan_pool_creation():
    """Test InsertionScanPool creation."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_overwrite_mode_num_states_calculation():
    """Test num_states calculation for overwrite mode."""
    # seq length = 24, insert length = 6, step_size = 1
    # positions = range(0, 24 - 6 + 1, 1) = range(0, 19) = 19 states
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=1)
    assert pool.num_states == 19
    
    # With step_size = 2
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=2)
    assert pool.num_states == 10
    
    # With start = 3
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', start=3, step_size=1)
    assert pool.num_states == 16


def test_insert_mode_num_states_calculation():
    """Test num_states calculation for insert mode."""
    # seq length = 24, step_size = 1
    # positions = range(0, 24 + 1, 1) = range(0, 25) = 25 states (can insert at end)
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='insert', step_size=1)
    assert pool.num_states == 25
    
    # With step_size = 2
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='insert', step_size=2)
    assert pool.num_states == 13
    
    # With start = 3
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='insert', start=3, step_size=1)
    assert pool.num_states == 22


def test_overwrite_mode_basic_operation():
    """Test basic overwrite operation."""
    # Use mode='sequential' for deterministic state-to-position mapping
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_24[6:]  # Overwrites positions 0-5
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[0:1] + INSERT_6 + TEST_SEQ_24[7:]  # Overwrites positions 1-6
    
    pool.set_state(2)
    assert pool.seq == TEST_SEQ_24[0:2] + INSERT_6 + TEST_SEQ_24[8:]  # Overwrites positions 2-7


def test_insert_mode_basic_operation():
    """Test basic insert operation."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='insert', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_18  # Inserts at position 0
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + INSERT_6 + TEST_SEQ_18[1:]  # Inserts after position 0
    
    pool.set_state(2)
    assert pool.seq == TEST_SEQ_18[0:2] + INSERT_6 + TEST_SEQ_18[2:]  # Inserts after position 1


def test_overwrite_mode_with_step_size():
    """Test overwrite mode with larger step_size."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=6, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_24[6:]  # Position 0
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[0:6] + INSERT_6 + TEST_SEQ_24[12:]  # Position 6
    
    pool.set_state(2)
    assert pool.seq == TEST_SEQ_24[0:12] + INSERT_6 + TEST_SEQ_24[18:]  # Position 12


def test_overwrite_mode_with_start():
    """Test overwrite mode with non-zero start."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', start=3, step_size=2, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[0:3] + INSERT_6 + TEST_SEQ_24[9:]  # Position 3
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[0:5] + INSERT_6 + TEST_SEQ_24[11:]  # Position 5


def test_insert_mode_with_start():
    """Test insert mode with non-zero start."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='insert', start=3, step_size=2, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_18[0:3] + INSERT_6 + TEST_SEQ_18[3:]  # Position 3
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:5] + INSERT_6 + TEST_SEQ_18[5:]  # Position 5


def test_mark_changes_true():
    """Test mark_changes=True functionality (swapcase)."""
    insert_lower = "tttttt"
    pool = InsertionScanPool(TEST_SEQ_18, insert_lower, insert_or_overwrite='overwrite', 
                            mark_changes=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "TTTTTT" + TEST_SEQ_18[6:]  # 'tttttt' becomes 'TTTTTT'
    
    # Test with uppercase insertion
    insert_upper = "GGGGGG"
    pool2 = InsertionScanPool(TEST_SEQ_18, insert_upper, insert_or_overwrite='overwrite',
                             mark_changes=True, step_size=1, mode='sequential')
    
    pool2.set_state(0)
    assert pool2.seq == "gggggg" + TEST_SEQ_18[6:]  # 'GGGGGG' becomes 'gggggg'


def test_mark_changes_false():
    """Test mark_changes=False (default) functionality."""
    insert_lower = "tttttt"
    pool = InsertionScanPool(TEST_SEQ_18, insert_lower, insert_or_overwrite='overwrite', 
                            mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "tttttt" + TEST_SEQ_18[6:]  # 'tttttt' stays as 'tttttt'


def test_with_pool_background():
    """Test InsertionScanPool with Pool object as background."""
    background_pool = Pool(seqs=[TEST_SEQ_18])
    pool = InsertionScanPool(background_pool, INSERT_6, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_18[6:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + INSERT_6 + TEST_SEQ_18[7:]


def test_with_pool_insertion():
    """Test InsertionScanPool with Pool object as insertion."""
    insertion_pool = Pool(seqs=[INSERT_6])
    pool = InsertionScanPool(TEST_SEQ_18, insertion_pool, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_18[6:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + INSERT_6 + TEST_SEQ_18[7:]


def test_with_both_pools():
    """Test InsertionScanPool with both Pool objects."""
    background_pool = Pool(seqs=[TEST_SEQ_18])
    insertion_pool = Pool(seqs=[INSERT_6])
    pool = InsertionScanPool(background_pool, insertion_pool, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_18[6:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + INSERT_6 + TEST_SEQ_18[7:]


def test_with_dynamic_pool_parents():
    """Test InsertionScanPool with dynamic Pool parents that update dynamically."""
    insertion_pool = Pool(seqs=["AAA"])
    background = "GGGGGGGGGGGGGGGG"  # 16 G's
    
    pool = InsertionScanPool(background, insertion_pool, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "AAAGGGGGGGGGGGGG"
    
    pool.set_state(1)
    assert pool.seq == "GAAAGGGGGGGGGGGG"
    
    pool.set_state(2)
    assert pool.seq == "GGAAAGGGGGGGGGGG"


def test_sequence_length_overwrite_mode():
    """Test that overwrite mode maintains background length."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    assert pool.seq_length == len(TEST_SEQ_24)
    
    for state in range(pool.num_states):
        pool.set_state(state)
        assert len(pool.seq) == len(TEST_SEQ_24)


def test_sequence_length_insert_mode():
    """Test that insert mode increases length by insertion size."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='insert', step_size=1, mode='sequential')
    
    expected_length = len(TEST_SEQ_18) + len(INSERT_6)
    assert pool.seq_length == expected_length
    
    for state in range(min(pool.num_states, 10)):
        pool.set_state(state)
        assert len(pool.seq) == expected_length


def test_state_wrapping():
    """Test that state wraps with modulo."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    first_seq = pool.seq
    
    pool.set_state(pool.num_states)
    assert pool.seq == first_seq


def test_sequential_iteration():
    """Test InsertionScanPool sequential iteration."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='overwrite', step_size=2, mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    # All sequences should be different
    assert len(sequences) == len(set(sequences))


def test_concatenation():
    """Test InsertionScanPool concatenation."""
    pool1 = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='overwrite', step_size=2, mode='sequential')
    constant = Pool(seqs=["--"])
    
    combined = constant + pool1 + constant
    
    combined.set_state(0)
    assert combined.seq.startswith("--")
    assert combined.seq.endswith("--")


def test_repr_overwrite_mode():
    """Test InsertionScanPool __repr__ for overwrite mode."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='overwrite', step_size=1)
    repr_str = repr(pool)
    assert "InsertionScanPool" in repr_str
    assert "overwrite" in repr_str


def test_repr_insert_mode():
    """Test InsertionScanPool __repr__ for insert mode."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='insert', step_size=2)
    repr_str = repr(pool)
    assert "InsertionScanPool" in repr_str
    assert "insert" in repr_str


def test_deterministic_with_same_state():
    """Test that InsertionScanPool is deterministic with same state."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=1)
    
    pool.set_state(5)
    seq1 = pool.seq
    pool.set_state(5)
    seq2 = pool.seq
    assert seq1 == seq2


def test_insert_at_end():
    """Test insertion at the end of the sequence (insert mode)."""
    pool = InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='insert', step_size=1, mode='sequential')
    
    # Last state should insert at the end
    pool.set_state(pool.num_states - 1)
    assert pool.seq == TEST_SEQ_18 + INSERT_6


def test_edge_case_short_background():
    """Test InsertionScanPool with short background."""
    seq = "ACGTACGTACGTACGT"  # 16 chars
    insert = "TTTT"  # 4 chars
    pool = InsertionScanPool(seq, insert, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    assert pool.num_states > 0
    pool.set_state(0)
    assert pool.seq == "TTTT" + seq[4:]


def test_edge_case_long_insertion():
    """Test InsertionScanPool with insertion almost as long as background."""
    insert = "TTTTTTTTTTTT"  # 12 chars
    pool = InsertionScanPool(TEST_SEQ_18, insert, insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    assert pool.num_states > 0
    pool.set_state(0)
    assert len(pool.seq) == len(TEST_SEQ_18)


def test_single_character_insertion():
    """Test InsertionScanPool with single character insertion."""
    pool = InsertionScanPool(TEST_SEQ_18, "T", insert_or_overwrite='overwrite', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "T" + TEST_SEQ_18[1:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + "T" + TEST_SEQ_18[2:]


def test_different_step_values():
    """Test InsertionScanPool with various step_size values."""
    # Test step_size=1 (sliding window)
    pool1 = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=1)
    assert pool1.num_states == 19
    
    # Test step_size=6 (non-overlapping)
    pool2 = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', step_size=6)
    assert pool2.num_states == 4


def test_mode_parameter():
    """Test that mode parameter is accepted."""
    pool_random = InsertionScanPool(TEST_SEQ_18, INSERT_6, mode='random')
    assert pool_random.mode == 'random'
    
    pool_sequential = InsertionScanPool(TEST_SEQ_18, INSERT_6, mode='sequential')
    assert pool_sequential.mode == 'sequential'


def test_insertion_longer_than_background_error():
    """Test that insertion longer than background raises error in overwrite mode."""
    long_insert = "T" * 25
    
    with pytest.raises(ValueError, match="cannot be longer than"):
        InsertionScanPool(TEST_SEQ_24, long_insert, insert_or_overwrite='overwrite')


def test_insertion_longer_than_background_allowed_in_insert_mode():
    """Test that insertion longer than background is allowed in insert mode."""
    long_insert = "T" * 25
    
    pool = InsertionScanPool(TEST_SEQ_24, long_insert, insert_or_overwrite='insert', mode='sequential')
    assert pool.num_states > 0
    
    pool.set_state(0)
    assert pool.seq == long_insert + TEST_SEQ_24


def test_position_based_interface():
    """Test InsertionScanPool with explicit positions."""
    pool = InsertionScanPool(TEST_SEQ_24, INSERT_6, insert_or_overwrite='overwrite', positions=[0, 6, 12], mode='sequential')
    
    assert pool.num_states == 3
    
    pool.set_state(0)
    assert pool.seq == INSERT_6 + TEST_SEQ_24[6:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[0:6] + INSERT_6 + TEST_SEQ_24[12:]
    
    pool.set_state(2)
    assert pool.seq == TEST_SEQ_24[0:12] + INSERT_6 + TEST_SEQ_24[18:]


def test_position_weights():
    """Test InsertionScanPool with position weights."""
    pool = InsertionScanPool(
        TEST_SEQ_24, 
        INSERT_6, 
        insert_or_overwrite='overwrite',
        positions=[0, 6, 12],
        position_weights=[1.0, 2.0, 1.0],
        mode='random'
    )
    
    assert pool.num_states == 3


def test_validation_errors():
    """Test validation errors."""
    # Cannot mix interfaces
    with pytest.raises(ValueError, match="Cannot specify both"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, step_size=2, positions=[0, 6])
    
    # Weights without positions
    with pytest.raises(ValueError, match="position_weights requires positions"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, position_weights=[1.0, 2.0])
    
    # Weights with sequential mode
    with pytest.raises(ValueError, match="Cannot specify position_weights with mode='sequential'"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, positions=[0, 6], 
                         position_weights=[1.0, 2.0], mode='sequential')
    
    # Empty positions list
    with pytest.raises(ValueError, match="positions must be a non-empty list"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, positions=[])
    
    # Invalid position for overwrite mode
    with pytest.raises(ValueError, match="Position .* is invalid"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='overwrite', positions=[15])
    
    # Invalid insert_or_overwrite value
    with pytest.raises(ValueError, match="insert_or_overwrite must be"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, insert_or_overwrite='invalid')


def test_duplicate_positions_error():
    """Test that duplicate positions raise an error."""
    with pytest.raises(ValueError, match="positions must not contain duplicates"):
        InsertionScanPool(TEST_SEQ_18, INSERT_6, positions=[0, 6, 6, 12])


def test_with_kmer_pool_insertion():
    """Test InsertionScanPool with KmerPool as insertion."""
    kmer_pool = KmerPool(length=6, alphabet='dna')
    pool = InsertionScanPool(TEST_SEQ_18, kmer_pool, insert_or_overwrite='overwrite', step_size=3, mode='sequential')
    
    pool.set_state(0)
    seq = pool.seq
    assert len(seq) == len(TEST_SEQ_18)
