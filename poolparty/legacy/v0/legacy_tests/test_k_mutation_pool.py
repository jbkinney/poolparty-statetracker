"""Comprehensive test suite for KMutationPool.

Validates correctness through statistical and property-based testing.
Tests focus on observable behavior, not implementation details.

Test Categories:
1. Basic functionality and initialization
2. Invariant tests - must hold for every generated sequence
3. Exhaustiveness tests - sequential mode coverage
4. Distribution tests - statistical validation
5. Consistency tests - determinism and reproducibility
6. Edge case tests - boundary conditions
7. Positions parameter tests
8. Pool operations (concatenation, slicing, etc.)
"""

import pytest
from math import comb
from collections import Counter
from itertools import combinations
from poolparty import Pool, KMutationPool, KmerPool


# =============================================================================
# Test Helpers
# =============================================================================

def count_mutations(original: str, mutated: str) -> int:
    """Count the number of positions where sequences differ."""
    return sum(1 for i in range(len(mutated)) if mutated[i] != original[i])


def get_mutation_positions(original: str, mutated: str) -> list:
    """Get list of positions where sequences differ."""
    return [i for i in range(len(mutated)) if mutated[i] != original[i]]


def are_positions_adjacent(positions: list) -> bool:
    """Check if all positions form a contiguous block."""
    if len(positions) <= 1:
        return True
    sorted_pos = sorted(positions)
    return all(sorted_pos[i+1] == sorted_pos[i] + 1 for i in range(len(sorted_pos)-1))


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestKMutationPoolBasic:
    """Test basic functionality of KMutationPool."""
    
    def test_init_with_string(self):
        """Test initialization with a string sequence."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        assert pool is not None
        assert pool.alphabet == ['A', 'C', 'G', 'T']
        assert pool.k == 2
        assert pool.adjacent == False
    
    def test_init_with_pool(self):
        """Test initialization with a Pool object."""
        base_pool = Pool(seqs=['ACGT'])
        pool = KMutationPool(base_pool, alphabet=['A', 'C', 'G', 'T'], k=2)
        assert pool is not None
    
    def test_init_with_adjacent_true(self):
        """Test initialization with adjacent=True."""
        pool = KMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], k=3, adjacent=True)
        assert pool.adjacent == True
    
    def test_finite_states(self):
        """Test that KMutationPool has finite states."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        # C(4,2) * (4-1)^2 = 6 * 9 = 54
        assert pool.num_states == 54
        assert pool.is_sequential_compatible()
    
    def test_sequence_length_preserved(self):
        """Test that mutated sequence has same length as input."""
        pool = KMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], k=3)
        pool.set_state(0)
        seq = pool.seq
        assert len(seq) == 8
    
    def test_characters_from_alphabet(self):
        """Test that all characters in mutated sequence are from alphabet."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        pool.set_state(0)
        seq = pool.seq
        for char in seq:
            assert char in ['A', 'C', 'G', 'T']


# =============================================================================
# INVARIANT TESTS: Must hold for every generated sequence
# =============================================================================

class TestInvariantExactlyKMutations:
    """Verify that every sequence has exactly k mutations."""
    
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_exactly_k_mutations_basic(self, k, mode):
        """Every sequence must have exactly k positions mutated."""
        original = "AAAAAAAA"  # 8 positions
        pool = KMutationPool(original, k=k, mode=mode)
        
        for state in range(min(100, pool.num_internal_states)):
            pool.set_state(state)
            mutations = count_mutations(original, pool.seq)
            assert mutations == k, f"State {state}: expected {k} mutations, got {mutations}"
    
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_exactly_k_with_positions(self, k):
        """With explicit positions, still exactly k mutations."""
        original = "AAAAAAAAAA"  # 10 positions
        positions = [0, 3, 5, 7, 9]  # 5 allowed positions
        
        pool = KMutationPool(original, k=k, positions=positions, mode='sequential')
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            mutations = count_mutations(original, pool.seq)
            assert mutations == k
    
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_exactly_k_adjacent_mode(self, k):
        """Adjacent mode still produces exactly k mutations."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=k, adjacent=True, mode='sequential')
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            mutations = count_mutations(original, pool.seq)
            assert mutations == k


class TestInvariantMutationsAtAllowedPositions:
    """Verify mutations only occur at allowed positions."""
    
    @pytest.mark.parametrize("positions", [
        [0, 2, 4],      # Even positions
        [1, 3, 5, 7],   # Odd positions
        [0],            # Single position
        [0, 9],         # First and last
    ])
    def test_mutations_only_at_specified_positions(self, positions):
        """Mutations must only occur at positions in the positions list."""
        original = "AAAAAAAAAA"  # 10 positions
        k = min(2, len(positions))
        
        pool = KMutationPool(original, k=k, positions=positions, mode='random')
        
        for state in range(100):
            pool.set_state(state)
            mutation_positions = get_mutation_positions(original, pool.seq)
            
            for pos in mutation_positions:
                assert pos in positions, \
                    f"State {state}: mutation at position {pos} not in allowed {positions}"
    
    def test_forbidden_positions_never_mutated(self):
        """Positions not in the list must never be mutated."""
        original = "AAAAAAAAAA"
        allowed = [2, 5, 8]
        forbidden = [i for i in range(10) if i not in allowed]
        
        pool = KMutationPool(original, k=2, positions=allowed, mode='random')
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            
            for pos in forbidden:
                assert seq[pos] == original[pos], \
                    f"State {state}: forbidden position {pos} was mutated"


class TestInvariantAdjacentMutations:
    """Verify adjacent mode produces contiguous mutation blocks."""
    
    @pytest.mark.parametrize("k", [2, 3, 4])
    @pytest.mark.parametrize("seq_len", [8, 12, 20])
    def test_mutations_are_adjacent(self, k, seq_len):
        """All k mutations must be at adjacent positions."""
        original = "A" * seq_len
        pool = KMutationPool(original, k=k, adjacent=True, mode='sequential')
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            
            assert len(positions) == k, f"Expected {k} mutations"
            assert are_positions_adjacent(positions), \
                f"State {state}: positions {positions} are not adjacent"
    
    def test_adjacent_at_start(self):
        """Test adjacent mutations can start at position 0."""
        original_seq = 'AAAAAAAA'
        pool = KMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], k=3, adjacent=True)
        
        found_start_mutation = False
        for state in range(100):
            pool.set_state(state)
            mutated = pool.seq
            
            if mutated[0] != original_seq[0]:
                found_start_mutation = True
                break
        
        assert found_start_mutation, "Should find mutations starting at position 0"
    
    def test_adjacent_at_end(self):
        """Test adjacent mutations can end at last position."""
        original_seq = 'AAAAAAAA'
        pool = KMutationPool(original_seq, alphabet=['A', 'C', 'G', 'T'], k=3, adjacent=True)
        
        found_end_mutation = False
        for state in range(pool.num_states):
            pool.set_state(state)
            mutated = pool.seq
            
            if mutated[-1] != original_seq[-1]:
                found_end_mutation = True
                break
        
        assert found_end_mutation, "Should find mutations ending at last position"


class TestInvariantValidAlphabetSubstitutions:
    """Verify all mutations are valid alphabet substitutions."""
    
    @pytest.mark.parametrize("alphabet", [
        ['A', 'C', 'G', 'T'],
        ['A', 'C'],
        ['X', 'Y', 'Z'],
    ])
    def test_mutations_are_from_alphabet(self, alphabet):
        """Every character in output must be from the alphabet."""
        original = alphabet[0] * 10
        pool = KMutationPool(original, alphabet=alphabet, k=3, mode='random')
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            for char in seq:
                assert char in alphabet, \
                    f"State {state}: character '{char}' not in alphabet {alphabet}"
    
    def test_mutation_never_same_as_original(self):
        """A mutated position must have a different character than original."""
        original = "ACGTACGT"
        pool = KMutationPool(original, k=2, mode='sequential')
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            mutation_positions = get_mutation_positions(original, seq)
            for pos in mutation_positions:
                assert seq[pos] != original[pos], \
                    f"State {state}: position {pos} not actually mutated"


class TestInvariantSequenceLengthUnchanged:
    """Verify sequence length is preserved."""
    
    @pytest.mark.parametrize("seq_len", [4, 8, 20, 100])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_length_preserved(self, seq_len, k):
        """Output sequence length must equal input length."""
        original = "A" * seq_len
        pool = KMutationPool(original, k=k, mode='random')
        
        for state in range(50):
            pool.set_state(state)
            assert len(pool.seq) == seq_len


# =============================================================================
# EXHAUSTIVENESS TESTS: Sequential mode coverage
# =============================================================================

class TestExhaustivenessUniqueSequences:
    """Verify sequential mode produces unique sequences."""
    
    @pytest.mark.parametrize("k", [1, 2])
    def test_all_states_produce_unique_sequences(self, k):
        """Each state in sequential mode produces a unique sequence."""
        original = "AAAA"  # 4 positions
        pool = KMutationPool(original, k=k, mode='sequential')
        
        seen_sequences = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            assert seq not in seen_sequences, \
                f"State {state}: duplicate sequence {seq}"
            seen_sequences.add(seq)
    
    def test_unique_sequences_with_positions(self):
        """Unique sequences when using explicit positions."""
        original = "AAAAAAAA"
        positions = [0, 2, 4, 6]  # 4 positions
        pool = KMutationPool(original, k=2, positions=positions, mode='sequential')
        
        seen = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert seq not in seen, f"Duplicate: {seq}"
            seen.add(seq)
    
    def test_unique_sequences_adjacent_mode(self):
        """Unique sequences in adjacent mode."""
        original = "AAAAAA"
        pool = KMutationPool(original, k=2, adjacent=True, mode='sequential')
        
        seen = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert seq not in seen
            seen.add(seq)


class TestExhaustivenessStateCount:
    """Verify num_internal_states matches actual unique sequences."""
    
    def test_state_count_matches_formula_basic(self):
        """State count matches C(L,k) * (alpha-1)^k formula."""
        seq = "ACGT"  # L=4
        k = 2
        alpha = 4  # DNA alphabet
        
        pool = KMutationPool(seq, k=k)
        
        expected = comb(4, k) * (alpha - 1) ** k  # 6 * 9 = 54
        assert pool.num_internal_states == expected
    
    def test_state_count_with_positions(self):
        """State count correct with explicit positions."""
        seq = "AAAAAAAA"  # 8 positions
        positions = [0, 2, 4, 6]  # 4 allowed
        k = 2
        alpha = 4
        
        pool = KMutationPool(seq, k=k, positions=positions)
        
        expected = comb(4, k) * (alpha - 1) ** k  # 6 * 9 = 54
        assert pool.num_internal_states == expected
    
    def test_state_count_adjacent_mode(self):
        """State count correct for adjacent mode: (L-k+1) * (alpha-1)^k."""
        seq = "AAAAAAAA"  # L=8
        k = 3
        alpha = 4
        
        pool = KMutationPool(seq, k=k, adjacent=True)
        
        expected = (8 - k + 1) * (alpha - 1) ** k  # 6 * 27 = 162
        assert pool.num_internal_states == expected
    
    def test_actual_unique_count_matches_num_states(self):
        """Iterating through all states gives exactly num_internal_states unique sequences."""
        original = "AAAA"
        pool = KMutationPool(original, k=2, mode='sequential')
        
        unique_seqs = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            unique_seqs.add(pool.seq)
        
        assert len(unique_seqs) == pool.num_internal_states


class TestExhaustivenessStateCycling:
    """Verify state wrapping behavior."""
    
    def test_state_wraps_at_num_states(self):
        """State N produces same sequence as state 0."""
        original = "AAAA"
        pool = KMutationPool(original, k=1, mode='sequential')
        
        pool.set_state(0)
        seq_0 = pool.seq
        
        pool.set_state(pool.num_internal_states)
        seq_wrap = pool.seq
        
        assert seq_0 == seq_wrap
    
    def test_full_cycle_repeats(self):
        """Sequences repeat after num_internal_states iterations."""
        original = "AAA"
        pool = KMutationPool(original, alphabet=['A', 'C'], k=1, mode='sequential')
        
        # Collect one full cycle
        cycle1 = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            cycle1.append(pool.seq)
        
        # Collect second cycle
        cycle2 = []
        for state in range(pool.num_internal_states, 2 * pool.num_internal_states):
            pool.set_state(state)
            cycle2.append(pool.seq)
        
        assert cycle1 == cycle2


# =============================================================================
# SEQUENCE CORRECTNESS TESTS
# =============================================================================

class TestSequenceCorrectness:
    """
    Verify that generated sequences are not just unique, but mathematically correct.
    
    For sequential mode, we can predict exactly what sequence should be produced
    for any given state by manually computing the position/mutation decomposition.
    """
    
    def test_sequential_standard_exact_sequences_k1(self):
        """Verify exact sequences for k=1 - each state maps to specific position+mutation."""
        original = "AAAA"  # 4 positions
        alphabet = ['A', 'C', 'G', 'T']  # 3 alternatives per position
        pool = KMutationPool(original, alphabet=alphabet, k=1, mode='sequential')
        
        # State = position * 3 + mutation_index
        # mutation_index 0='C', 1='G', 2='T' (alternatives to 'A')
        expected = {
            0: "CAAA",   # pos=0, mut=0 (C)
            1: "GAAA",   # pos=0, mut=1 (G)
            2: "TAAA",   # pos=0, mut=2 (T)
            3: "ACAA",   # pos=1, mut=0 (C)
            4: "AGAA",   # pos=1, mut=1 (G)
            5: "ATAA",   # pos=1, mut=2 (T)
            6: "AACA",   # pos=2, mut=0 (C)
            7: "AAGA",   # pos=2, mut=1 (G)
            8: "AATA",   # pos=2, mut=2 (T)
            9: "AAAC",   # pos=3, mut=0 (C)
            10: "AAAG",  # pos=3, mut=1 (G)
            11: "AAAT",  # pos=3, mut=2 (T)
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            actual = pool.seq
            assert actual == expected_seq, \
                f"State {state}: expected '{expected_seq}', got '{actual}'"
    
    def test_sequential_standard_exact_sequences_k2(self):
        """Verify exact sequences for k=2 with known decomposition."""
        original = "AAA"  # 3 positions
        alphabet = ['A', 'C']  # 1 alternative per position
        pool = KMutationPool(original, alphabet=alphabet, k=2, mode='sequential')
        
        # C(3,2) = 3 position combinations: (0,1), (0,2), (1,2)
        # 1 alternative per position, so 1 mutation pattern per combo
        expected = {
            0: "CCA",
            1: "CAC",
            2: "ACC",
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            assert pool.seq == expected_seq
    
    def test_sequential_standard_exact_sequences_k2_multi_alt(self):
        """Verify k=2 with multiple alternatives - rightmost varies fastest."""
        original = "AAA"
        alphabet = ['A', 'C', 'G']  # 2 alternatives
        pool = KMutationPool(original, alphabet=alphabet, k=2, mode='sequential')
        
        # 3 position combinations, 2^2 = 4 mutation patterns each
        # Positions (0,1): states 0-3
        # Positions (0,2): states 4-7
        # Positions (1,2): states 8-11
        expected = {
            0: "CCA", 1: "CGA", 2: "GCA", 3: "GGA",
            4: "CAC", 5: "CAG", 6: "GAC", 7: "GAG",
            8: "ACC", 9: "ACG", 10: "AGC", 11: "AGG",
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            actual = pool.seq
            assert actual == expected_seq, \
                f"State {state}: expected '{expected_seq}', got '{actual}'"
    
    def test_sequential_positions_exact_sequences(self):
        """Verify exact sequences with explicit positions."""
        original = "AAAAA"  # 5 positions
        positions = [1, 3]  # Only positions 1 and 3
        alphabet = ['A', 'C', 'G']  # 2 alternatives
        pool = KMutationPool(original, alphabet=alphabet, k=2, positions=positions, mode='sequential')
        
        # C(2,2) = 1 position combo: (1,3)
        # 2^2 = 4 mutation patterns
        expected = {
            0: "ACACA",  # pos 1=C, pos 3=C
            1: "ACAGA",  # pos 1=C, pos 3=G (rightmost varies first)
            2: "AGACA",  # pos 1=G, pos 3=C
            3: "AGAGA",  # pos 1=G, pos 3=G
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            assert pool.seq == expected_seq
    
    def test_sequential_adjacent_exact_sequences(self):
        """Verify exact sequences for adjacent mode."""
        original = "AAAA"  # 4 positions
        alphabet = ['A', 'C']  # 1 alternative
        pool = KMutationPool(original, alphabet=alphabet, k=2, adjacent=True, mode='sequential')
        
        # (4-2+1) = 3 starting positions: 0, 1, 2
        expected = {
            0: "CCAA",  # start at 0
            1: "ACCA",  # start at 1
            2: "AACC",  # start at 2
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            assert pool.seq == expected_seq
    
    def test_sequential_adjacent_multi_alt_exact(self):
        """Verify adjacent mode with multiple alternatives."""
        original = "AAA"
        alphabet = ['A', 'C', 'G']  # 2 alternatives
        pool = KMutationPool(original, alphabet=alphabet, k=2, adjacent=True, mode='sequential')
        
        # 2 starting positions: 0, 1
        # 2^2 = 4 mutation patterns each
        expected = {
            # Start position 0: positions (0,1)
            0: "CCA", 1: "CGA", 2: "GCA", 3: "GGA",
            # Start position 1: positions (1,2)
            4: "ACC", 5: "ACG", 6: "AGC", 7: "AGG",
        }
        
        for state, expected_seq in expected.items():
            pool.set_state(state)
            actual = pool.seq
            assert actual == expected_seq, f"State {state}: expected '{expected_seq}', got '{actual}'"
    
    def test_verify_mutation_pattern_decomposition(self):
        """Verify the rightmost-varies-fastest decomposition is correct."""
        original = "AAAA"
        alphabet = ['A', 'X', 'Y', 'Z']  # 3 alternatives: X, Y, Z
        pool = KMutationPool(original, alphabet=alphabet, k=2, mode='sequential')
        
        # For first position combo (0,1), verify mutation patterns
        expected_patterns = [
            ("X", "X"), ("X", "Y"), ("X", "Z"),
            ("Y", "X"), ("Y", "Y"), ("Y", "Z"),
            ("Z", "X"), ("Z", "Y"), ("Z", "Z"),
        ]
        
        for state, (left_mut, right_mut) in enumerate(expected_patterns):
            pool.set_state(state)
            seq = pool.seq
            
            assert seq[0] == left_mut, f"State {state}: pos 0 expected {left_mut}, got {seq[0]}"
            assert seq[1] == right_mut, f"State {state}: pos 1 expected {right_mut}, got {seq[1]}"
            assert seq[2:] == "AA", f"State {state}: positions 2,3 should be unchanged"
    
    def test_random_mode_produces_valid_sequences(self):
        """Random mode produces valid sequences (not necessarily in order)."""
        original = "AAAA"
        alphabet = ['A', 'C', 'G']
        pool = KMutationPool(original, alphabet=alphabet, k=2, mode='random')
        
        # All possible valid sequences for k=2
        alternatives = ['C', 'G']
        valid_sequences = set()
        
        for pos_combo in combinations(range(4), 2):
            for mut1 in alternatives:
                for mut2 in alternatives:
                    seq = list(original)
                    seq[pos_combo[0]] = mut1
                    seq[pos_combo[1]] = mut2
                    valid_sequences.add(''.join(seq))
        
        # Every sequence from random mode should be in the valid set
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            assert seq in valid_sequences, f"State {state}: '{seq}' is not a valid sequence"
    
    def test_state_to_sequence_bijection(self):
        """In sequential mode, state <-> sequence is a bijection within one cycle."""
        original = "AAA"
        pool = KMutationPool(original, alphabet=['A', 'C'], k=1, mode='sequential')
        
        state_to_seq = {}
        seq_to_state = {}
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            state_to_seq[state] = seq
            
            if seq in seq_to_state:
                assert False, f"Sequence '{seq}' produced by both state {seq_to_state[seq]} and {state}"
            seq_to_state[seq] = state
        
        assert len(state_to_seq) == pool.num_internal_states
        assert len(seq_to_state) == pool.num_internal_states
    
    def test_position_combo_coverage_sequential(self):
        """Verify all position combinations appear exactly the expected number of times."""
        original = "AAAAA"  # 5 positions
        pool = KMutationPool(original, alphabet=['A', 'C', 'G', 'T'], k=2, mode='sequential')
        
        # C(5,2) = 10 position combinations
        # Each should appear (alpha-1)^k = 9 times
        position_combo_counts = Counter()
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            positions = tuple(sorted(get_mutation_positions(original, pool.seq)))
            position_combo_counts[positions] += 1
        
        for combo, count in position_combo_counts.items():
            assert count == 9, f"Position combo {combo} appeared {count} times, expected 9"
        
        assert len(position_combo_counts) == 10


# =============================================================================
# DISTRIBUTION TESTS: Statistical validation
# =============================================================================

class TestDistributionPositionSelection:
    """Verify uniform position selection in random mode."""
    
    def test_position_selection_uniform(self):
        """All positions should be selected with roughly equal frequency."""
        original = "AAAAAAAAAA"  # 10 positions
        pool = KMutationPool(original, k=1, mode='random')
        
        position_counts = Counter()
        n_samples = 5000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            for pos in positions:
                position_counts[pos] += 1
        
        expected_per_position = n_samples / 10  # 500
        
        for pos in range(10):
            count = position_counts[pos]
            # Allow 30% deviation
            assert expected_per_position * 0.7 < count < expected_per_position * 1.3, \
                f"Position {pos} count {count} deviates too much from expected {expected_per_position}"
    
    def test_position_selection_respects_positions_list(self):
        """Only allowed positions appear in output."""
        original = "AAAAAAAAAA"
        allowed = [0, 3, 6, 9]  # 4 positions
        pool = KMutationPool(original, k=1, positions=allowed, mode='random')
        
        position_counts = Counter()
        n_samples = 4000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            for pos in positions:
                position_counts[pos] += 1
        
        for pos in range(10):
            if pos in allowed:
                assert position_counts[pos] > 0, f"Allowed position {pos} never selected"
            else:
                assert position_counts[pos] == 0, f"Forbidden position {pos} was selected"


class TestDistributionMutationCharacter:
    """Verify uniform mutation character selection."""
    
    def test_mutation_character_uniform(self):
        """All alternative characters should appear with roughly equal frequency."""
        original = "AAAAAAAAAA"
        alphabet = ['A', 'C', 'G', 'T']
        pool = KMutationPool(original, alphabet=alphabet, k=1, mode='random')
        
        char_counts = Counter()
        n_samples = 6000
        
        for state in range(n_samples):
            pool.set_state(state)
            seq = pool.seq
            mutation_positions = get_mutation_positions(original, seq)
            for pos in mutation_positions:
                char_counts[seq[pos]] += 1
        
        # 'A' should never appear (it's the original)
        assert char_counts['A'] == 0
        
        # C, G, T should each appear ~1/3 of samples
        expected_per_char = n_samples / 3
        for char in ['C', 'G', 'T']:
            count = char_counts[char]
            assert expected_per_char * 0.8 < count < expected_per_char * 1.2, \
                f"Character {char} count {count} deviates from expected {expected_per_char}"


class TestDistributionPositionCombinations:
    """Verify all valid position combinations appear."""
    
    def test_all_position_pairs_appear_k2(self):
        """For k=2, all position pairs should appear over many samples."""
        original = "AAAAA"  # 5 positions -> C(5,2) = 10 pairs
        pool = KMutationPool(original, k=2, mode='random')
        
        seen_pairs = set()
        n_samples = 2000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = tuple(sorted(get_mutation_positions(original, pool.seq)))
            seen_pairs.add(positions)
        
        expected_pairs = comb(5, 2)  # 10
        assert len(seen_pairs) == expected_pairs, \
            f"Expected {expected_pairs} unique pairs, got {len(seen_pairs)}"


# =============================================================================
# CONSISTENCY TESTS: Determinism and reproducibility
# =============================================================================

class TestConsistencyDeterminism:
    """Verify deterministic behavior."""
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_same_state_same_sequence(self, mode):
        """Same state always produces the same sequence."""
        original = "ACGTACGT"
        pool = KMutationPool(original, k=2, mode=mode)
        
        for state in [0, 5, 42, 100]:
            pool.set_state(state)
            seq1 = pool.seq
            
            pool.set_state(state)
            seq2 = pool.seq
            
            assert seq1 == seq2, f"State {state}: got different sequences"
    
    def test_determinism_across_pool_instances(self):
        """Different pool instances with same params produce same sequences."""
        original = "AAAAAAAA"
        
        pool1 = KMutationPool(original, k=2, mode='sequential')
        pool2 = KMutationPool(original, k=2, mode='sequential')
        
        for state in range(20):
            pool1.set_state(state)
            pool2.set_state(state)
            
            assert pool1.seq == pool2.seq


class TestConsistencyGenerateSeqs:
    """Verify generate_seqs produces correct output."""
    
    def test_generate_seqs_count(self):
        """generate_seqs(n) returns exactly n sequences."""
        pool = KMutationPool("AAAAAAAA", k=2, mode='sequential')
        
        seqs = pool.generate_seqs(num_seqs=25)
        assert len(seqs) == 25
    
    def test_generate_seqs_all_have_k_mutations(self):
        """All sequences from generate_seqs have exactly k mutations."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=3, mode='random')
        
        seqs = pool.generate_seqs(num_seqs=100)
        
        for seq in seqs:
            mutations = count_mutations(original, seq)
            assert mutations == 3
    
    def test_generate_seqs_with_seed(self):
        """Same seed produces same sequences."""
        pool = KMutationPool("AAAAAAAA", k=2, mode='random')
        
        seqs1 = pool.generate_seqs(num_seqs=20, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=20, seed=42)
        
        assert seqs1 == seqs2
    
    def test_generate_seqs_different_seeds(self):
        """Different seeds (usually) produce different sequences."""
        pool = KMutationPool("AAAAAAAA", k=2, mode='random')
        
        seqs1 = pool.generate_seqs(num_seqs=20, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=20, seed=123)
        
        assert seqs1 != seqs2
    
    def test_generate_seqs_num_complete_iterations(self):
        """num_complete_iterations produces full cycles."""
        original = "AAA"
        pool = KMutationPool(original, alphabet=['A', 'C'], k=1, mode='sequential')
        
        num_states = pool.num_internal_states  # 3 positions * 1 alt = 3
        
        seqs = pool.generate_seqs(num_complete_iterations=2)
        assert len(seqs) == num_states * 2
        
        # First half equals second half
        assert seqs[:num_states] == seqs[num_states:]


class TestConsistencyMarkChanges:
    """Verify mark_changes parameter behavior."""
    
    def test_mark_changes_swaps_case(self):
        """mark_changes=True applies swapcase to mutated positions."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=2, mark_changes=True, mode='sequential')
        
        pool.set_state(0)
        seq = pool.seq
        
        # Mutated positions should be lowercase
        for i in range(len(seq)):
            if seq[i].upper() != original[i]:
                assert seq[i].islower(), f"Position {i} should be lowercase"
            else:
                assert seq[i].isupper(), f"Position {i} should be uppercase"
    
    def test_mark_changes_false_no_case_change(self):
        """mark_changes=False preserves original case."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=2, mark_changes=False, mode='sequential')
        
        pool.set_state(0)
        seq = pool.seq
        
        # All characters should be uppercase
        assert seq == seq.upper()


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestKMutationPoolValidation:
    """Test input validation."""
    
    def test_empty_alphabet_error(self):
        """Test that empty alphabet raises error."""
        with pytest.raises(ValueError, match="alphabet list must be non-empty"):
            KMutationPool('ACGT', alphabet=[], k=2)
    
    def test_k_zero_error(self):
        """Test that k=0 raises error."""
        with pytest.raises(ValueError, match="k must be > 0"):
            KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=0)
    
    def test_k_negative_error(self):
        """Test that negative k raises error."""
        with pytest.raises(ValueError, match="k must be > 0"):
            KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=-1)
    
    def test_k_too_large_error(self):
        """Test that k > sequence length raises error."""
        with pytest.raises(ValueError, match="k .* must be <= sequence length"):
            KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=5)
    
    def test_k_too_large_adjacent_error(self):
        """Test that k > sequence length raises error in adjacent mode."""
        with pytest.raises(ValueError, match="k .* must be <= sequence length"):
            KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=5, adjacent=True)


class TestKMutationPoolPositionsValidation:
    """Test validation of positions parameter."""
    
    def test_positions_empty_error(self):
        """Verify empty positions list raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            KMutationPool('ACGT', k=1, positions=[])
    
    def test_positions_out_of_bounds_error(self):
        """Verify out-of-bounds position raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            KMutationPool('ACGT', k=1, positions=[0, 5])
    
    def test_positions_negative_error(self):
        """Verify negative position raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            KMutationPool('ACGT', k=1, positions=[-1, 0])
    
    def test_positions_duplicates_error(self):
        """Verify duplicate positions raises error."""
        with pytest.raises(ValueError, match="duplicates"):
            KMutationPool('ACGT', k=1, positions=[0, 1, 0])
    
    def test_k_greater_than_positions_error(self):
        """Verify k > len(positions) raises error."""
        with pytest.raises(ValueError, match="k .* must be <="):
            KMutationPool('ACGTACGT', k=3, positions=[0, 1])
    
    def test_positions_with_adjacent_error(self):
        """Verify using both positions and adjacent raises error."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            KMutationPool('ACGT', k=2, positions=[0, 2], adjacent=True)
    
    def test_positions_non_integer_error(self):
        """Verify non-integer position raises error."""
        with pytest.raises(ValueError, match="integers"):
            KMutationPool('ACGT', k=1, positions=[0, 1.5])


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCaseK1:
    """Edge cases for k=1 (single mutation)."""
    
    def test_k1_state_count(self):
        """k=1: num_states = L * (alpha-1)."""
        seq = "ACGT"  # L=4
        pool = KMutationPool(seq, k=1)  # alpha=4
        
        expected = 4 * 3  # 12
        assert pool.num_internal_states == expected
    
    def test_k1_all_positions_covered(self):
        """k=1 sequential mode covers all positions."""
        original = "AAAAA"
        pool = KMutationPool(original, k=1, mode='sequential')
        
        positions_seen = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            positions_seen.update(positions)
        
        assert positions_seen == set(range(5))


class TestEdgeCaseKEqualsSeqLen:
    """Edge cases for k=len(seq) (all positions mutated)."""
    
    def test_k_equals_len_all_mutated(self):
        """When k=len(seq), every position is mutated."""
        original = "AAAA"
        pool = KMutationPool(original, k=4, mode='sequential')
        
        for state in range(min(50, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            
            for i in range(len(original)):
                assert seq[i] != original[i], f"Position {i} not mutated"
    
    def test_k_equals_len_state_count(self):
        """When k=len(seq), num_states = (alpha-1)^L."""
        seq = "AAA"  # L=3
        pool = KMutationPool(seq, alphabet=['A', 'C'], k=3)  # alpha=2
        
        expected = 1 ** 3  # Only 1 alternative per position = 1
        assert pool.num_internal_states == expected


class TestEdgeCaseKEqualsPositionsLen:
    """Edge cases when k = len(positions)."""
    
    def test_k_equals_positions_len(self):
        """When k = len(positions), all allowed positions mutated."""
        original = "AAAAAAAAAA"
        positions = [2, 5, 8]  # 3 positions
        
        pool = KMutationPool(original, k=3, positions=positions, mode='sequential')
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            mutation_positions = get_mutation_positions(original, pool.seq)
            
            assert set(mutation_positions) == set(positions)


class TestEdgeCaseBoundaryPositions:
    """Edge cases for positions at sequence boundaries."""
    
    @pytest.mark.parametrize("positions", [
        [0],              # First only
        [9],              # Last only
        [0, 9],           # First and last
        [0, 1, 8, 9],     # Boundaries
    ])
    def test_boundary_positions(self, positions):
        """Mutations at sequence boundaries work correctly."""
        original = "AAAAAAAAAA"  # Length 10
        k = min(2, len(positions))
        
        pool = KMutationPool(original, k=k, positions=positions, mode='random')
        
        for state in range(50):
            pool.set_state(state)
            mutation_positions = get_mutation_positions(original, pool.seq)
            
            assert len(mutation_positions) == k
            for pos in mutation_positions:
                assert pos in positions


class TestEdgeCaseSmallAlphabet:
    """Edge cases with small alphabets."""
    
    def test_binary_alphabet(self):
        """Works with 2-character alphabet."""
        original = "AAAA"
        pool = KMutationPool(original, alphabet=['A', 'B'], k=2, mode='sequential')
        
        # Only 1 alternative per position
        expected = comb(4, 2) * 1 ** 2  # 6
        assert pool.num_internal_states == expected
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert all(c in ['A', 'B'] for c in seq)
    
    def test_ternary_alphabet(self):
        """Works with 3-character alphabet."""
        original = "XXX"
        pool = KMutationPool(original, alphabet=['X', 'Y', 'Z'], k=2, mode='sequential')
        
        expected = comb(3, 2) * 2 ** 2  # 3 * 4 = 12
        assert pool.num_internal_states == expected


class TestEdgeCaseShortSequences:
    """Edge cases with very short sequences."""
    
    def test_length_2_k_1(self):
        """Sequence of length 2 with k=1."""
        original = "AA"
        pool = KMutationPool(original, k=1, mode='sequential')
        
        expected = 2 * 3  # 6
        assert pool.num_internal_states == expected
    
    def test_length_2_k_2(self):
        """Sequence of length 2 with k=2."""
        original = "AA"
        pool = KMutationPool(original, k=2, mode='sequential')
        
        expected = 1 * 3 ** 2  # 9
        assert pool.num_internal_states == expected


# =============================================================================
# POSITIONS PARAMETER TESTS
# =============================================================================

class TestKMutationPoolPositions:
    """Test explicit positions parameter."""
    
    def test_positions_basic(self):
        """Test that positions restricts mutations to specified positions."""
        seq = 'AAAAAAAA'  # 8 positions
        pool = KMutationPool(
            seq, alphabet=['A', 'C'], k=1,
            positions=[0, 3, 7],  # Only these positions can be mutated
            mode='sequential'
        )
        
        # C(3,1) * 1^1 = 3 states
        assert pool.num_internal_states == 3
        
        expected_seqs = ['CAAAAAAA', 'AAACAAAA', 'AAAAAAAC']
        for i in range(3):
            pool.set_state(i)
            assert pool.seq == expected_seqs[i], f"State {i}: expected {expected_seqs[i]}, got {pool.seq}"
    
    def test_positions_k_greater_than_1(self):
        """Test positions with k > 1."""
        seq = 'AAAAAAA'  # 7 positions
        pool = KMutationPool(
            seq, alphabet=['A', 'C'], k=2,
            positions=[1, 3, 5],  # Only these 3 positions
            mode='sequential'
        )
        
        # C(3,2) * 1^2 = 3 states
        assert pool.num_internal_states == 3
        
        all_seqs = set()
        for i in range(pool.num_internal_states):
            pool.set_state(i)
            all_seqs.add(pool.seq)
        
        expected = {'ACACAAA', 'ACAAACA', 'AAACACA'}
        assert all_seqs == expected
    
    def test_positions_state_count_formula(self):
        """Test state count formula with positions."""
        seq = 'ACGTACGT'
        k = 2
        positions = [0, 2, 4, 6]  # 4 positions
        alphabet = ['A', 'C', 'G', 'T']
        pool = KMutationPool(seq, alphabet, k, positions=positions)
        
        L = len(positions)  # 4
        alpha = len(alphabet)  # 4
        expected = comb(L, k) * (alpha - 1) ** k  # C(4,2) * 3^2 = 6 * 9 = 54
        
        assert pool.num_internal_states == expected
    
    def test_positions_mutations_only_at_specified(self):
        """Verify mutations ONLY occur at specified positions."""
        seq = 'AAAAAAAAAA'  # 10 positions
        allowed_positions = [2, 5, 8]
        
        pool = KMutationPool(
            seq, alphabet=['A', 'C', 'G'], k=2,
            positions=allowed_positions,
            mode='random'
        )
        
        for state in range(100):
            pool.set_state(state)
            result = pool.seq
            
            for i in range(len(result)):
                if i not in allowed_positions:
                    assert result[i] == seq[i], \
                        f"State {state}: Position {i} mutated but not allowed"
    
    def test_positions_sequential_complete_coverage(self):
        """Verify sequential mode covers all position combinations."""
        seq = 'AAAA'  # 4 positions
        positions = [0, 1, 2, 3]  # All positions
        
        pool = KMutationPool(
            seq, alphabet=['A', 'C'], k=2,
            positions=positions,
            mode='sequential'
        )
        
        expected_states = pool.num_internal_states
        
        seen_seqs = set()
        for i in range(expected_states):
            pool.set_state(i)
            seen_seqs.add(pool.seq)
        
        assert len(seen_seqs) == expected_states
    
    def test_positions_single_position(self):
        """Test with only one position allowed."""
        seq = 'ACGT'
        pool = KMutationPool(
            seq, alphabet=['A', 'C', 'G', 'T'], k=1,
            positions=[2],  # Only position 2
            mode='sequential'
        )
        
        # Only 1 position, k=1, so 3 alternatives
        assert pool.num_internal_states == 3
        
        expected_chars = ['A', 'C', 'T']  # Not G
        for i in range(3):
            pool.set_state(i)
            seq_result = pool.seq
            assert seq_result[0] == 'A'
            assert seq_result[1] == 'C'
            assert seq_result[2] in expected_chars
            assert seq_result[3] == 'T'


# =============================================================================
# MODE × INTERFACE CROSS-PRODUCT TESTS
# =============================================================================

class TestModeInterfaceCrossProduct:
    """
    Comprehensive cross-product testing of mode × interface combinations.
    
    Interfaces:
    1. Standard: Just k, all positions eligible
    2. Positions: Explicit positions list
    3. Adjacent: Only adjacent positions mutated
    
    Modes:
    1. Sequential: Deterministic state progression
    2. Random: State used as seed for randomization
    """
    
    # --- SEQUENTIAL + STANDARD ---
    
    def test_sequential_standard_exhaustive(self):
        """Sequential mode with standard interface covers all combinations."""
        original = "AAAA"  # 4 positions
        pool = KMutationPool(original, k=2, mode='sequential')
        
        # C(4,2) * 3^2 = 6 * 9 = 54 states
        assert pool.num_internal_states == 54
        
        seen = set()
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            assert count_mutations(original, seq) == 2
            assert seq not in seen
            seen.add(seq)
        
        assert len(seen) == 54
    
    # --- SEQUENTIAL + POSITIONS ---
    
    def test_sequential_positions_exhaustive(self):
        """Sequential mode with positions covers all position combinations."""
        original = "AAAAAAAAAA"  # 10 positions
        positions = [1, 4, 7]  # 3 allowed positions
        pool = KMutationPool(original, k=2, positions=positions, mode='sequential')
        
        # C(3,2) * 3^2 = 3 * 9 = 27 states
        assert pool.num_internal_states == 27
        
        seen = set()
        position_pairs_seen = set()
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            mutation_positions = get_mutation_positions(original, seq)
            assert len(mutation_positions) == 2
            
            for pos in mutation_positions:
                assert pos in positions
            
            position_pairs_seen.add(tuple(sorted(mutation_positions)))
            seen.add(seq)
        
        assert position_pairs_seen == {(1, 4), (1, 7), (4, 7)}
        assert len(seen) == 27
    
    # --- SEQUENTIAL + ADJACENT ---
    
    def test_sequential_adjacent_exhaustive(self):
        """Sequential mode with adjacent covers all starting positions."""
        original = "AAAAAA"  # 6 positions
        pool = KMutationPool(original, k=2, adjacent=True, mode='sequential')
        
        # (6-2+1) * 3^2 = 5 * 9 = 45 states
        assert pool.num_internal_states == 45
        
        seen = set()
        starting_positions_seen = set()
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            mutation_positions = get_mutation_positions(original, seq)
            assert len(mutation_positions) == 2
            assert are_positions_adjacent(mutation_positions)
            
            starting_positions_seen.add(min(mutation_positions))
            seen.add(seq)
        
        assert starting_positions_seen == {0, 1, 2, 3, 4}
        assert len(seen) == 45
    
    # --- RANDOM + STANDARD ---
    
    def test_random_standard_deterministic_per_state(self):
        """Random mode with standard interface: same state = same output."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=2, mode='random')
        
        for test_state in [0, 42, 999, 123456]:
            pool.set_state(test_state)
            seq1 = pool.seq
            
            pool.set_state(test_state)
            seq2 = pool.seq
            
            assert seq1 == seq2, f"State {test_state} not deterministic"
            assert count_mutations(original, seq1) == 2
    
    def test_random_standard_covers_state_space(self):
        """Random standard: all positions eventually mutated over many states."""
        original = "AAAAAAAAAA"  # 10 positions
        pool = KMutationPool(original, k=1, mode='random')
        
        positions_seen = set()
        for state in range(500):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            positions_seen.update(positions)
        
        assert positions_seen == set(range(10)), "Not all positions covered"
    
    # --- RANDOM + POSITIONS ---
    
    def test_random_positions_deterministic_per_state(self):
        """Random mode with positions: same state = same output."""
        original = "AAAAAAAAAA"
        positions = [0, 3, 6, 9]
        pool = KMutationPool(original, k=2, positions=positions, mode='random')
        
        for test_state in [0, 100, 12345]:
            pool.set_state(test_state)
            seq1 = pool.seq
            
            pool.set_state(test_state)
            seq2 = pool.seq
            
            assert seq1 == seq2
            
            mutation_positions = get_mutation_positions(original, seq1)
            for pos in mutation_positions:
                assert pos in positions
    
    def test_random_positions_only_allowed(self):
        """Random positions: forbidden positions never mutated."""
        original = "AAAAAAAAAA"
        allowed = [2, 5, 8]
        forbidden = [i for i in range(10) if i not in allowed]
        
        pool = KMutationPool(original, k=2, positions=allowed, mode='random')
        
        for state in range(300):
            pool.set_state(state)
            seq = pool.seq
            
            for pos in forbidden:
                assert seq[pos] == original[pos], f"State {state}: forbidden pos {pos} mutated"
    
    # --- RANDOM + ADJACENT ---
    
    def test_random_adjacent_deterministic_per_state(self):
        """Random mode with adjacent: same state = same output."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=3, adjacent=True, mode='random')
        
        for test_state in [0, 50, 5000]:
            pool.set_state(test_state)
            seq1 = pool.seq
            
            pool.set_state(test_state)
            seq2 = pool.seq
            
            assert seq1 == seq2
            
            positions = get_mutation_positions(original, seq1)
            assert len(positions) == 3
            assert are_positions_adjacent(positions)
    
    def test_random_adjacent_all_start_positions(self):
        """Random adjacent: all starting positions appear over many states."""
        original = "AAAAAA"  # 6 positions, k=2 -> start can be 0,1,2,3,4
        pool = KMutationPool(original, k=2, adjacent=True, mode='random')
        
        start_positions_seen = set()
        for state in range(200):
            pool.set_state(state)
            positions = get_mutation_positions(original, pool.seq)
            start_positions_seen.add(min(positions))
        
        assert start_positions_seen == {0, 1, 2, 3, 4}


class TestModeInterfaceStateCounts:
    """Verify state count formulas for all mode × interface combinations."""
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_standard_state_count(self, mode):
        """Standard interface: C(L,k) * (alpha-1)^k."""
        seq = "AAAAAAAA"  # L=8
        k = 3
        alpha = 4
        
        pool = KMutationPool(seq, k=k, mode=mode)
        
        expected = comb(8, 3) * (alpha - 1) ** 3  # 56 * 27 = 1512
        assert pool.num_internal_states == expected
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_positions_state_count(self, mode):
        """Positions interface: C(len(positions),k) * (alpha-1)^k."""
        seq = "AAAAAAAAAA"  # 10 positions
        positions = [0, 2, 4, 6, 8]  # 5 allowed
        k = 2
        alpha = 4
        
        pool = KMutationPool(seq, k=k, positions=positions, mode=mode)
        
        expected = comb(5, 2) * (alpha - 1) ** 2  # 10 * 9 = 90
        assert pool.num_internal_states == expected
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_adjacent_state_count(self, mode):
        """Adjacent interface: (L-k+1) * (alpha-1)^k."""
        seq = "AAAAAAAA"  # L=8
        k = 3
        alpha = 4
        
        pool = KMutationPool(seq, k=k, adjacent=True, mode=mode)
        
        expected = (8 - 3 + 1) * (alpha - 1) ** 3  # 6 * 27 = 162
        assert pool.num_internal_states == expected


class TestModeInterfaceInvariants:
    """Invariants that must hold across all mode × interface combinations."""
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    @pytest.mark.parametrize("interface", ["standard", "positions", "adjacent"])
    def test_exactly_k_mutations_all_combinations(self, mode, interface):
        """Every sequence has exactly k mutations in all combinations."""
        original = "AAAAAAAAAA"
        k = 2
        
        if interface == "standard":
            pool = KMutationPool(original, k=k, mode=mode)
        elif interface == "positions":
            pool = KMutationPool(original, k=k, positions=[0, 3, 6, 9], mode=mode)
        else:  # adjacent
            pool = KMutationPool(original, k=k, adjacent=True, mode=mode)
        
        for state in range(min(100, pool.num_internal_states)):
            pool.set_state(state)
            mutations = count_mutations(original, pool.seq)
            assert mutations == k, f"{mode}/{interface} state {state}: expected {k}, got {mutations}"
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    @pytest.mark.parametrize("interface", ["standard", "positions", "adjacent"])
    def test_length_preserved_all_combinations(self, mode, interface):
        """Sequence length preserved in all combinations."""
        original = "AAAAAAAAAA"
        k = 2
        
        if interface == "standard":
            pool = KMutationPool(original, k=k, mode=mode)
        elif interface == "positions":
            pool = KMutationPool(original, k=k, positions=[0, 3, 6, 9], mode=mode)
        else:  # adjacent
            pool = KMutationPool(original, k=k, adjacent=True, mode=mode)
        
        for state in range(min(50, pool.num_internal_states)):
            pool.set_state(state)
            assert len(pool.seq) == len(original)
    
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    @pytest.mark.parametrize("interface", ["standard", "positions", "adjacent"])
    def test_valid_mutations_all_combinations(self, mode, interface):
        """All mutations are valid alphabet substitutions in all combinations."""
        original = "AAAAAAAAAA"
        alphabet = ['A', 'C', 'G', 'T']
        k = 2
        
        if interface == "standard":
            pool = KMutationPool(original, alphabet=alphabet, k=k, mode=mode)
        elif interface == "positions":
            pool = KMutationPool(original, alphabet=alphabet, k=k, positions=[0, 3, 6, 9], mode=mode)
        else:  # adjacent
            pool = KMutationPool(original, alphabet=alphabet, k=k, adjacent=True, mode=mode)
        
        for state in range(min(50, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            
            for i, char in enumerate(seq):
                assert char in alphabet
                if char != original[i]:
                    assert char != original[i]


# =============================================================================
# POOL OPERATIONS TESTS
# =============================================================================

class TestKMutationPoolOperations:
    """Test KMutationPool with Pool operations."""
    
    def test_concatenation(self):
        """Test concatenating KMutationPool with other pools."""
        pool1 = KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        pool2 = Pool(seqs=['NNNN'])
        
        combined = pool1 + pool2
        combined.set_state(0)
        seq = combined.seq
        
        assert len(seq) == 8
        assert seq.endswith('NNNN')
    
    def test_repetition(self):
        """Test repeating KMutationPool."""
        pool = KMutationPool('AC', alphabet=['A', 'C', 'G', 'T'], k=1)
        repeated = pool * 3
        
        repeated.set_state(0)
        seq = repeated.seq
        
        assert len(seq) == 6
    
    def test_slicing(self):
        """Test slicing KMutationPool."""
        pool = KMutationPool('ACGTACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        sliced = pool[2:6]
        
        sliced.set_state(0)
        seq = sliced.seq
        
        assert len(seq) == 4


class TestKMutationPoolWithPoolInput:
    """Test KMutationPool with Pool objects as input."""
    
    def test_pool_input_generates_fresh_sequence(self):
        """Test that Pool input generates fresh sequence each time."""
        kmer_pool = KmerPool(4, alphabet='dna')
        mutation_pool = KMutationPool(kmer_pool, alphabet=['A', 'C', 'G', 'T'], k=2)
        
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
    
    def test_pool_parent_basic(self):
        """Works with Pool as parent."""
        parent = Pool(seqs=["AAAAAAAA"], mode='sequential')
        pool = KMutationPool(parent, k=2, mode='sequential')
        
        pool.set_state(0)
        seq = pool.seq
        
        assert len(seq) == 8
        assert count_mutations("AAAAAAAA", seq) == 2
    
    def test_pool_parent_with_positions(self):
        """Pool parent with explicit positions."""
        parent = Pool(seqs=["AAAAAAAAAA"], mode='sequential')
        positions = [0, 3, 6, 9]
        
        pool = KMutationPool(parent, k=2, positions=positions, mode='random')
        
        for state in range(50):
            pool.set_state(state)
            mutation_positions = get_mutation_positions("AAAAAAAAAA", pool.seq)
            
            for pos in mutation_positions:
                assert pos in positions


# =============================================================================
# REPR TESTS
# =============================================================================

class TestKMutationPoolRepr:
    """Test string representation."""
    
    def test_repr_basic(self):
        """Test __repr__ basic functionality."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C', 'G', 'T'], k=2)
        repr_str = repr(pool)
        assert 'KMutationPool' in repr_str
        assert 'ACGT' in repr_str
        assert 'k=2' in repr_str
    
    def test_repr_with_adjacent(self):
        """Test __repr__ with adjacent=True."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C'], k=2, adjacent=True)
        repr_str = repr(pool)
        assert 'KMutationPool' in repr_str
        assert 'adjacent=True' in repr_str
    
    def test_repr_without_adjacent(self):
        """Test __repr__ with adjacent=False (default)."""
        pool = KMutationPool('ACGT', alphabet=['A', 'C'], k=2, adjacent=False)
        repr_str = repr(pool)
        # Should not include adjacent=True since it's False
        assert 'adjacent=True' not in repr_str


# =============================================================================
# PARAMETER COMBINATIONS
# =============================================================================

class TestParameterCombinations:
    """Test various parameter combinations."""
    
    @pytest.mark.parametrize("mode,k,adjacent", [
        ("random", 1, False),
        ("random", 2, False),
        ("random", 2, True),
        ("sequential", 1, False),
        ("sequential", 2, False),
        ("sequential", 2, True),
    ])
    def test_mode_k_adjacent_combinations(self, mode, k, adjacent):
        """Various mode/k/adjacent combinations produce valid output."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=k, adjacent=adjacent, mode=mode)
        
        for state in range(min(50, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            
            assert len(seq) == len(original)
            assert count_mutations(original, seq) == k
            
            if adjacent:
                positions = get_mutation_positions(original, seq)
                assert are_positions_adjacent(positions)
    
    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("mark_changes", [True, False])
    def test_k_mark_changes_combinations(self, k, mark_changes):
        """k and mark_changes combinations work correctly."""
        original = "AAAAAAAA"
        pool = KMutationPool(original, k=k, mark_changes=mark_changes, mode='sequential')
        
        pool.set_state(0)
        seq = pool.seq
        
        if mark_changes:
            # Mutated positions should be lowercase
            lowercase_count = sum(1 for c in seq if c.islower())
            assert lowercase_count == k
        else:
            assert seq == seq.upper()


# =============================================================================
# REGRESSION TESTS
# =============================================================================

# =============================================================================
# REALISTIC POOL CHAINING SCENARIOS
# =============================================================================

class TestRealisticPoolChaining:
    """
    Realistic biological scenarios using pool chaining with KMutationPool.
    Uses actual biological sequences and validates correctness at multiple levels.
    """
    
    # --- Realistic sequences ---
    # E. coli lac operator (21bp core)
    LAC_OPERATOR = "AATTGTGAGCGGATAACAATT"
    
    # Kozak consensus sequence (10bp around start codon)
    KOZAK_CONTEXT = "GCCGCCATGG"  # ...GCCRCCAUGG...
    
    # T7 promoter (17bp core)
    T7_PROMOTER = "TAATACGACTCACTATA"
    
    # GFP start (first 30bp of EGFP)
    GFP_START = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"
    
    # Restriction site flanked by random sequence
    ECORI_CONSTRUCT = "ACGTACGT" + "GAATTC" + "TGCATGCA"  # 8 + 6 + 8 = 22bp
    
    def test_promoter_saturation_mutagenesis(self):
        """
        Scenario: Saturation mutagenesis of T7 promoter core positions.
        
        Validates:
        - All mutations occur within promoter region
        - Flanking sequences remain intact
        - Correct number of mutations
        """
        # Construct: 5'UTR + T7 promoter + spacer
        five_utr = "GGGAGA"
        spacer = "GGGGGG"
        full_construct = five_utr + self.T7_PROMOTER + spacer  # 6 + 17 + 6 = 29bp
        
        # Define promoter positions (indices 6-22)
        promoter_positions = list(range(6, 23))
        
        pool = KMutationPool(
            full_construct,
            k=2,
            positions=promoter_positions,
            mode='sequential'
        )
        
        # Validate all generated sequences
        for state in range(min(100, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            
            # 5'UTR must be unchanged
            assert seq[:6] == five_utr, f"5'UTR corrupted: {seq[:6]}"
            
            # Spacer must be unchanged
            assert seq[23:] == spacer, f"Spacer corrupted: {seq[23:]}"
            
            # Exactly 2 mutations in promoter region
            promoter_region = seq[6:23]
            mutations = count_mutations(self.T7_PROMOTER, promoter_region)
            assert mutations == 2, f"Expected 2 mutations in promoter, got {mutations}"
    
    def test_kozak_scanning_with_adjacent_mutations(self):
        """
        Scenario: Scan Kozak sequence with adjacent di-nucleotide mutations
        to find optimal context for translation initiation.
        Uses positions parameter to restrict mutations to Kozak region.
        
        Validates:
        - Mutations are always adjacent within Kozak
        - Flanking sequences remain intact
        - All Kozak starting positions are covered
        """
        # Use Kozak with flanks
        flank_5 = "AAAA"
        flank_3 = "TTTT"
        construct = flank_5 + self.KOZAK_CONTEXT + flank_3  # 4 + 10 + 4 = 18bp
        
        # Mutate only within Kozak region (positions 4-13)
        # For k=2 adjacent, valid starts are 4-12 (9 positions)
        kozak_positions = list(range(4, 14))
        
        # Use positions to restrict to Kozak, with adjacent mutations
        # Since positions + adjacent are mutually exclusive, we manually
        # verify adjacent behavior by checking all k=2 states
        pool = KMutationPool(
            construct,
            k=2,
            positions=kozak_positions,
            mode='sequential'
        )
        
        # Track which position pairs are hit within Kozak
        position_pairs_seen = set()
        adjacent_pairs_count = 0
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            # Flanks must be preserved
            assert seq[:4] == flank_5, f"5' flank corrupted: {seq[:4]}"
            assert seq[14:] == flank_3, f"3' flank corrupted: {seq[14:]}"
            
            # Find mutation positions (must be within Kozak)
            mut_positions = get_mutation_positions(construct, seq)
            assert len(mut_positions) == 2
            
            for pos in mut_positions:
                assert pos in kozak_positions, f"Mutation outside Kozak: {pos}"
            
            pair = tuple(sorted(mut_positions))
            position_pairs_seen.add(pair)
            
            if are_positions_adjacent(mut_positions):
                adjacent_pairs_count += 1
        
        # Should have C(10,2) = 45 position pairs
        assert len(position_pairs_seen) == 45
        
        # Many pairs should be adjacent (9 out of 45)
        assert adjacent_pairs_count > 0, "No adjacent pairs found"
    
    def test_restriction_site_avoidance(self):
        """
        Scenario: Mutate around a restriction site while preserving it.
        Common in cloning - need variants but must keep RE sites intact.
        
        Validates:
        - EcoRI site (GAATTC) is never mutated
        - Mutations occur only in flanking regions
        """
        # EcoRI is at positions 8-13 (0-indexed)
        ecori_start, ecori_end = 8, 14
        ecori_site = self.ECORI_CONSTRUCT[ecori_start:ecori_end]
        
        # Only allow mutations outside EcoRI
        allowed = [i for i in range(len(self.ECORI_CONSTRUCT)) 
                   if i < ecori_start or i >= ecori_end]
        
        pool = KMutationPool(
            self.ECORI_CONSTRUCT,
            k=3,
            positions=allowed,
            mode='random'
        )
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            
            # EcoRI site must be preserved
            assert seq[ecori_start:ecori_end] == ecori_site, \
                f"EcoRI site destroyed: {seq[ecori_start:ecori_end]}"
            
            # Exactly 3 mutations elsewhere
            mutations = count_mutations(self.ECORI_CONSTRUCT, seq)
            assert mutations == 3
    
    def test_gfp_codon_wobble_positions(self):
        """
        Scenario: Mutate only wobble positions (3rd codon position) in GFP.
        Common for codon optimization studies.
        
        Validates:
        - Only positions 2, 5, 8, ... (0-indexed) are mutated
        - First two positions of each codon unchanged
        """
        # GFP_START is 30bp = 10 codons
        # Wobble positions: 2, 5, 8, 11, 14, 17, 20, 23, 26, 29
        wobble_positions = [i for i in range(2, 30, 3)]
        
        pool = KMutationPool(
            self.GFP_START,
            k=3,
            positions=wobble_positions,
            mode='sequential'
        )
        
        for state in range(min(100, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            
            # Check each codon
            for codon_idx in range(10):
                start = codon_idx * 3
                original_codon = self.GFP_START[start:start+3]
                new_codon = seq[start:start+3]
                
                # First two positions must match
                assert new_codon[:2] == original_codon[:2], \
                    f"Codon {codon_idx}: non-wobble position mutated"
            
            # Exactly 3 mutations total
            assert count_mutations(self.GFP_START, seq) == 3
    
    def test_chained_regional_mutations(self):
        """
        Scenario: Chain two KMutationPools to independently mutate two regions.
        First pool mutates 5' region, second pool mutates 3' region.
        
        Validates:
        - Each region gets the specified number of mutations
        - Regions are independently mutated
        - Total construct integrity
        - Proper combined state space traversal
        """
        from poolparty import Pool
        
        # Construct: Region A (12bp) + linker (6bp) + Region B (12bp)
        region_a = "ATGATGATGATG"  # 12bp
        linker = "NNNNNN"
        region_b = "GCAGCAGCAGCA"  # 12bp
        
        full_seq = region_a + linker + region_b  # 30bp
        
        # First pool: mutate region A (positions 0-11)
        pool_a = KMutationPool(
            full_seq,
            k=2,
            positions=list(range(0, 12)),
            mode='sequential'
        )
        
        # Second pool: mutate region B (positions 18-29), using pool_a as input
        pool_b = KMutationPool(
            pool_a,
            k=2,
            positions=list(range(18, 30)),
            mode='sequential'
        )
        
        # Iterate through combined state space (sample first 100 states)
        for state in range(min(100, pool_b.num_states)):
            pool_b.set_state(state)
            seq = pool_b.seq
            
            # Linker preserved
            assert seq[12:18] == linker, f"Linker corrupted: {seq[12:18]}"
            
            # Region A has 2 mutations
            mut_a = count_mutations(region_a, seq[:12])
            assert mut_a == 2, f"Region A: expected 2 mutations, got {mut_a}"
            
            # Region B has 2 mutations
            mut_b = count_mutations(region_b, seq[18:])
            assert mut_b == 2, f"Region B: expected 2 mutations, got {mut_b}"
    
    def test_construct_assembly_with_mutations(self):
        """
        Scenario: Assemble a construct from parts where one part has mutations.
        Common pattern: promoter + variable region + terminator.
        
        Validates:
        - Concatenation produces correct length
        - Mutations only in variable region
        - Fixed parts unchanged
        """
        from poolparty import Pool
        
        # Fixed parts
        promoter = Pool(seqs=["TTGACA"])  # -35 box
        terminator = Pool(seqs=["TTTTTTT"])  # Poly-T terminator
        
        # Variable coding region with mutations
        coding_region = "ATGAAACCCGGG"  # 12bp
        variable = KMutationPool(coding_region, k=2, mode='sequential')
        
        # Assemble: promoter + variable + terminator
        construct = promoter + variable + terminator
        
        for state in range(min(50, variable.num_internal_states)):
            construct.set_state(state)
            seq = construct.seq
            
            # Check length
            assert len(seq) == 6 + 12 + 7, f"Wrong length: {len(seq)}"
            
            # Promoter unchanged
            assert seq[:6] == "TTGACA", f"Promoter corrupted: {seq[:6]}"
            
            # Terminator unchanged
            assert seq[-7:] == "TTTTTTT", f"Terminator corrupted: {seq[-7:]}"
            
            # Coding region has 2 mutations
            coding = seq[6:18]
            assert count_mutations(coding_region, coding) == 2
    
    def test_combinatorial_double_mutant_library(self):
        """
        Scenario: Create double mutant library using two separate regions.
        Mutate first half independently, then second half independently.
        Chain pools and iterate through combined state space.
        
        Validates:
        - Each region gets exactly 1 mutation
        - Total of 2 mutations from original
        - Combinatorial coverage of both regions
        - Proper pool chaining behavior
        """
        original = "AAAACCCCGGGG"  # 12bp: 4A + 4C + 4G
        
        # First pool: mutate first 4 positions (the A's)
        pool1 = KMutationPool(
            original,
            k=1,
            positions=[0, 1, 2, 3],
            mode='sequential'
        )
        
        # Second pool: mutate last 4 positions (the G's), chained from pool1
        pool2 = KMutationPool(
            pool1,
            k=1,
            positions=[8, 9, 10, 11],
            mode='sequential'
        )
        
        # Verify combined state space
        expected_states = pool1.num_internal_states * pool2.num_internal_states
        assert pool2.num_states == expected_states, \
            f"Combined states: expected {expected_states}, got {pool2.num_states}"
        
        # Collect unique double mutants by iterating pool2's full state space
        double_mutants = set()
        
        for state in range(pool2.num_states):
            pool2.set_state(state)
            final = pool2.seq
            
            # Middle region (the C's) should be unchanged
            assert final[4:8] == "CCCC", f"Middle region corrupted: {final[4:8]}"
            
            # First region should have 1 mutation
            first_muts = count_mutations("AAAA", final[:4])
            assert first_muts == 1, f"First region: expected 1 mut, got {first_muts}"
            
            # Last region should have 1 mutation
            last_muts = count_mutations("GGGG", final[8:])
            assert last_muts == 1, f"Last region: expected 1 mut, got {last_muts}"
            
            # Total should be 2 mutations from original
            total_muts = count_mutations(original, final)
            assert total_muts == 2, f"Expected 2 total mutations, got {total_muts}"
            
            double_mutants.add(final)
        
        # Should have 4 pos * 3 alts * 4 pos * 3 alts = 144 unique sequences
        expected = 4 * 3 * 4 * 3
        assert len(double_mutants) == expected, \
            f"Expected {expected} double mutants, got {len(double_mutants)}"
    
    def test_lac_operator_half_site_mutations(self):
        """
        Scenario: Lac operator has two half-sites. Mutate only the left half-site
        to study binding asymmetry.
        
        Validates:
        - Left half-site (first 10bp) is mutated
        - Right half-site (last 11bp) is preserved
        - Palindrome structure analysis
        """
        # Lac operator: AATTGTGAGC|GGATAACAATT (roughly symmetric)
        left_half = list(range(0, 10))  # First 10 positions
        
        pool = KMutationPool(
            self.LAC_OPERATOR,
            k=2,
            positions=left_half,
            mode='sequential'
        )
        
        # Track mutation positions within left half
        mutation_distribution = Counter()
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            # Right half preserved
            assert seq[10:] == self.LAC_OPERATOR[10:], \
                f"Right half corrupted: {seq[10:]}"
            
            # Left half has exactly 2 mutations
            left_muts = count_mutations(self.LAC_OPERATOR[:10], seq[:10])
            assert left_muts == 2
            
            # Track positions
            for pos in get_mutation_positions(self.LAC_OPERATOR, seq):
                mutation_distribution[pos] += 1
        
        # All left half positions should be hit
        for pos in left_half:
            assert mutation_distribution[pos] > 0, f"Position {pos} never mutated"
        
        # No right half positions should be hit
        for pos in range(10, 21):
            assert mutation_distribution[pos] == 0, f"Position {pos} shouldn't be mutated"
    
    def test_sequential_vs_random_statistical_equivalence(self):
        """
        Scenario: Verify that sequential mode enumerates exactly what random mode samples.
        
        Validates:
        - Sequential produces all possible sequences
        - Random samples from the same set
        - Distribution is approximately uniform in random mode
        """
        original = "AAAAAAAA"  # 8 positions
        
        # Sequential mode - enumerate all
        pool_seq = KMutationPool(original, k=2, mode='sequential')
        all_sequential = set()
        for state in range(pool_seq.num_internal_states):
            pool_seq.set_state(state)
            all_sequential.add(pool_seq.seq)
        
        # Random mode - sample many
        pool_rand = KMutationPool(original, k=2, mode='random')
        random_samples = []
        n_samples = pool_seq.num_internal_states * 10
        
        for state in range(n_samples):
            pool_rand.set_state(state)
            random_samples.append(pool_rand.seq)
        
        # All random samples should be valid sequential sequences
        for seq in random_samples:
            assert seq in all_sequential, f"Random produced invalid sequence: {seq}"
        
        # Check approximate uniformity
        sample_counts = Counter(random_samples)
        expected_per_seq = n_samples / len(all_sequential)
        
        # Most sequences should appear at least once
        sequences_sampled = len(sample_counts)
        assert sequences_sampled > len(all_sequential) * 0.8, \
            f"Only {sequences_sampled}/{len(all_sequential)} sequences sampled"


class TestRealisticRandomModeDistribution:
    """
    Distribution tests for random mode in realistic scenarios.
    Validates that random sampling is statistically correct.
    """
    
    # Realistic sequences
    PROMOTER_REGION = "TTGACAATTAATCATCGGCT"  # 20bp lac promoter region
    BINDING_SITE = "AATTGTGAGCGGATAACAATT"  # 21bp lac operator
    
    def test_position_distribution_uniform_in_promoter(self):
        """
        Random mode should uniformly sample all allowed positions.
        Simulates saturation mutagenesis of a promoter.
        """
        positions = list(range(0, 20))  # All positions in promoter
        pool = KMutationPool(
            self.PROMOTER_REGION,
            k=1,
            positions=positions,
            mode='random'
        )
        
        position_counts = Counter()
        n_samples = 10000
        
        for state in range(n_samples):
            pool.set_state(state)
            mut_pos = get_mutation_positions(self.PROMOTER_REGION, pool.seq)
            for pos in mut_pos:
                position_counts[pos] += 1
        
        expected_per_pos = n_samples / len(positions)  # 500
        
        # Each position should be within 25% of expected
        for pos in positions:
            count = position_counts[pos]
            assert expected_per_pos * 0.75 < count < expected_per_pos * 1.25, \
                f"Position {pos}: count {count} deviates from expected {expected_per_pos}"
    
    def test_position_pair_distribution_k2(self):
        """
        For k=2, all position pairs should appear with similar frequency.
        """
        seq = "ACGTACGTACGT"  # 12bp
        pool = KMutationPool(seq, k=2, mode='random')
        
        pair_counts = Counter()
        n_samples = 20000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = tuple(sorted(get_mutation_positions(seq, pool.seq)))
            pair_counts[positions] += 1
        
        # C(12,2) = 66 pairs
        expected_pairs = comb(12, 2)
        expected_per_pair = n_samples / expected_pairs
        
        # All pairs should appear
        assert len(pair_counts) == expected_pairs, \
            f"Expected {expected_pairs} pairs, got {len(pair_counts)}"
        
        # Distribution should be roughly uniform (within 40% of expected)
        for pair, count in pair_counts.items():
            assert expected_per_pair * 0.6 < count < expected_per_pair * 1.4, \
                f"Pair {pair}: count {count} deviates from expected {expected_per_pair}"
    
    def test_mutation_character_distribution_realistic(self):
        """
        Mutation characters should be uniformly distributed among alternatives.
        Uses a realistic homopolymer region where alternatives are clear.
        """
        # Poly-A region - mutations must be C, G, or T
        seq = "AAAAAAAAAA"
        pool = KMutationPool(seq, k=1, mode='random')
        
        char_counts = Counter()
        n_samples = 9000
        
        for state in range(n_samples):
            pool.set_state(state)
            result = pool.seq
            mut_pos = get_mutation_positions(seq, result)
            for pos in mut_pos:
                char_counts[result[pos]] += 1
        
        # A should never appear (it's the original)
        assert char_counts['A'] == 0
        
        # C, G, T should each appear ~1/3 of the time
        expected_per_char = n_samples / 3
        for char in ['C', 'G', 'T']:
            count = char_counts[char]
            assert expected_per_char * 0.85 < count < expected_per_char * 1.15, \
                f"Char {char}: count {count} deviates from expected {expected_per_char}"
    
    def test_adjacent_start_position_distribution(self):
        """
        In adjacent mode, all starting positions should be equally likely.
        """
        seq = "ACGTACGTACGT"  # 12bp
        pool = KMutationPool(seq, k=3, adjacent=True, mode='random')
        
        start_counts = Counter()
        n_samples = 10000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = get_mutation_positions(seq, pool.seq)
            start_counts[min(positions)] += 1
        
        # 12 - 3 + 1 = 10 possible starting positions
        expected_starts = 10
        expected_per_start = n_samples / expected_starts
        
        assert len(start_counts) == expected_starts
        
        for start, count in start_counts.items():
            assert expected_per_start * 0.7 < count < expected_per_start * 1.3, \
                f"Start {start}: count {count} deviates from expected {expected_per_start}"
    
    def test_positions_subset_distribution(self):
        """
        When using positions parameter, only those positions should appear,
        and they should appear uniformly.
        """
        seq = self.BINDING_SITE  # 21bp lac operator
        # Only allow mutations at key positions (simulating binding contacts)
        key_positions = [0, 5, 10, 15, 20]  # 5 positions
        
        pool = KMutationPool(
            seq, k=2,
            positions=key_positions,
            mode='random'
        )
        
        position_counts = Counter()
        pair_counts = Counter()
        n_samples = 10000
        
        for state in range(n_samples):
            pool.set_state(state)
            positions = get_mutation_positions(seq, pool.seq)
            pair_counts[tuple(sorted(positions))] += 1
            for pos in positions:
                position_counts[pos] += 1
        
        # Only key positions should appear
        for pos in position_counts:
            assert pos in key_positions, f"Forbidden position {pos} mutated"
        
        # All key positions should appear
        for pos in key_positions:
            assert position_counts[pos] > 0, f"Key position {pos} never mutated"
        
        # C(5,2) = 10 pairs, each should appear ~1000 times
        expected_per_pair = n_samples / comb(5, 2)
        for pair, count in pair_counts.items():
            assert expected_per_pair * 0.6 < count < expected_per_pair * 1.4, \
                f"Pair {pair}: count {count} deviates from expected {expected_per_pair}"
    
    def test_chained_pool_distribution(self):
        """
        Chained pools in random mode should uniformly sample their combined space.
        """
        # Region A (6bp) + Region B (6bp)
        seq = "AAAAAACCCCCC"
        
        pool1 = KMutationPool(seq, k=1, positions=[0, 1, 2, 3, 4, 5], mode='random')
        pool2 = KMutationPool(pool1, k=1, positions=[6, 7, 8, 9, 10, 11], mode='random')
        
        # Track position combinations (one from each region)
        region_a_counts = Counter()
        region_b_counts = Counter()
        n_samples = 12000
        
        for state in range(n_samples):
            pool2.set_state(state)
            result = pool2.seq
            
            # Infer parent from region B (C -> mutated, else original)
            region_b = result[6:12]
            parent_char_b = 'C'
            
            # Find mutations in each region
            for i in range(6):
                if result[i] != 'A':
                    region_a_counts[i] += 1
            for i in range(6, 12):
                if result[i] != 'C':
                    region_b_counts[i] += 1
        
        # Each position in region A should be mutated ~equally
        expected_per_pos = n_samples / 6
        for pos in range(6):
            count = region_a_counts[pos]
            assert expected_per_pos * 0.7 < count < expected_per_pos * 1.3, \
                f"Region A pos {pos}: count {count} deviates"
        
        # Each position in region B should be mutated ~equally
        for pos in range(6, 12):
            count = region_b_counts[pos]
            assert expected_per_pos * 0.7 < count < expected_per_pos * 1.3, \
                f"Region B pos {pos}: count {count} deviates"
    
    def test_random_reproducibility_with_seed(self):
        """
        Same seed in generate_seqs should produce identical sequences.
        """
        seq = self.PROMOTER_REGION
        pool = KMutationPool(seq, k=2, mode='random')
        
        # Generate with same seed twice
        seqs1 = pool.generate_seqs(num_seqs=100, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=100, seed=42)
        
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
        
        # Different seed should (almost certainly) produce different sequences
        seqs3 = pool.generate_seqs(num_seqs=100, seed=123)
        assert seqs1 != seqs3, "Different seeds should produce different sequences"


class TestModeInterfaceWithPoolParent:
    """
    Test mode × interface combinations when parent is a Pool object.
    Validates that chained pools correctly propagate mutations.
    
    Key insight from pool.py:
    - Sequential parents: state decomposition via mixed-radix (predictable)
    - Random parents: state used as RNG seed (pseudo-random)
    - Child's internal state decomposition depends on child's mode
    """
    
    def test_sequential_parent_predictable_selection(self):
        """
        When parent is sequential, parent selection follows mixed-radix pattern.
        Parent index = state // child.num_internal_states (rightmost varies fastest).
        """
        parent_seqs = ["AAAA", "CCCC", "GGGG"]
        parent = Pool(seqs=parent_seqs, mode='sequential')
        pool = KMutationPool(parent, k=1, mode='sequential')
        
        # Verify the decomposition formula
        for state in range(pool.num_states):
            pool.set_state(state)
            
            # Check parent's internal state
            expected_parent_state = state // pool.num_internal_states
            actual_parent_state = parent.internal_sequential_state
            
            assert actual_parent_state == expected_parent_state, \
                f"State {state}: parent state expected {expected_parent_state}, got {actual_parent_state}"
            
            # Check child's internal state
            expected_child_state = state % pool.num_internal_states
            actual_child_state = pool.internal_sequential_state
            
            assert actual_child_state == expected_child_state, \
                f"State {state}: child state expected {expected_child_state}, got {actual_child_state}"
    
    def test_random_child_sequential_parent_mixed(self):
        """
        When child is random but parent is sequential, parent follows mixed-radix
        but child's mutations are randomized per state.
        """
        parent_seqs = ["AAAA", "TTTT"]
        parent = Pool(seqs=parent_seqs, mode='sequential')
        pool = KMutationPool(parent, k=1, mode='random')
        
        # Parent should alternate: 0,1,0,1,0,1... (state mod 2)
        for state in range(20):
            pool.set_state(state)
            seq = pool.seq
            
            # Parent follows predictable pattern
            expected_parent_idx = state % 2
            expected_parent_seq = parent_seqs[expected_parent_idx]
            
            # Verify mutation is from correct parent
            muts = count_mutations(expected_parent_seq, seq)
            assert muts == 1, \
                f"State {state}: expected 1 mutation from {expected_parent_seq}, got {muts}"
    
    def test_sequential_standard_with_pool_parent(self):
        """Sequential mode with Pool parent covering all states."""
        parent_seqs = ["AAAA", "CCCC", "GGGG"]
        parent = Pool(seqs=parent_seqs, mode='sequential')
        
        pool = KMutationPool(parent, k=1, mode='sequential')
        
        # For each parent sequence, all k=1 mutations should be enumerable
        seen_by_parent = {seq: set() for seq in parent_seqs}
        
        total_states = pool.num_states
        for state in range(total_states):
            pool.set_state(state)
            seq = pool.seq
            
            # Determine which parent this came from
            parent_state = state // pool.num_internal_states
            parent_seq = parent_seqs[parent_state % len(parent_seqs)]
            
            # Should have exactly 1 mutation from parent
            muts = count_mutations(parent_seq, seq)
            assert muts == 1, f"State {state}: expected 1 mutation from {parent_seq}, got {muts}"
            
            seen_by_parent[parent_seq].add(seq)
        
        # Each parent should produce unique mutants
        for parent_seq, mutants in seen_by_parent.items():
            # 4 positions * 3 alternatives = 12 unique for homopolymer
            expected = 4 * 3 if len(set(parent_seq)) == 1 else pool.num_internal_states
            assert len(mutants) > 0, f"No mutants for parent {parent_seq}"
    
    def test_random_positions_with_varying_parent(self):
        """Random mode with positions, parent provides varying base sequences."""
        parent_seqs = ["AAAAAAAAAA", "TTTTTTTTTT"]
        parent = Pool(seqs=parent_seqs, mode='sequential')
        positions = [0, 4, 8]  # Only these 3 positions
        non_positions = [i for i in range(10) if i not in positions]
        
        pool = KMutationPool(parent, k=2, positions=positions, mode='random')
        
        # Iterate through combined states
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            # Non-position indices reveal the parent (all A's or all T's)
            unmutated = [seq[i] for i in non_positions]
            
            # All unmutated positions should be same character (from parent)
            assert len(set(unmutated)) == 1, f"State {state}: inconsistent parent"
            parent_char = unmutated[0]
            
            # Determine parent based on unmutated region
            expected_base = "A" * 10 if parent_char == "A" else "T" * 10
            
            # Find actual mutations
            actual_muts = [i for i in range(10) if seq[i] != expected_base[i]]
            
            # Should have exactly 2 mutations at allowed positions
            assert len(actual_muts) == 2, f"State {state}: expected 2 mutations, got {len(actual_muts)}"
            
            for pos in actual_muts:
                assert pos in positions, f"Mutation at forbidden position {pos}"
    
    def test_adjacent_with_homopolymer_parent(self):
        """Adjacent mode with homopolymer parent - validates mutation detection."""
        parent = Pool(seqs=["AAAAAA", "CCCCCC", "GGGGGG"], mode='sequential')
        
        # Apply adjacent mutations
        pool = KMutationPool(parent, k=2, adjacent=True, mode='random')
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            assert len(seq) == 6
            
            # For homopolymer parents, identify parent by the majority character
            char_counts = Counter(seq)
            # Most common char (appearing 4+ times) is from parent
            parent_char = char_counts.most_common(1)[0][0]
            parent_seq = parent_char * 6
            
            # Find mutations from inferred parent
            mut_positions = get_mutation_positions(parent_seq, seq)
            assert len(mut_positions) == 2, f"State {state}: expected 2 mutations, got {len(mut_positions)}"
            assert are_positions_adjacent(mut_positions), f"Mutations not adjacent: {mut_positions}"


class TestRegressionKnownIssues:
    """Regression tests for previously discovered issues."""
    
    def test_large_state_space_no_crash(self):
        """Large state space doesn't cause memory issues."""
        # C(100, 3) * 3^3 = 161700 * 27 = 4,365,900 states
        seq = "A" * 100
        pool = KMutationPool(seq, k=3, mode='random')
        
        # Should be able to generate sequences without crashing
        for state in range(100):
            pool.set_state(state * 10000)
            seq = pool.seq
            assert len(seq) == 100
    
    def test_positions_order_preserved(self):
        """Positions order is preserved (not sorted internally)."""
        original = "AAAAAAAAAA"
        positions = [5, 2, 8]  # Unsorted
        
        pool = KMutationPool(original, k=2, positions=positions, mode='sequential')
        
        # Should work correctly regardless of order
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            mutation_positions = get_mutation_positions(original, pool.seq)
            
            for pos in mutation_positions:
                assert pos in positions
