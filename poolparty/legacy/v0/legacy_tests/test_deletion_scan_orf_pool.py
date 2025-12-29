"""Tests for the refactored DeletionScanORFPool class.

Note: The following original tests are no longer applicable due to API changes:
- test_repr_with_seed: Seed parameter not supported in new API
- TestSetSeed: Seed parameter not supported in new API
- test_iter_protocol, test_next_advances_state: Iteration removed from base Pool
- offset parameter replaced by start
- shift parameter replaced by step_size
- mark_deletion parameter replaced by mark_changes
"""

import pytest
from poolparty import DeletionScanORFPool, Pool


def test_deletion_scan_orf_pool_creation():
    """Test DeletionScanORFPool creation."""
    orf = "ATGGCCAAA"  # 3 codons
    pool = DeletionScanORFPool(background_seq=orf, deletion_size=1)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_stores_codons():
    """Test that initialization properly handles codon-level operations."""
    orf = "ATGGCCAAA"  # 3 codons
    pool = DeletionScanORFPool(background_seq=orf, deletion_size=1, mark_changes=True)
    
    assert pool.codons == ['ATG', 'GCC', 'AAA']
    assert pool.num_codons == 3


class TestValidation:
    """Test input validation."""
    
    def test_orf_non_dna_characters_error(self):
        """Test that non-DNA characters in seq raise error."""
        with pytest.raises(ValueError, match="must contain only ACGT"):
            DeletionScanORFPool('ATGXXX', deletion_size=1)
    
    def test_orf_length_not_divisible_by_3_error(self):
        """Test that seq length not divisible by 3 raises error."""
        with pytest.raises(ValueError, match="(ORF|orf).*(length|Length).*divisible by 3"):
            DeletionScanORFPool('ATGG', deletion_size=1)
    
    def test_deletion_size_zero_error(self):
        """Test that deletion_size of 0 raises error."""
        with pytest.raises(ValueError, match="deletion_size must be > 0"):
            DeletionScanORFPool('ATGGCC', deletion_size=0)
    
    def test_deletion_size_longer_than_orf_error(self):
        """Test that deletion_size longer than orf raises error."""
        orf = "ATGGCC"  # 2 codons
        with pytest.raises(ValueError, match="deletion_size.*cannot.*exceed"):
            DeletionScanORFPool(orf, deletion_size=3)
    
    def test_position_weights_without_positions(self):
        """Test that position_weights without positions raises error."""
        with pytest.raises(ValueError, match="position_weights requires positions"):
            DeletionScanORFPool('ATGGCC', deletion_size=1, position_weights=[1.0, 2.0])
    
    def test_position_weights_with_sequential_mode(self):
        """Test that position_weights with sequential mode raises error."""
        with pytest.raises(ValueError, match="position_weights.*sequential"):
            DeletionScanORFPool(
                'ATGGCCAAA', deletion_size=1,
                positions=[0, 1],
                position_weights=[1.0, 2.0],
                mode='sequential'
            )
    
    def test_non_integer_position(self):
        """Test that non-integer position raises error."""
        with pytest.raises(ValueError, match="positions must be integers"):
            DeletionScanORFPool('ATGGCCAAA', deletion_size=1, positions=[0, 1.5])
    
    def test_position_weights_length_mismatch(self):
        """Test that position_weights length mismatch raises error."""
        with pytest.raises(ValueError, match="position_weights length"):
            DeletionScanORFPool(
                'ATGGCCAAA', deletion_size=1,
                positions=[0, 1],
                position_weights=[1.0],
                mode='random'
            )
    
    def test_position_weights_non_positive_sum(self):
        """Test that non-positive sum of weights raises error."""
        with pytest.raises(ValueError, match="Sum of position_weights must be positive"):
            DeletionScanORFPool(
                'ATGGCCAAA', deletion_size=1,
                positions=[0, 1],
                position_weights=[0.5, -0.5],
                mode='random'
            )
    
    def test_deletion_window_exceeds_orf(self):
        """Test that deletion position exceeding ORF raises error."""
        with pytest.raises(ValueError, match="(Position|window).*must fit within ORF"):
            DeletionScanORFPool(
                'ATGATGATG',  # 3 codons
                deletion_size=2,
                positions=[2]  # Position 2 + size 2 = 4, exceeds 3 codons
            )
    
    def test_cannot_mix_interfaces(self):
        """Test that mixing range and position interfaces raises error."""
        with pytest.raises(ValueError):
            DeletionScanORFPool(
                "ATGGCCAAA",
                deletion_size=1,
                start=0,
                positions=[0, 1]
            )
    
    def test_duplicate_positions_error(self):
        """Test that duplicate positions raise error."""
        with pytest.raises(ValueError, match="duplicates"):
            DeletionScanORFPool(
                "ATGGCCAAA",
                deletion_size=1,
                positions=[0, 1, 0]
            )
    
    def test_empty_range_error(self):
        """Test that empty range raises error."""
        with pytest.raises(ValueError, match="(empty|no valid)"):
            DeletionScanORFPool(
                "ATGATGATG",  # 3 codons
                deletion_size=3,
                start=1  # No room for 3-codon deletion starting at 1
            )


class TestMarkedDeletionMode:
    """Test marked deletion mode operations at codon level."""
    
    def test_num_states_calculation_basic(self):
        """Test num_states calculation for marked deletion mode at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        
        # L=6, W=2, start=0, step=1
        # range(0, 6-2+1, 1) = range(0, 5, 1) = [0, 1, 2, 3, 4] -> 5 states
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0
        )
        assert pool.num_internal_states == 5
    
    def test_num_states_with_step_size(self):
        """Test num_states calculation with larger step_size."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 1  # 1 codon
        
        # L=6, W=1, start=0, step=2
        # range(0, 6-1+1, 2) = range(0, 6, 2) = [0, 2, 4] -> 3 states
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=2, start=0
        )
        assert pool.num_internal_states == 3
    
    def test_num_states_with_start(self):
        """Test num_states calculation with non-zero start."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        deletion_size = 2  # 2 codons
        
        # L=5, W=2, start=1, step=1
        # range(1, 5-2+1, 1) = range(1, 4, 1) = [1, 2, 3] -> 3 states
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=1
        )
        assert pool.num_internal_states == 3
    
    def test_basic_operation(self):
        """Test basic marked deletion operation at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons: ATG-GCC-AAA-CCC-TTT-GGG
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Deletes codons 0-1 (ATG-GCC -> -------)
        assert pool.seq == "------AAACCCTTTGGG"
        
        pool.set_state(1)
        # Deletes codons 1-2 (GCC-AAA -> -------)
        assert pool.seq == "ATG------CCCTTTGGG"
        
        pool.set_state(2)
        # Deletes codons 2-3 (AAA-CCC -> -------)
        assert pool.seq == "ATGGCC------TTTGGG"
    
    def test_with_step_size(self):
        """Test marked deletion mode with larger step_size."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=2, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Position 0 (codons 0-1)
        assert pool.seq == "------AAACCCTTTGGG"
        
        pool.set_state(1)
        # Position 2 (codons 2-3)
        assert pool.seq == "ATGGCC------TTTGGG"
        
        pool.set_state(2)
        # Position 4 (codons 4-5)
        assert pool.seq == "ATGGCCAAACCC------"
    
    def test_with_start(self):
        """Test marked deletion mode with non-zero start."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=2, start=1,
            mode='sequential'
        )
        
        # start=1, step=2, positions = [1, 3]
        pool.set_state(0)
        assert pool.seq == "ATG------CCCTTTGGG"  # Position 1 (codons 1-2)
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCAAA------GGG"  # Position 3 (codons 3-4)
    
    def test_single_codon_deletion(self):
        """Test marked deletion mode with single codon deletion."""
        orf = "ATGGCCAAACCC"  # 4 codons
        deletion_size = 1  # 1 codon
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "---GCCAAACCC"  # Deletes codon 0 (ATG->---)
        
        pool.set_state(1)
        assert pool.seq == "ATG---AAACCC"  # Deletes codon 1 (GCC->---)
        
        pool.set_state(2)
        assert pool.seq == "ATGGCC---CCC"  # Deletes codon 2 (AAA->---)
    
    def test_custom_deletion_character(self):
        """Test marked deletion mode with custom deletion character."""
        orf = "ATGGCCAAA"  # 3 codons
        deletion_size = 1  # 1 codon
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, deletion_character='X',
            step_size=1, start=0, mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "XXXGCCAAA"  # Deletes codon 0 with 'X'
        
        pool.set_state(1)
        assert pool.seq == "ATGXXXAAA"  # Deletes codon 1 with 'X'


class TestUnmarkedDeletionMode:
    """Test unmarked deletion mode (actual removal) operations at codon level."""
    
    def test_num_states_calculation_basic(self):
        """Test num_states calculation for unmarked deletion mode at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        
        # Same formula as marked mode
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=1, start=0
        )
        assert pool.num_internal_states == 5
    
    def test_basic_operation(self):
        """Test basic unmarked deletion operation at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons: ATG-GCC-AAA-CCC-TTT-GGG
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Actually removes codons 0-1 (ATG-GCC removed)
        assert pool.seq == "AAACCCTTTGGG"
        assert len(pool.seq) == 12  # 4 codons * 3 = 12 nt
        
        pool.set_state(1)
        # Removes codons 1-2 (GCC-AAA removed)
        assert pool.seq == "ATGCCCTTTGGG"
        
        pool.set_state(2)
        # Removes codons 2-3 (AAA-CCC removed)
        assert pool.seq == "ATGGCCTTTGGG"
    
    def test_with_step_size(self):
        """Test unmarked deletion mode with larger step_size."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=2, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Position 0 (codons 0-1 removed)
        assert pool.seq == "AAACCCTTTGGG"
        
        pool.set_state(1)
        # Position 2 (codons 2-3 removed)
        assert pool.seq == "ATGGCCTTTGGG"
        
        pool.set_state(2)
        # Position 4 (codons 4-5 removed)
        assert pool.seq == "ATGGCCAAACCC"
    
    def test_with_start(self):
        """Test unmarked deletion mode with non-zero start."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=2, start=1,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "ATGCCCTTTGGG"  # Position 1 (codons 1-2 removed)
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCAAAGGG"  # Position 3 (codons 3-4 removed)
    
    def test_single_codon_deletion(self):
        """Test unmarked deletion mode with single codon deletion."""
        orf = "ATGGCCAAACCC"  # 4 codons
        deletion_size = 1  # 1 codon
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "GCCAAACCC"  # Removes codon 0 (ATG removed)
        assert len(pool.seq) == 9  # 3 codons * 3 = 9 nt
        
        pool.set_state(1)
        assert pool.seq == "ATGAAACCC"  # Removes codon 1 (GCC removed)
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCCCC"  # Removes codon 2 (AAA removed)


class TestSequenceLength:
    """Test sequence length calculations."""
    
    def test_length_marked_mode(self):
        """Test that marked deletion mode maintains orf length."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons = 15 nt
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        assert pool.seq_length == 15
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            assert len(pool.seq) == 15
    
    def test_length_unmarked_mode(self):
        """Test that unmarked deletion mode decreases length by deletion size."""
        orf = "ATGGCCAAA"  # 3 codons = 9 nt
        deletion_size = 1  # 1 codon = 3 nt
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=1, start=0,
            mode='sequential'
        )
        
        expected_length = 6  # (3 - 1) * 3 = 6 nt
        assert pool.seq_length == expected_length
        
        for state in range(min(pool.num_internal_states, 10)):
            pool.set_state(state)
            assert len(pool.seq) == expected_length


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_codon_orf(self):
        """Test with single codon ORF and single codon deletion."""
        orf = "ATG"  # 1 codon
        deletion_size = 1  # 1 codon
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "---"
        assert len(pool.seq) == 3
    
    def test_deletion_size_equal_to_orf(self):
        """Test deletion size equal to orf length."""
        orf = "ATGGCC"  # 2 codons
        deletion_size = 2  # 2 codons
        
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, mode='sequential'
        )
        
        assert pool.num_internal_states == 1
        pool.set_state(0)
        assert pool.seq == "------"
    
    def test_state_wrapping(self):
        """Test that state wraps with modulo."""
        orf = "ATGGCCAAA"
        deletion_size = 1
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        first_seq = pool.seq
        
        # Setting state beyond num_internal_states should wrap
        pool.set_state(pool.num_internal_states)
        assert pool.seq == first_seq
    
    def test_full_orf_deletion(self):
        """Test deleting entire ORF."""
        orf = "ATGATGATG"  # 3 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=3,
            mark_changes=True, mode='sequential'
        )
        assert pool.num_internal_states == 1
        pool.set_state(0)
        assert pool.seq == "---------"


class TestPositionBasedInterface:
    """Test position-based interface with explicit positions."""
    
    def test_explicit_positions(self):
        """Test with explicit positions list."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            positions=[0, 2, 4],
            mode='sequential'
        )
        
        assert pool.num_internal_states == 3
        
        pool.set_state(0)
        assert pool.seq == "---GCCAAACCCTTT"  # Position 0
        
        pool.set_state(1)
        assert pool.seq == "ATGGCC---CCCTTT"  # Position 2
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCAAACCC---"  # Position 4
    
    def test_weighted_positions(self):
        """Test position-based with weights in random mode."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            positions=[0, 4],
            position_weights=[9.0, 1.0],
            mode='random'
        )
        
        # Verify weighted sampling produces expected distribution
        counts = {0: 0, 4: 0}
        for state in range(500):
            pool.set_state(state)
            seq = pool.seq
            if seq.startswith("---"):
                counts[0] += 1
            else:
                counts[4] += 1
        
        # Position 0 should be selected much more often
        assert counts[0] > counts[4] * 4
    
    def test_weighted_positions_full_verification(self):
        """Test that weighted positions produce correct content AND distribution."""
        orf_seq = "AAACCCGGGTTT"  # 4 codons: AAA CCC GGG TTT
        
        pool = DeletionScanORFPool(
            background_seq=orf_seq,
            deletion_size=1,
            positions=[0, 3],  # Delete AAA or TTT
            position_weights=[1.0, 9.0],  # 10% pos 0, 90% pos 3
            mode='random'
        )
        
        counts = {"pos0": 0, "pos3": 0}
        for state in range(500):
            pool.set_state(state)
            seq = pool.seq
            codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
            
            if codons[0] == "---":
                counts["pos0"] += 1
                assert codons == ["---", "CCC", "GGG", "TTT"], f"Pos 0 delete wrong: {codons}"
            elif codons[3] == "---":
                counts["pos3"] += 1
                assert codons == ["AAA", "CCC", "GGG", "---"], f"Pos 3 delete wrong: {codons}"
            else:
                assert False, f"Unexpected deletion result: {codons}"
        
        # Weight distribution check
        assert counts["pos3"] > counts["pos0"] * 4, "Weight distribution not respected"


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_same_state_same_sequence(self):
        """Test that DeletionScanORFPool is deterministic with same state."""
        orf = "ATGGCCAAACCCTTT"
        deletion_size = 2
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(2)
        seq1 = pool.seq
        pool.set_state(2)
        seq2 = pool.seq
        assert seq1 == seq2
    
    def test_different_states_different_sequences(self):
        """Test that different states produce different sequences."""
        orf = "ATGGCCAAACCCTTT"
        deletion_size = 1
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        sequences = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            sequences.append(pool.seq)
        
        # All sequences should be different (for these parameters)
        assert len(sequences) == len(set(sequences))


class TestModes:
    """Test random and sequential modes."""
    
    def test_mode_parameter(self):
        """Test that mode parameter is accepted."""
        orf = "ATGGCCAAA"
        deletion_size = 1
        
        pool_random = DeletionScanORFPool(orf, deletion_size=deletion_size, mode='random')
        assert pool_random.mode == 'random'
        
        pool_sequential = DeletionScanORFPool(orf, deletion_size=deletion_size, mode='sequential')
        assert pool_sequential.mode == 'sequential'
    
    def test_sequential_iteration(self):
        """Test DeletionScanORFPool sequential iteration."""
        orf = "ATGGCCAAACCC"
        deletion_size = 1
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        sequences = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            sequences.append(pool.seq)
        
        # All sequences should be different
        assert len(sequences) == len(set(sequences))
    
    def test_is_sequential_compatible(self):
        """Test that pool is sequential compatible."""
        orf = "ATGGCCAAA"
        deletion_size = 1
        pool = DeletionScanORFPool(orf, deletion_size=deletion_size)
        
        assert pool.is_sequential_compatible()


class TestIntegration:
    """Test integration with Pool operations."""
    
    def test_concatenation(self):
        """Test DeletionScanORFPool concatenation."""
        orf = "ATGGCCAAA"
        deletion_size = 1
        pool1 = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        constant = Pool(seqs=["XXX"])
        
        combined = constant + pool1 + constant
        
        combined.set_state(0)
        assert combined.seq.startswith("XXX")
        assert combined.seq.endswith("XXX")
    
    def test_generate_seqs(self):
        """Test that DeletionScanORFPool works with generate_seqs."""
        orf = "ATGGCCAAA"
        deletion_size = 1
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        sequences = pool.generate_seqs(num_seqs=pool.num_internal_states)
        assert len(sequences) == pool.num_internal_states
        # All sequences should be different
        assert len(set(sequences)) == pool.num_internal_states
    
    def test_with_pool_background(self):
        """Test DeletionScanORFPool with Pool object as background."""
        orf_pool = Pool(seqs=["ATGGCCAAA"], mode='sequential')
        deletion_size = 1
        pool = DeletionScanORFPool(
            orf_pool, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "---GCCAAA"


class TestGenerateSeqs:
    """Test generate_seqs functionality and quirks."""
    
    def test_generate_seqs_with_seed_reproducibility(self):
        """Test that seed produces reproducible results with random mode."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mode='random'
        )
        
        # Same seed should produce same sequences
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
        
        # Different seed should (likely) produce different sequences
        seqs3 = pool.generate_seqs(num_seqs=10, seed=123)
        assert seqs1 != seqs3, "Different seeds should produce different sequences"
    
    def test_generate_seqs_with_weighted_positions(self):
        """Test generate_seqs respects position_weights in random mode."""
        orf = "AAACCCGGGTTT"  # 4 codons
        
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            positions=[0, 3],  # First or last codon
            position_weights=[9.0, 1.0],  # 90% position 0, 10% position 3
            mode='random'
        )
        
        sequences = pool.generate_seqs(num_seqs=500, seed=42)
        
        # Count which position was used
        pos0_count = sum(1 for s in sequences if s.startswith("---"))
        pos3_count = sum(1 for s in sequences if s.endswith("---"))
        
        # Position 0 should be much more common
        assert pos0_count > pos3_count * 4, f"Weights not respected: pos0={pos0_count}, pos3={pos3_count}"
    
    def test_generate_seqs_num_complete_iterations(self):
        """Test generate_seqs with num_complete_iterations."""
        orf = "ATGGCCAAA"  # 3 codons
        
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mode='sequential'
        )
        
        # 3 positions × 2 iterations = 6 sequences
        sequences = pool.generate_seqs(num_complete_iterations=2)
        assert len(sequences) == pool.num_internal_states * 2
        
        # First iteration
        first_iter = sequences[:pool.num_internal_states]
        # Second iteration
        second_iter = sequences[pool.num_internal_states:]
        
        # Both iterations should produce same sequences in same order
        assert first_iter == second_iter, "Iterations should repeat"
    
    def test_generate_seqs_return_computation_graph(self):
        """Test generate_seqs with return_computation_graph=True."""
        orf = "ATGGCCAAA"
        
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_computation_graph=True)
        
        assert isinstance(result, dict)
        assert "sequences" in result
        assert "graph" in result
        assert "node_sequences" in result
        
        assert len(result["sequences"]) == 3
        assert isinstance(result["graph"], dict)
        assert "nodes" in result["graph"]
    
    def test_generate_seqs_sequential_wraps(self):
        """Test that generate_seqs wraps around for sequential pools."""
        orf = "ATGGCC"  # 2 codons
        
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mode='sequential'
        )
        
        # Pool has 2 states, request 5 sequences
        sequences = pool.generate_seqs(num_seqs=5)
        assert len(sequences) == 5
        
        # Should wrap: [state0, state1, state0, state1, state0]
        assert sequences[0] == sequences[2] == sequences[4]
        assert sequences[1] == sequences[3]
    
    def test_generate_seqs_with_pool_parent_and_seed(self):
        """Test generate_seqs with Pool parent respects seed."""
        parent_pool = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='random')
        
        pool = DeletionScanORFPool(
            parent_pool, deletion_size=1,
            mode='random'
        )
        
        # Same seed should produce same results
        seqs1 = pool.generate_seqs(num_seqs=20, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=20, seed=42)
        assert seqs1 == seqs2
    
    def test_generate_seqs_mixed_modes(self):
        """Test generate_seqs with mixed sequential/random ancestry."""
        # Parent pool is sequential
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        # Child is random
        pool = DeletionScanORFPool(
            parent, deletion_size=1,
            positions=[0, 1, 2],  # 3 positions
            mode='random'
        )
        
        # With seed, should be reproducible
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        assert seqs1 == seqs2


class TestRepr:
    """Test string representation."""
    
    def test_repr_marked_mode(self):
        """Test DeletionScanORFPool __repr__ for marked mode."""
        orf = "ATGGCC"
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mark_changes=True, step_size=1, start=0
        )
        repr_str = repr(pool)
        assert "DeletionScanORFPool" in repr_str
    
    def test_repr_unmarked_mode(self):
        """Test DeletionScanORFPool __repr__ for unmarked mode."""
        orf = "ATGGCC"
        pool = DeletionScanORFPool(
            orf, deletion_size=1,
            mark_changes=False
        )
        repr_str = repr(pool)
        assert "DeletionScanORFPool" in repr_str
    
    def test_repr_with_long_sequence(self):
        """Test __repr__ with long sequence gets truncated."""
        orf = 'ATG' * 10  # 30 nucleotides
        pool = DeletionScanORFPool(orf, deletion_size=1)
        repr_str = repr(pool)
        
        assert "DeletionScanORFPool" in repr_str
        assert "..." in repr_str  # Should be truncated
    
    def test_repr_with_positions(self):
        """Test __repr__ with explicit positions."""
        orf = "ATGGCCAAA"
        pool = DeletionScanORFPool(orf, deletion_size=1, positions=[0, 2])
        repr_str = repr(pool)
        assert "DeletionScanORFPool" in repr_str
        assert "positions" in repr_str


class TestFlankingRegions:
    """Test flanking region (UTR) handling."""
    
    def test_with_flanking_regions(self):
        """Test DeletionScanORFPool with orf_start/orf_end."""
        # Full sequence: 5'UTR + ORF + 3'UTR
        full_seq = "GGGGG" + "ATGATGATG" + "CCCCC"  # 5 + 9 + 5 = 19 nt
        
        pool = DeletionScanORFPool(
            full_seq, deletion_size=1,
            orf_start=5, orf_end=14,
            mode='sequential'
        )
        
        assert pool.upstream_flank == "GGGGG"
        assert pool.downstream_flank == "CCCCC"
        assert pool.orf_seq == "ATGATGATG"
        
        # Check flanks are preserved in all outputs
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert seq.startswith("GGGGG"), f"5' UTR not preserved: {seq}"
            assert seq.endswith("CCCCC"), f"3' UTR not preserved: {seq}"
    
    def test_flanks_with_positions_verification(self):
        """Test that flanks are preserved and content is correct with explicit positions."""
        # Full construct: 5'UTR + ORF + 3'UTR
        full_seq = "GGGGG" + "ATGCCCGGGTTT" + "AAAAA"  # 5 + 12 + 5 = 22 nt
        
        pool = DeletionScanORFPool(
            background_seq=full_seq,
            deletion_size=1,
            positions=[1, 3],  # Delete CCC (pos 1) or TTT (pos 3)
            orf_start=5,
            orf_end=17,
            mode='sequential'
        )
        
        # State 0: delete at position 1 (CCC)
        pool.set_state(0)
        seq = pool.seq
        assert seq.startswith("GGGGG"), "5'UTR corrupted"
        assert seq.endswith("AAAAA"), "3'UTR corrupted"
        orf_part = seq[5:17]
        orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
        assert orf_codons == ["ATG", "---", "GGG", "TTT"]
        
        # State 1: delete at position 3 (TTT)
        pool.set_state(1)
        seq = pool.seq
        assert seq.startswith("GGGGG"), "5'UTR corrupted"
        assert seq.endswith("AAAAA"), "3'UTR corrupted"
        orf_part = seq[5:17]
        orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
        assert orf_codons == ["ATG", "CCC", "GGG", "---"]
    
    def test_flanks_with_unmarked_deletion_length_change(self):
        """Test that flanks are preserved when unmarked deletion changes ORF length."""
        # Full sequence: 5'UTR + ORF + 3'UTR
        full_seq = "CCCCC" + "ATGATGATG" + "GGGGG"  # 5 + 9 + 5 = 19 nt
        
        pool = DeletionScanORFPool(
            full_seq, deletion_size=1,
            mark_changes=False,  # Actually remove
            orf_start=5, orf_end=14,
            mode='sequential'
        )
        
        # Original ORF is 3 codons = 9 nt
        # After deletion, ORF is 2 codons = 6 nt
        # Total should be 5 + 6 + 5 = 16 nt
        expected_len = 16
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert len(seq) == expected_len, f"State {state}: len={len(seq)}, expected {expected_len}"
            assert seq.startswith("CCCCC"), f"5' UTR corrupted at state {state}: {seq}"
            assert seq.endswith("GGGGG"), f"3' UTR corrupted at state {state}: {seq}"
    
    def test_flanks_no_upstream(self):
        """Test with only downstream flank (orf_start=0)."""
        full_seq = "ATGATGATG" + "CCCCC"  # ORF starts at 0
        
        pool = DeletionScanORFPool(
            full_seq, deletion_size=1,
            orf_start=0, orf_end=9,
            mode='sequential'
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == "CCCCC"
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.endswith("CCCCC"), f"3' UTR not preserved: {seq}"
    
    def test_flanks_no_downstream(self):
        """Test with only upstream flank (orf_end=len)."""
        full_seq = "CCCCC" + "ATGATGATG"  # ORF ends at end
        
        pool = DeletionScanORFPool(
            full_seq, deletion_size=1,
            orf_start=5, orf_end=14,
            mode='sequential'
        )
        
        assert pool.upstream_flank == "CCCCC"
        assert pool.downstream_flank == ""
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.startswith("CCCCC"), f"5' UTR not preserved: {seq}"


class TestPoolChaining:
    """Test pool chaining / transformer pattern."""
    
    def test_pool_as_seq_input(self):
        """Test DeletionScanORFPool with Pool as seq input."""
        # Create a parent pool
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        pool = DeletionScanORFPool(
            parent, deletion_size=1,
            positions=[0],
            mode='sequential'
        )
        
        # Test with first parent sequence
        pool.set_state(0)
        assert pool.seq == "---ATGATG"
        
        # Test with second parent sequence
        pool.set_state(1)
        assert pool.seq == "---GGGAAA"
    
    def test_chain_with_length_change(self):
        """Test chaining where length changes in parent pool."""
        from poolparty import InsertionScanPool
        
        base_seq = "AAACCCGGGTTT"  # 12 nt
        
        # Insert 3 nt at position 3 - results in 15 nt
        insertion_pool = InsertionScanPool(
            background_seq=base_seq,
            insert_seq="ATG",
            insert_or_overwrite='insert',
            positions=[3],
            mode='sequential'
        )
        
        # Now delete 1 codon at position 0 of the LENGTHENED sequence
        deletion_pool = DeletionScanORFPool(
            insertion_pool, deletion_size=1,
            positions=[0],
            mark_changes=False,  # Actually remove
            mode='sequential'
        )
        
        deletion_pool.set_state(0)
        result = deletion_pool.seq
        # After insert: AAA ATG CCC GGG TTT (15 nt)
        # After delete at 0: ATG CCC GGG TTT (12 nt)
        assert result == "ATGCCCGGGTTT", f"Chain result wrong: {result}"


class TestStateSpaceCalculation:
    """Test state space and positions calculation."""
    
    def test_positions_list_range_mode(self):
        """Test that positions list is correctly computed from range parameters."""
        orf_seq = "ATGATGATGATGATGATGATGATG"  # 8 codons
        
        pool = DeletionScanORFPool(
            background_seq=orf_seq,
            deletion_size=2,
            start=1,
            end=6,
            step_size=1
        )
        # range(1, min(6,8)-2+1, 1) = range(1, 5, 1) = [1, 2, 3, 4]
        assert pool.positions == [1, 2, 3, 4], f"Expected [1,2,3,4], got {pool.positions}"
        assert pool.num_internal_states == 4
    
    def test_positions_list_with_step_size(self):
        """Test positions list with non-trivial step_size."""
        orf_seq = "ATGATGATGATGATGATGATGATG"  # 8 codons
        
        pool = DeletionScanORFPool(
            background_seq=orf_seq,
            deletion_size=1,
            start=0,
            end=8,
            step_size=2
        )
        # range(0, 8-1+1, 2) = range(0, 8, 2) = [0, 2, 4, 6]
        assert pool.positions == [0, 2, 4, 6], f"Expected [0,2,4,6], got {pool.positions}"
        assert pool.num_internal_states == 4
    
    def test_step_size_exact_positions(self):
        """Test step_size produces exact correct positions at each state."""
        orf_seq = "ATGCCCGGGTTTAAACCCGGGTTTAAACCC"  # 10 codons
        
        pool = DeletionScanORFPool(
            background_seq=orf_seq,
            deletion_size=1,
            start=1,
            end=9,
            step_size=3,  # Should hit positions 1, 4, 7
            mode='sequential'
        )
        
        expected_positions = [1, 4, 7]
        assert pool.positions == expected_positions, f"Step size positions wrong"
        
        # Verify each state deletes at the correct position
        for i, expected_pos in enumerate(expected_positions):
            pool.set_state(i)
            seq = pool.seq
            codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
            deleted_pos = codons.index('---')
            assert deleted_pos == expected_pos, f"State {i}: expected deletion at {expected_pos}, got {deleted_pos}"


class TestComplexChaining:
    """Test complex chaining with multiple ORF pools."""
    
    def test_deletion_to_insertion_chain(self):
        """Test chaining DeletionScanORFPool -> InsertionScanORFPool."""
        from poolparty import InsertionScanORFPool
        
        base_orf = "AAACCCGGGTTT"  # 4 codons (12 nt)
        
        # Delete 1 codon at position 1 (CCC) - results in 3 codons
        deletion_pool = DeletionScanORFPool(
            background_seq=base_orf,
            deletion_size=1,
            positions=[1],
            mark_changes=False,  # Actually remove
            mode='sequential'
        )
        
        # After deletion: AAA GGG TTT (3 codons, 9 nt)
        # Insert ATG at position 1
        insertion_pool = InsertionScanORFPool(
            background_seq=deletion_pool,
            insert_seq="ATG",
            insert_or_overwrite='insert',
            positions=[1],
            mode='sequential'
        )
        
        # Verify deletion result
        deletion_pool.set_state(0)
        del_result = deletion_pool.seq
        del_codons = [del_result[j:j+3] for j in range(0, len(del_result), 3)]
        assert del_codons == ["AAA", "GGG", "TTT"], f"Deletion result wrong: {del_codons}"
        
        # Verify chain result
        insertion_pool.set_state(0)
        ins_result = insertion_pool.seq
        ins_codons = [ins_result[j:j+3] for j in range(0, len(ins_result), 3)]
        assert ins_codons == ["AAA", "ATG", "GGG", "TTT"], f"Chain result wrong: {ins_codons}"


class TestMultiCodonOperations:
    """Test operations with multiple codons."""
    
    def test_delete_multiple_codons(self):
        """Test deleting 2 codons at a specific position."""
        orf_seq = "AAACCCGGGTTTATGATG"  # 6 codons
        
        pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=2,
            positions=[2],  # Delete codons 2-3 (GGG TTT)
            mode='sequential'
        )
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
        assert codons == ["AAA", "CCC", "---", "---", "ATG", "ATG"], f"Multi-codon delete wrong: {codons}"
    
    def test_delete_3_codons(self):
        """Test deleting 3 codons at position 1."""
        orf_seq = "AAACCCGGGTTTATGATG"  # 6 codons
        
        pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=3,
            positions=[1],  # Delete codons 1-3 (CCC GGG TTT)
            mode='sequential'
        )
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
        assert codons == ["AAA", "---", "---", "---", "ATG", "ATG"], f"3-codon delete wrong: {codons}"


class TestReadingFrameIntegrity:
    """Test that reading frame is maintained."""
    
    def test_marked_mode_maintains_reading_frame(self):
        """Test that marked deletion mode maintains reading frame."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        deletion_size = 2  # 2 codons
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=True, step_size=1, start=0,
            mode='sequential'
        )
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            # Length should still be divisible by 3
            assert len(seq) % 3 == 0
            # Can split into codons
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            assert len(codons) == 6
    
    def test_unmarked_mode_maintains_reading_frame(self):
        """Test that unmarked deletion mode maintains reading frame."""
        orf = "ATGGCCAAA"  # 3 codons
        deletion_size = 1  # 1 codon
        pool = DeletionScanORFPool(
            orf, deletion_size=deletion_size,
            mark_changes=False, step_size=1, start=0,
            mode='sequential'
        )
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            # Length should be divisible by 3
            assert len(seq) % 3 == 0
            # Should be 2 codons total
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            assert len(codons) == 2


class TestLowercaseInput:
    """Test handling of lowercase DNA sequences."""
    
    def test_lowercase_orf_from_pool_parent(self):
        """Test that lowercase from Pool parent is preserved."""
        # When seq comes from a Pool, lowercase is allowed
        parent = Pool(seqs=["atggccaaa"])
        pool = DeletionScanORFPool(
            parent, deletion_size=1,
            mark_changes=True, mode='sequential'
        )
        
        pool.set_state(0)
        # Lowercase should be preserved (comes from parent)
        assert "---" in pool.seq
        assert "gccaaa" in pool.seq
