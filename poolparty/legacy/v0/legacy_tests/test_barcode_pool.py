"""Tests for the BarcodePool class.

Comprehensive tests covering:
- Initialization and parameter validation
- Distance constraints (edit distance, Hamming distance)
- Sequence quality constraints (GC content, homopolymer runs)
- Variable-length barcodes with padding and proportions
- State management and determinism
- Integration with other pools
- Edge cases and error handling
"""

import pytest
from poolparty import BarcodePool, Pool, InsertionScanPool


class TestBarcodePoolInit:
    """Tests for BarcodePool initialization and validation."""
    
    def test_basic_init_illumina_i7_style(self):
        """Test basic initialization mimicking Illumina i7 index barcodes (8bp)."""
        pool = BarcodePool(num_barcodes=96, length=8, seed=42)
        assert len(pool.barcodes) == 96
        assert pool.num_barcodes == 96
        assert pool.lengths == [8]
        assert pool.max_length == 8
    
    def test_init_with_error_correction_distance(self):
        """Test initialization with edit distance=3 for single-error correction."""
        # Edit distance 3 allows detection and correction of 1 substitution error
        pool = BarcodePool(
            num_barcodes=48,
            length=8,
            min_edit_distance=3,
            seed=42
        )
        assert len(pool.barcodes) == 48
        assert pool.min_edit_distance == 3
    
    def test_init_sequencing_optimized(self):
        """Test initialization with constraints optimized for Illumina sequencing."""
        # Typical constraints: balanced GC, no homopolymers, good edit distance
        pool = BarcodePool(
            num_barcodes=24,
            length=10,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=3,
            seed=42
        )
        assert len(pool.barcodes) == 24
        assert pool.gc_range == (0.4, 0.6)
        assert pool.max_homopolymer == 3
    
    def test_init_variable_length_umi_style(self):
        """Test variable length barcodes like UMIs (unique molecular identifiers)."""
        pool = BarcodePool(
            num_barcodes=48,
            length=[8, 10, 12],
            min_edit_distance=3,
            seed=42
        )
        assert len(pool.barcodes) == 48
        assert pool.lengths == [8, 10, 12]
        assert pool.max_length == 12
    
    def test_init_avoiding_illumina_adapters(self):
        """Test initialization avoiding Illumina TruSeq adapter sequences."""
        # Common Illumina adapter sequences to avoid
        truseq_adapters = [
            "AGATCGGAAGAGC",  # TruSeq Read 1
            "AGATCGGAAGAGC",  # TruSeq Read 2 (similar)
        ]
        pool = BarcodePool(
            num_barcodes=24,
            length=8,
            min_edit_distance=3,
            avoid_sequences=truseq_adapters,
            avoid_min_distance=4,
            seed=42
        )
        assert len(pool.avoid_sequences) == 2
        assert pool.avoid_min_distance == 4


class TestBarcodePoolValidation:
    """Tests for parameter validation and error handling."""
    
    def test_invalid_num_barcodes_zero(self):
        """Test that num_barcodes=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_barcodes must be a positive integer"):
            BarcodePool(num_barcodes=0, length=8)
    
    def test_invalid_num_barcodes_negative(self):
        """Test that negative num_barcodes raises ValueError."""
        with pytest.raises(ValueError, match="num_barcodes must be a positive integer"):
            BarcodePool(num_barcodes=-5, length=8)
    
    def test_invalid_length_zero(self):
        """Test that length=0 raises ValueError."""
        with pytest.raises(ValueError, match="All lengths must be positive integers"):
            BarcodePool(num_barcodes=10, length=0)
    
    def test_invalid_length_negative(self):
        """Test that negative length raises ValueError."""
        with pytest.raises(ValueError, match="All lengths must be positive integers"):
            BarcodePool(num_barcodes=10, length=-5)
    
    def test_invalid_length_empty_list(self):
        """Test that empty length list raises ValueError."""
        with pytest.raises(ValueError, match="length must be a non-empty"):
            BarcodePool(num_barcodes=10, length=[])
    
    def test_hamming_with_variable_length(self):
        """Test that min_hamming_distance with variable length raises ValueError."""
        with pytest.raises(ValueError, match="min_hamming_distance cannot be used with variable-length"):
            BarcodePool(
                num_barcodes=10,
                length=[6, 8],
                min_hamming_distance=3
            )
    
    def test_avoid_sequences_without_distance(self):
        """Test that avoid_sequences without avoid_min_distance raises ValueError."""
        with pytest.raises(ValueError, match="avoid_min_distance is required"):
            BarcodePool(
                num_barcodes=10,
                length=8,
                avoid_sequences=["ACGTACGT"]
            )
    
    def test_invalid_gc_range_values(self):
        """Test that gc_range values outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="gc_range values must be in"):
            BarcodePool(num_barcodes=10, length=8, gc_range=(0.3, 1.5))
    
    def test_invalid_gc_range_order(self):
        """Test that gc_range min > max raises ValueError."""
        with pytest.raises(ValueError, match="gc_range min.*cannot exceed max"):
            BarcodePool(num_barcodes=10, length=8, gc_range=(0.7, 0.3))
    
    def test_length_proportions_wrong_length(self):
        """Test that length_proportions with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="length_proportions length.*must match"):
            BarcodePool(
                num_barcodes=10,
                length=[6, 8, 10],
                length_proportions=[0.5, 0.5]
            )
    
    def test_length_proportions_negative_value(self):
        """Test that negative length_proportions raises ValueError."""
        with pytest.raises(ValueError, match="length_proportions values must be positive"):
            BarcodePool(
                num_barcodes=10,
                length=[6, 8],
                length_proportions=[0.5, -0.5]
            )
    
    def test_invalid_padding_side(self):
        """Test that invalid padding_side raises ValueError."""
        with pytest.raises(ValueError, match="padding_side must be 'left' or 'right'"):
            BarcodePool(
                num_barcodes=10,
                length=[6, 8],
                padding_side='center'
            )
    
    def test_constraints_too_strict(self):
        """Test that overly strict constraints raise ValueError."""
        # 4-mer with edit distance 4 and only 100 attempts - impossible
        with pytest.raises(ValueError, match="Could only generate"):
            BarcodePool(
                num_barcodes=100,
                length=4,
                min_edit_distance=4,
                max_attempts=100,
                seed=42
            )


class TestBarcodePoolProperties:
    """Tests for BarcodePool properties."""
    
    def test_num_states_equals_num_barcodes(self):
        """Test that num_states equals num_barcodes."""
        pool = BarcodePool(num_barcodes=25, length=8, seed=42)
        assert pool.num_states == 25
        assert pool.num_internal_states == 25
    
    def test_seq_length_fixed(self):
        """Test seq_length for fixed-length barcodes."""
        pool = BarcodePool(num_barcodes=10, length=12, seed=42)
        assert pool.seq_length == 12
        assert len(pool.seq) == 12
    
    def test_seq_length_variable(self):
        """Test seq_length for variable-length barcodes (equals max)."""
        pool = BarcodePool(num_barcodes=10, length=[6, 8, 10], seed=42)
        assert pool.seq_length == 10  # Max length
        assert len(pool.seq) == 10
    
    def test_mode_random(self):
        """Test that mode can be set to random."""
        pool = BarcodePool(num_barcodes=10, length=8, mode='random', seed=42)
        assert pool.mode == 'random'
    
    def test_mode_sequential(self):
        """Test that mode can be set to sequential."""
        pool = BarcodePool(num_barcodes=10, length=8, mode='sequential', seed=42)
        assert pool.mode == 'sequential'
    
    def test_is_sequential_compatible(self):
        """Test that BarcodePool is sequential-compatible (finite states)."""
        pool = BarcodePool(num_barcodes=50, length=8, seed=42)
        assert pool.is_sequential_compatible()


class TestEditDistanceConstraint:
    """Tests for edit distance constraint."""
    
    def test_edit_distance_basic(self):
        """Test that all barcodes satisfy min_edit_distance."""
        pool = BarcodePool(
            num_barcodes=20,
            length=8,
            min_edit_distance=3,
            seed=42
        )
        
        barcodes = pool.get_all_barcodes(padded=False)
        
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                dist = BarcodePool._edit_distance(barcodes[i], barcodes[j])
                assert dist >= 3, \
                    f"Barcodes {i} and {j} have edit distance {dist} < 3"
    
    def test_edit_distance_variable_length(self):
        """Test edit distance with variable length barcodes."""
        pool = BarcodePool(
            num_barcodes=15,
            length=[6, 8, 10],
            min_edit_distance=3,
            seed=42
        )
        
        barcodes = pool.get_all_barcodes(padded=False)
        
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                dist = BarcodePool._edit_distance(barcodes[i], barcodes[j])
                assert dist >= 3, \
                    f"Barcodes {barcodes[i]} and {barcodes[j]} have edit distance {dist} < 3"
    
    def test_edit_distance_high_value(self):
        """Test high edit distance constraint."""
        pool = BarcodePool(
            num_barcodes=10,
            length=12,
            min_edit_distance=5,
            seed=42
        )
        
        barcodes = pool.get_all_barcodes(padded=False)
        
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                dist = BarcodePool._edit_distance(barcodes[i], barcodes[j])
                assert dist >= 5
    
    def test_edit_distance_calculation_accuracy(self):
        """Test edit distance calculation is correct."""
        # Known edit distances
        assert BarcodePool._edit_distance("ACGT", "ACGT") == 0
        assert BarcodePool._edit_distance("ACGT", "ACGG") == 1
        assert BarcodePool._edit_distance("ACGT", "TGCA") == 4
        assert BarcodePool._edit_distance("ACGT", "ACG") == 1  # Deletion
        assert BarcodePool._edit_distance("ACG", "ACGT") == 1  # Insertion
        assert BarcodePool._edit_distance("ACGT", "AACGT") == 1


class TestHammingDistanceConstraint:
    """Tests for Hamming distance constraint."""
    
    def test_hamming_distance_basic(self):
        """Test that all barcodes satisfy min_hamming_distance."""
        pool = BarcodePool(
            num_barcodes=20,
            length=8,
            min_hamming_distance=3,
            seed=42
        )
        
        barcodes = pool.barcodes
        
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                dist = BarcodePool._hamming_distance(barcodes[i], barcodes[j])
                assert dist >= 3, \
                    f"Barcodes {i} and {j} have Hamming distance {dist} < 3"
    
    def test_hamming_distance_high_value(self):
        """Test high Hamming distance constraint."""
        pool = BarcodePool(
            num_barcodes=10,
            length=12,
            min_hamming_distance=6,
            seed=42
        )
        
        barcodes = pool.barcodes
        
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                dist = BarcodePool._hamming_distance(barcodes[i], barcodes[j])
                assert dist >= 6
    
    def test_hamming_distance_calculation_accuracy(self):
        """Test Hamming distance calculation is correct."""
        assert BarcodePool._hamming_distance("ACGT", "ACGT") == 0
        assert BarcodePool._hamming_distance("ACGT", "ACGG") == 1
        assert BarcodePool._hamming_distance("ACGT", "TGCA") == 4
        assert BarcodePool._hamming_distance("AAAA", "TTTT") == 4
        assert BarcodePool._hamming_distance("ACGT", "ACTT") == 1


class TestGCContentConstraint:
    """Tests for GC content constraint."""
    
    def test_gc_range_basic(self):
        """Test that all barcodes satisfy gc_range."""
        pool = BarcodePool(
            num_barcodes=30,
            length=10,
            gc_range=(0.4, 0.6),
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            gc_count = sum(1 for base in bc if base in 'GC')
            gc_fraction = gc_count / len(bc)
            assert 0.4 <= gc_fraction <= 0.6, \
                f"Barcode {bc} has GC content {gc_fraction} outside [0.4, 0.6]"
    
    def test_gc_range_strict(self):
        """Test strict GC range (exactly 50%)."""
        pool = BarcodePool(
            num_barcodes=20,
            length=10,
            gc_range=(0.5, 0.5),
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            gc_count = sum(1 for base in bc if base in 'GC')
            gc_fraction = gc_count / len(bc)
            assert gc_fraction == 0.5, \
                f"Barcode {bc} has GC content {gc_fraction} != 0.5"
    
    def test_gc_range_low(self):
        """Test low GC range."""
        pool = BarcodePool(
            num_barcodes=20,
            length=10,
            gc_range=(0.2, 0.3),
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            gc_count = sum(1 for base in bc if base in 'GC')
            gc_fraction = gc_count / len(bc)
            assert 0.2 <= gc_fraction <= 0.3
    
    def test_gc_range_high(self):
        """Test high GC range."""
        pool = BarcodePool(
            num_barcodes=20,
            length=10,
            gc_range=(0.7, 0.8),
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            gc_count = sum(1 for base in bc if base in 'GC')
            gc_fraction = gc_count / len(bc)
            assert 0.7 <= gc_fraction <= 0.8


class TestHomopolymerConstraint:
    """Tests for homopolymer run constraint."""
    
    def test_max_homopolymer_3(self):
        """Test that no homopolymer runs exceed 3."""
        pool = BarcodePool(
            num_barcodes=50,
            length=12,
            max_homopolymer=3,
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            max_run = self._get_max_run(bc)
            assert max_run <= 3, \
                f"Barcode {bc} has homopolymer run of {max_run} > 3"
    
    def test_max_homopolymer_2(self):
        """Test that no homopolymer runs exceed 2 (strict)."""
        pool = BarcodePool(
            num_barcodes=30,
            length=10,
            max_homopolymer=2,
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            max_run = self._get_max_run(bc)
            assert max_run <= 2, \
                f"Barcode {bc} has homopolymer run of {max_run} > 2"
    
    def test_max_homopolymer_1(self):
        """Test that no consecutive identical bases (alternating only)."""
        pool = BarcodePool(
            num_barcodes=20,
            length=8,
            max_homopolymer=1,
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            max_run = self._get_max_run(bc)
            assert max_run <= 1, \
                f"Barcode {bc} has homopolymer run of {max_run} > 1"
    
    @staticmethod
    def _get_max_run(seq: str) -> int:
        """Get the maximum homopolymer run length in a sequence."""
        if not seq:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run


class TestAvoidSequencesConstraint:
    """Tests for avoid_sequences constraint."""
    
    def test_avoid_single_sequence(self):
        """Test avoiding a single sequence."""
        adapter = "AGATCGGAAGAG"
        pool = BarcodePool(
            num_barcodes=20,
            length=8,
            min_edit_distance=2,
            avoid_sequences=[adapter],
            avoid_min_distance=4,
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            dist = BarcodePool._edit_distance(bc, adapter)
            assert dist >= 4, \
                f"Barcode {bc} has distance {dist} < 4 from adapter"
    
    def test_avoid_multiple_sequences(self):
        """Test avoiding multiple sequences."""
        adapters = ["AGATCGGAAG", "CTGTCTCTTA", "GTGACTGGAG"]
        pool = BarcodePool(
            num_barcodes=20,
            length=8,
            min_edit_distance=2,
            avoid_sequences=adapters,
            avoid_min_distance=3,
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            for adapter in adapters:
                dist = BarcodePool._edit_distance(bc, adapter)
                assert dist >= 3, \
                    f"Barcode {bc} has distance {dist} < 3 from {adapter}"


class TestVariableLengthBarcodes:
    """Tests for variable-length barcode handling."""
    
    def test_padding_right(self):
        """Test right padding (default)."""
        pool = BarcodePool(
            num_barcodes=9,
            length=[6, 8, 10],
            padding_char='-',
            padding_side='right',
            seed=42
        )
        
        for bc in pool.barcodes:
            # Should end with padding or be max length
            stripped = bc.rstrip('-')
            assert len(bc) == 10
            assert bc == stripped + '-' * (10 - len(stripped))
    
    def test_padding_left(self):
        """Test left padding."""
        pool = BarcodePool(
            num_barcodes=9,
            length=[6, 8, 10],
            padding_char='-',
            padding_side='left',
            seed=42
        )
        
        for bc in pool.barcodes:
            # Should start with padding or be max length
            stripped = bc.lstrip('-')
            assert len(bc) == 10
            assert bc == '-' * (10 - len(stripped)) + stripped
    
    def test_custom_padding_char(self):
        """Test custom padding character."""
        pool = BarcodePool(
            num_barcodes=9,
            length=[6, 8, 10],
            padding_char='N',
            seed=42
        )
        
        for bc in pool.barcodes:
            stripped = bc.rstrip('N')
            assert len(bc) == 10
            assert 'N' not in stripped  # N only as padding
    
    def test_sorted_by_length(self):
        """Test that barcodes are sorted by unpadded length."""
        pool = BarcodePool(
            num_barcodes=12,
            length=[6, 8, 10],
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        lengths = [len(bc) for bc in unpadded]
        
        # Should be sorted (non-decreasing)
        assert lengths == sorted(lengths), \
            f"Barcodes not sorted by length: {lengths}"
    
    def test_get_unpadded_barcode(self):
        """Test get_unpadded_barcode method."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 10],
            padding_char='-',
            seed=42
        )
        
        for i in range(10):
            padded = pool.barcodes[i]
            unpadded = pool.get_unpadded_barcode(i)
            
            assert '-' not in unpadded
            assert unpadded == padded.replace('-', '')
    
    def test_get_all_barcodes_padded(self):
        """Test get_all_barcodes with padded=True."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 10],
            seed=42
        )
        
        padded = pool.get_all_barcodes(padded=True)
        assert all(len(bc) == 10 for bc in padded)
    
    def test_get_all_barcodes_unpadded(self):
        """Test get_all_barcodes with padded=False."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 10],
            padding_char='-',
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        assert all('-' not in bc for bc in unpadded)
        assert all(len(bc) in [6, 10] for bc in unpadded)


class TestLengthProportions:
    """Tests for length_proportions parameter."""
    
    def test_equal_distribution_none(self):
        """Test equal distribution with length_proportions=None."""
        pool = BarcodePool(
            num_barcodes=12,
            length=[6, 8, 10],
            length_proportions=None,
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        counts = {}
        for bc in unpadded:
            L = len(bc)
            counts[L] = counts.get(L, 0) + 1
        
        # Should be equal: 4 of each
        assert counts == {6: 4, 8: 4, 10: 4}
    
    def test_custom_proportions(self):
        """Test custom length proportions."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 8, 10],
            length_proportions=[0.5, 0.3, 0.2],
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        counts = {}
        for bc in unpadded:
            L = len(bc)
            counts[L] = counts.get(L, 0) + 1
        
        # Should be approximately 5, 3, 2
        assert counts == {6: 5, 8: 3, 10: 2}
    
    def test_proportions_normalization(self):
        """Test that proportions are normalized."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 8],
            length_proportions=[2, 3],  # Not summing to 1
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        counts = {}
        for bc in unpadded:
            L = len(bc)
            counts[L] = counts.get(L, 0) + 1
        
        # 2:3 ratio = 4:6 for 10 barcodes
        assert counts == {6: 4, 8: 6}
    
    def test_proportions_with_uneven_split(self):
        """Test proportions when split isn't exact."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 8, 10],
            length_proportions=[0.33, 0.33, 0.34],
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        counts = {}
        for bc in unpadded:
            L = len(bc)
            counts[L] = counts.get(L, 0) + 1
        
        # Should sum to 10
        assert sum(counts.values()) == 10
        # Each should be approximately 3-4
        assert all(2 <= c <= 5 for c in counts.values())
    
    def test_proportions_preserved_in_generate_seqs_random_mode(self):
        """Test that length proportions are approximately preserved in generate_seqs with random mode."""
        pool = BarcodePool(
            num_barcodes=30,
            length=[6, 8, 10],
            length_proportions=[0.5, 0.3, 0.2],  # 15, 9, 6 in stored barcodes
            mode='random',
            seed=42
        )
        
        # Generate many sequences in random mode
        seqs = pool.generate_seqs(num_seqs=300, seed=123)
        
        # Count lengths in generated sequences
        counts = {6: 0, 8: 0, 10: 0}
        for seq in seqs:
            unpadded_len = len(seq.replace('-', ''))
            counts[unpadded_len] += 1
        
        # With 300 samples from 30 barcodes (15 short, 9 medium, 6 long),
        # each barcode should be sampled ~10 times on average.
        # Expected: 6-mers ~150 (50%), 8-mers ~90 (30%), 10-mers ~60 (20%)
        # Allow reasonable tolerance for random sampling
        total = sum(counts.values())
        assert total == 300
        
        # Check proportions are approximately correct (within 15% tolerance)
        assert 0.35 <= counts[6] / total <= 0.65, f"6-mer proportion {counts[6]/total:.2f} outside expected range"
        assert 0.15 <= counts[8] / total <= 0.45, f"8-mer proportion {counts[8]/total:.2f} outside expected range"
        assert 0.05 <= counts[10] / total <= 0.35, f"10-mer proportion {counts[10]/total:.2f} outside expected range"
    
    def test_proportions_exact_in_generate_seqs_sequential_mode(self):
        """Test that length proportions are exactly preserved in generate_seqs with sequential mode."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 8, 10],
            length_proportions=[0.5, 0.3, 0.2],  # 5, 3, 2
            mode='sequential',
            seed=42
        )
        
        # Generate exactly num_barcodes sequences (one complete iteration)
        seqs = pool.generate_seqs(num_seqs=10)
        
        # Count lengths
        counts = {6: 0, 8: 0, 10: 0}
        for seq in seqs:
            unpadded_len = len(seq.replace('-', ''))
            counts[unpadded_len] += 1
        
        # Should exactly match stored proportions
        assert counts == {6: 5, 8: 3, 10: 2}
    
    def test_proportions_in_multiple_complete_iterations(self):
        """Test proportions are preserved across multiple complete iterations."""
        pool = BarcodePool(
            num_barcodes=12,
            length=[6, 8],
            length_proportions=None,  # Equal: 6 each
            mode='sequential',
            seed=42
        )
        
        # Generate 3 complete iterations
        seqs = pool.generate_seqs(num_complete_iterations=3)
        assert len(seqs) == 36
        
        # Count lengths
        counts = {6: 0, 8: 0}
        for seq in seqs:
            unpadded_len = len(seq.replace('-', ''))
            counts[unpadded_len] += 1
        
        # Should be exactly 18 each (6 per iteration × 3 iterations)
        assert counts == {6: 18, 8: 18}


class TestStateManagement:
    """Tests for state management and determinism."""
    
    def test_set_state_deterministic(self):
        """Test that setting state produces deterministic results."""
        pool = BarcodePool(num_barcodes=20, length=8, seed=42)
        
        pool.set_state(5)
        seq1 = pool.seq
        
        pool.set_state(10)
        seq2 = pool.seq
        
        pool.set_state(5)
        seq3 = pool.seq
        
        assert seq1 == seq3, "Same state should produce same sequence"
        assert seq1 != seq2 or seq2 == pool.barcodes[5], "Different states may differ"
    
    def test_sequential_mode_iteration(self):
        """Test sequential mode iterates through all barcodes."""
        pool = BarcodePool(
            num_barcodes=5,
            length=8,
            mode='sequential',
            seed=42
        )
        
        seqs = []
        for state in range(5):
            pool.set_state(state)
            seqs.append(pool.seq)
        
        # Should match stored barcodes in order
        assert seqs == pool.barcodes
    
    def test_sequential_mode_wrapping(self):
        """Test sequential mode wraps around."""
        pool = BarcodePool(
            num_barcodes=5,
            length=8,
            mode='sequential',
            seed=42
        )
        
        pool.set_state(0)
        seq0 = pool.seq
        
        pool.set_state(5)  # Should wrap to state 0
        seq5 = pool.seq
        
        assert seq0 == seq5
    
    def test_generate_seqs_sequential(self):
        """Test generate_seqs in sequential mode."""
        pool = BarcodePool(
            num_barcodes=5,
            length=8,
            mode='sequential',
            seed=42
        )
        
        seqs = pool.generate_seqs(num_seqs=10)
        
        # Should iterate through all 5, then repeat
        assert seqs[:5] == pool.barcodes
        assert seqs[5:10] == pool.barcodes
    
    def test_generate_seqs_complete_iterations(self):
        """Test generate_seqs with num_complete_iterations."""
        pool = BarcodePool(
            num_barcodes=5,
            length=8,
            mode='sequential',
            seed=42
        )
        
        seqs = pool.generate_seqs(num_complete_iterations=3)
        
        assert len(seqs) == 15
        assert seqs[:5] == pool.barcodes
        assert seqs[5:10] == pool.barcodes
        assert seqs[10:15] == pool.barcodes
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same barcodes."""
        pool1 = BarcodePool(num_barcodes=20, length=8, min_edit_distance=2, seed=12345)
        pool2 = BarcodePool(num_barcodes=20, length=8, min_edit_distance=2, seed=12345)
        
        assert pool1.barcodes == pool2.barcodes
    
    def test_different_seeds_different_barcodes(self):
        """Test that different seeds produce different barcodes."""
        pool1 = BarcodePool(num_barcodes=20, length=8, seed=111)
        pool2 = BarcodePool(num_barcodes=20, length=8, seed=222)
        
        assert pool1.barcodes != pool2.barcodes


class TestIntegration:
    """Tests for integration with other Pool classes."""
    
    def test_concatenation_with_string(self):
        """Test concatenating BarcodePool with strings."""
        pool = BarcodePool(num_barcodes=5, length=8, seed=42)
        
        combined = "AAA" + pool + "TTT"
        
        assert isinstance(combined, Pool)
        seq = combined.seq
        assert seq.startswith("AAA")
        assert seq.endswith("TTT")
        assert len(seq) == 3 + 8 + 3
    
    def test_with_insertion_scan_pool(self):
        """Test using BarcodePool as insert_seq in InsertionScanPool."""
        barcodes = BarcodePool(
            num_barcodes=3,
            length=6,
            mode='sequential',
            seed=42
        )
        
        scan = InsertionScanPool(
            background_seq="AAAAAATTTTTT",
            insert_seq=barcodes,
            positions=[0, 6],
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        # 3 barcodes × 2 positions = 6 combinations
        seqs = set()
        for state in range(scan.num_states):
            scan.set_state(state)
            seqs.add(scan.seq)
        
        assert len(seqs) == 6
    
    def test_computation_graph(self):
        """Test that BarcodePool works in computation graph."""
        pool = BarcodePool(num_barcodes=5, length=8, seed=42)
        combined = "PREFIX_" + pool + "_SUFFIX"
        
        result = combined.generate_seqs(num_seqs=10, return_computation_graph=True)
        
        assert "sequences" in result
        assert "graph" in result
        assert len(result["sequences"]) == 10
    
    def test_proportions_preserved_with_string_concatenation(self):
        """Test that length proportions are preserved when concatenated with strings."""
        pool = BarcodePool(
            num_barcodes=12,
            length=[6, 8, 10],
            length_proportions=None,  # Equal: 4 each
            mode='sequential',
            seed=42
        )
        
        combined = "AAA" + pool + "TTT"
        seqs = combined.generate_seqs(num_seqs=12)
        
        # Extract barcode part and count lengths
        counts = {6: 0, 8: 0, 10: 0}
        for seq in seqs:
            # Remove prefix and suffix
            barcode_part = seq[3:-3]
            unpadded_len = len(barcode_part.replace('-', ''))
            counts[unpadded_len] += 1
        
        # Should be exactly 4 of each length
        assert counts == {6: 4, 8: 4, 10: 4}
    
    def test_proportions_preserved_with_insertion_scan_pool_sequential(self):
        """Test length proportions with InsertionScanPool in sequential mode."""
        barcodes = BarcodePool(
            num_barcodes=9,
            length=[4, 6, 8],
            length_proportions=None,  # Equal: 3 each
            mode='sequential',
            seed=42
        )
        
        # Background must be long enough to fit padded barcode (8) at all positions
        scan = InsertionScanPool(
            background_seq="AAAAAAAAAAAAAAAAAAAA",  # 20 bases
            insert_seq=barcodes,
            positions=[0, 10],  # 2 positions (both can fit 8-base barcode)
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        # 9 barcodes × 2 positions = 18 combinations
        seqs = scan.generate_seqs(num_seqs=18)
        assert len(seqs) == 18
        
        # Count barcode lengths across all sequences
        # Each barcode appears at 2 positions, so we should see each barcode twice
        counts = {4: 0, 6: 0, 8: 0}
        
        for seq in seqs:
            # The overwritten part is the barcode
            # Find the barcode by checking which stored barcode appears
            for bc in barcodes.barcodes:
                unpadded = bc.replace('-', '')
                if unpadded in seq:
                    counts[len(unpadded)] += 1
                    break
        
        # Each of the 9 barcodes appears twice (once per position)
        # So total counts should be 6 for each length (3 barcodes × 2 positions)
        assert counts == {4: 6, 6: 6, 8: 6}
    
    def test_proportions_with_insertion_scan_pool_random_mode(self):
        """Test length proportions approximately preserved with InsertionScanPool in random mode."""
        barcodes = BarcodePool(
            num_barcodes=30,
            length=[6, 8, 10],
            length_proportions=[0.5, 0.3, 0.2],  # 15, 9, 6
            mode='random',
            seed=42
        )
        
        # Background must be long enough to fit padded barcode (10) at all positions
        scan = InsertionScanPool(
            background_seq="AAAAAAAAAAAAAAAAAAAAAAAAAA",  # 26 bases
            insert_seq=barcodes,
            positions=[0, 12],  # 2 positions (both can fit 10-base barcode)
            insert_or_overwrite='overwrite',
            mode='random'
        )
        
        # Generate many sequences
        seqs = scan.generate_seqs(num_seqs=300, seed=123)
        
        # Count barcode lengths
        counts = {6: 0, 8: 0, 10: 0}
        for seq in seqs:
            for bc in barcodes.barcodes:
                unpadded = bc.replace('-', '')
                if unpadded in seq:
                    counts[len(unpadded)] += 1
                    break
        
        # Check proportions are approximately correct
        total = sum(counts.values())
        assert total == 300
        
        # Allow reasonable tolerance for random sampling
        assert 0.35 <= counts[6] / total <= 0.65
        assert 0.15 <= counts[8] / total <= 0.45
        assert 0.05 <= counts[10] / total <= 0.35
    
    def test_proportions_in_chained_pools(self):
        """Test proportions preserved when BarcodePool is part of a longer chain."""
        barcodes = BarcodePool(
            num_barcodes=12,
            length=[6, 8],
            length_proportions=None,  # Equal: 6 each
            mode='sequential',
            seed=42
        )
        
        # Create a chain: prefix + barcode + suffix
        chain = "GGG" + barcodes + "CCC"
        
        # Generate 2 complete iterations
        seqs = chain.generate_seqs(num_complete_iterations=2)
        assert len(seqs) == 24
        
        # Count lengths
        counts = {6: 0, 8: 0}
        for seq in seqs:
            barcode_part = seq[3:-3]  # Remove GGG and CCC
            unpadded_len = len(barcode_part.replace('-', ''))
            counts[unpadded_len] += 1
        
        # Should be exactly 12 of each (6 per iteration × 2 iterations)
        assert counts == {6: 12, 8: 12}
    
    def test_variable_length_barcodes_with_insertion_overwrite(self):
        """Test variable-length barcodes work correctly with overwrite insertion."""
        barcodes = BarcodePool(
            num_barcodes=6,
            length=[4, 8],  # Variable lengths
            length_proportions=None,  # 3 each
            padding_char='-',
            mode='sequential',
            seed=42
        )
        
        # Background is 12 bases, insert at position 2
        scan = InsertionScanPool(
            background_seq="AAAAAAAAAAAA",
            insert_seq=barcodes,
            positions=[2],
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        seqs = scan.generate_seqs(num_seqs=6)
        
        # All sequences should be same length (background length)
        assert all(len(seq) == 12 for seq in seqs)
        
        # Count barcode lengths by examining the inserted region
        counts = {4: 0, 8: 0}
        for seq in seqs:
            # The inserted region starts at position 2
            # Check which barcode is present
            for bc in barcodes.barcodes:
                unpadded = bc.replace('-', '')
                # The unpadded barcode should appear starting at position 2
                if seq[2:2+len(unpadded)] == unpadded:
                    counts[len(unpadded)] += 1
                    break
        
        assert counts == {4: 3, 8: 3}


class TestRepr:
    """Tests for string representation."""
    
    def test_repr_basic(self):
        """Test repr for basic BarcodePool."""
        pool = BarcodePool(num_barcodes=10, length=8, seed=42)
        repr_str = repr(pool)
        
        assert "BarcodePool" in repr_str
        assert "n=10" in repr_str
        assert "L=8" in repr_str
    
    def test_repr_with_constraints(self):
        """Test repr shows constraints."""
        pool = BarcodePool(
            num_barcodes=10,
            length=8,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=3,
            seed=42
        )
        repr_str = repr(pool)
        
        assert "edit≥3" in repr_str
        assert "gc=" in repr_str
        assert "homopoly≤3" in repr_str
    
    def test_repr_variable_length(self):
        """Test repr for variable length."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 8, 10],
            seed=42
        )
        repr_str = repr(pool)
        
        assert "[6, 8, 10]" in repr_str


class TestEdgeCases:
    """Tests for edge cases and realistic scenarios."""
    
    def test_single_barcode_control(self):
        """Test generating a single barcode (e.g., for positive control)."""
        pool = BarcodePool(num_barcodes=1, length=8, seed=42)
        
        assert len(pool.barcodes) == 1
        assert pool.num_states == 1
        assert len(pool.seq) == 8
    
    def test_minimum_viable_barcode_6bp(self):
        """Test 6bp barcodes - minimum practical length for multiplexing."""
        # 6bp allows 4^6 = 4096 possible sequences
        pool = BarcodePool(
            num_barcodes=12,
            length=6,
            min_edit_distance=2,
            seed=42
        )
        
        assert all(len(bc) == 6 for bc in pool.barcodes)
        # Verify edit distance constraint
        barcodes = pool.barcodes
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                assert BarcodePool._edit_distance(barcodes[i], barcodes[j]) >= 2
    
    def test_long_umi_barcode_16bp(self):
        """Test 16bp barcodes typical for UMIs (unique molecular identifiers)."""
        pool = BarcodePool(
            num_barcodes=50,
            length=16,
            min_edit_distance=4,
            max_homopolymer=3,
            seed=42
        )
        
        assert all(len(bc) == 16 for bc in pool.barcodes)
    
    def test_large_barcode_set_384_well_plate(self):
        """Test generating barcodes for 384-well plate multiplexing."""
        pool = BarcodePool(
            num_barcodes=384,
            length=10,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            seed=42
        )
        
        assert len(pool.barcodes) == 384
        
        # Spot check distance constraints
        barcodes = pool.get_all_barcodes(padded=False)
        for i in range(0, len(barcodes), 40):
            for j in range(i + 1, min(i + 10, len(barcodes))):
                dist = BarcodePool._edit_distance(barcodes[i], barcodes[j])
                assert dist >= 3
    
    def test_production_quality_barcodes(self):
        """Test production-quality barcode set with all recommended constraints."""
        # Typical production constraints for Illumina sequencing
        pool = BarcodePool(
            num_barcodes=96,
            length=10,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=3,
            avoid_sequences=["AGATCGGAAGAGC"],  # Illumina adapter
            avoid_min_distance=4,
            seed=42
        )
        
        barcodes = pool.get_all_barcodes(padded=False)
        
        # Verify all constraints
        for i, bc in enumerate(barcodes):
            # GC content
            gc = sum(1 for b in bc if b in 'GC') / len(bc)
            assert 0.4 <= gc <= 0.6, f"BC {i}: GC {gc} out of range"
            
            # Homopolymer
            max_run = TestHomopolymerConstraint._get_max_run(bc)
            assert max_run <= 3, f"BC {i}: homopolymer run {max_run} > 3"
            
            # Distance from adapter
            dist = BarcodePool._edit_distance(bc, "AGATCGGAAGAGC")
            assert dist >= 4, f"BC {i}: distance {dist} < 4 from adapter"
            
            # Distance from other barcodes (spot check)
            if i < 20:
                for j in range(i + 1, min(i + 10, len(barcodes))):
                    dist = BarcodePool._edit_distance(bc, barcodes[j])
                    assert dist >= 3
    
    def test_dual_index_compatible_barcodes(self):
        """Test barcodes designed for dual-indexing (i5 + i7 style)."""
        # Create two barcode sets that could be used together
        pool_i7 = BarcodePool(
            num_barcodes=12,
            length=8,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            seed=100
        )
        pool_i5 = BarcodePool(
            num_barcodes=8,
            length=8,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            seed=200
        )
        
        # Both should generate valid barcode sets
        assert len(pool_i7.barcodes) == 12
        assert len(pool_i5.barcodes) == 8
        # Combined gives 12 × 8 = 96 unique combinations
    
    def test_balanced_gc_only(self):
        """Test barcodes with only balanced GC constraint (common requirement)."""
        pool = BarcodePool(
            num_barcodes=48,
            length=8,
            gc_range=(0.5, 0.5),  # Exactly 50% GC
            seed=42
        )
        
        for bc in pool.barcodes:
            gc = sum(1 for b in bc if b in 'GC') / len(bc)
            assert gc == 0.5
    
    def test_single_length_in_list_format(self):
        """Test with single length specified as list (API compatibility)."""
        pool = BarcodePool(
            num_barcodes=24,
            length=[8],  # Single length in list format
            min_edit_distance=3,
            seed=42
        )
        
        assert pool.lengths == [8]
        assert pool.max_length == 8


class TestDNAValidity:
    """Tests to verify all barcodes are valid DNA."""
    
    def test_only_acgt(self):
        """Test that barcodes only contain ACGT."""
        pool = BarcodePool(num_barcodes=50, length=12, seed=42)
        
        for bc in pool.get_all_barcodes(padded=False):
            assert all(base in 'ACGT' for base in bc), \
                f"Barcode {bc} contains non-ACGT characters"
    
    def test_uppercase(self):
        """Test that barcodes are uppercase."""
        pool = BarcodePool(num_barcodes=20, length=8, seed=42)
        
        for bc in pool.barcodes:
            unpadded = bc.replace('-', '')
            assert unpadded.isupper(), \
                f"Barcode {bc} is not uppercase"
    
    def test_padding_char_not_in_sequence(self):
        """Test that padding char is not in actual sequence."""
        pool = BarcodePool(
            num_barcodes=10,
            length=[6, 10],
            padding_char='X',
            seed=42
        )
        
        for bc in pool.get_all_barcodes(padded=False):
            assert 'X' not in bc, \
                f"Padding char 'X' found in unpadded barcode {bc}"


class TestCombinedConstraintsFeasibility:
    """Tests for realistic constraint combinations used in practice."""
    
    def test_high_stringency_small_set(self):
        """Test high-stringency barcodes for small, error-tolerant set."""
        # High edit distance (4) for maximum error tolerance
        # Useful when multiplexing few samples with high accuracy needs
        pool = BarcodePool(
            num_barcodes=8,
            length=8,
            min_edit_distance=4,
            gc_range=(0.4, 0.6),
            max_homopolymer=2,
            seed=42
        )
        
        assert len(pool.barcodes) == 8
        
        # Verify high stringency
        barcodes = pool.barcodes
        for i in range(len(barcodes)):
            for j in range(i + 1, len(barcodes)):
                assert BarcodePool._edit_distance(barcodes[i], barcodes[j]) >= 4
    
    def test_nanopore_optimized_barcodes(self):
        """Test barcodes optimized for Nanopore sequencing (strict homopolymer)."""
        # Nanopore has higher error rate in homopolymer regions
        pool = BarcodePool(
            num_barcodes=24,
            length=12,
            min_edit_distance=4,
            gc_range=(0.4, 0.6),
            max_homopolymer=2,  # Strict homopolymer limit for Nanopore
            seed=42
        )
        
        for bc in pool.barcodes:
            gc = sum(1 for b in bc if b in 'GC') / len(bc)
            assert 0.4 <= gc <= 0.6
            
            max_run = TestHomopolymerConstraint._get_max_run(bc)
            assert max_run <= 2
    
    def test_variable_length_for_size_selection(self):
        """Test variable length barcodes for size-selection multiplexing."""
        # Different barcode lengths can be used to identify samples by size
        pool = BarcodePool(
            num_barcodes=24,
            length=[8, 10, 12],
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=3,
            length_proportions=[0.5, 0.3, 0.2],  # More short barcodes
            seed=42
        )
        
        unpadded = pool.get_all_barcodes(padded=False)
        
        # Verify length distribution
        length_counts = {}
        for bc in unpadded:
            L = len(bc)
            length_counts[L] = length_counts.get(L, 0) + 1
        
        # Should have 12, 7, 5 approximately (50%, 30%, 20% of 24)
        assert length_counts[8] >= 10  # ~50%
        assert length_counts[10] >= 5  # ~30%
        assert length_counts[12] >= 3  # ~20%
        
        # Verify all constraints on unpadded barcodes
        for bc in unpadded:
            gc = sum(1 for b in bc if b in 'GC') / len(bc)
            assert 0.4 <= gc <= 0.6
            
            max_run = TestHomopolymerConstraint._get_max_run(bc)
            assert max_run <= 3

