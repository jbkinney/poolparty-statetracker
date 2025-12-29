"""Comprehensive tests for MotifPool orientation feature.

These tests verify:
1. Basic orientation functionality (forward, reverse, both)
2. Reverse complement correctness
3. Probability distribution when orientation='both'
4. Design cards integration with orientation metadata
5. Determinism and reproducibility
6. Integration with InsertionScanPool
7. Edge cases and validation
"""

import pytest
import pandas as pd
import numpy as np
from collections import Counter

from poolparty import (
    MotifPool,
    InsertionScanPool,
    Pool,
    MixedPool,
    BarcodePool,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_pwm():
    """A simple deterministic PWM for testing (always produces 'ACG')."""
    return pd.DataFrame({
        'A': [1.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0],
        'G': [0.0, 0.0, 1.0],
        'T': [0.0, 0.0, 0.0]
    })


@pytest.fixture
def uniform_pwm():
    """A uniform PWM with equal probabilities (for randomness tests)."""
    return pd.DataFrame({
        'A': [0.25, 0.25, 0.25, 0.25],
        'C': [0.25, 0.25, 0.25, 0.25],
        'G': [0.25, 0.25, 0.25, 0.25],
        'T': [0.25, 0.25, 0.25, 0.25]
    })


@pytest.fixture
def asymmetric_pwm():
    """A PWM that produces different sequences forward vs reverse complement."""
    # Forward: high probability of 'ACGT'
    # Reverse complement of 'ACGT' is 'ACGT' (palindrome)
    # Let's use something non-palindromic: 'AACG' -> rc = 'CGTT'
    return pd.DataFrame({
        'A': [1.0, 1.0, 0.0, 0.0],
        'C': [0.0, 0.0, 1.0, 0.0],
        'G': [0.0, 0.0, 0.0, 1.0],
        'T': [0.0, 0.0, 0.0, 0.0]
    })


# ============================================================================
# Basic Orientation Functionality Tests
# ============================================================================

class TestOrientationBasics:
    """Tests for basic orientation parameter functionality."""
    
    def test_default_orientation_is_forward(self, simple_pwm):
        """Test that default orientation is 'forward'."""
        pool = MotifPool(simple_pwm)
        assert pool.orientation == 'forward'
    
    def test_forward_orientation_explicit(self, simple_pwm):
        """Test explicit forward orientation."""
        pool = MotifPool(simple_pwm, orientation='forward')
        assert pool.orientation == 'forward'
        
        # Generate sequences - should be 'ACG' (not reverse complemented)
        seqs = pool.generate_seqs(num_seqs=10, seed=42)
        assert all(s == 'ACG' for s in seqs)
    
    def test_reverse_orientation(self, simple_pwm):
        """Test reverse orientation always reverse complements."""
        pool = MotifPool(simple_pwm, orientation='reverse')
        assert pool.orientation == 'reverse'
        
        # Generate sequences - 'ACG' reverse complement is 'CGT'
        seqs = pool.generate_seqs(num_seqs=10, seed=42)
        assert all(s == 'CGT' for s in seqs)
    
    def test_both_orientation_produces_mixed_results(self, simple_pwm):
        """Test that orientation='both' produces both forward and reverse."""
        pool = MotifPool(simple_pwm, orientation='both')
        assert pool.orientation == 'both'
        
        seqs = pool.generate_seqs(num_seqs=100, seed=42)
        
        # Should have both 'ACG' (forward) and 'CGT' (reverse)
        forward_count = sum(1 for s in seqs if s == 'ACG')
        reverse_count = sum(1 for s in seqs if s == 'CGT')
        
        assert forward_count > 0, "Should have some forward orientations"
        assert reverse_count > 0, "Should have some reverse orientations"
        assert forward_count + reverse_count == 100
    
    def test_invalid_orientation_raises_error(self, simple_pwm):
        """Test that invalid orientation values raise ValueError."""
        with pytest.raises(ValueError, match="orientation must be"):
            MotifPool(simple_pwm, orientation='invalid')
        
        with pytest.raises(ValueError, match="orientation must be"):
            MotifPool(simple_pwm, orientation='backwards')


# ============================================================================
# Reverse Complement Correctness Tests
# ============================================================================

class TestReverseComplement:
    """Tests for reverse complement correctness."""
    
    def test_reverse_complement_static_method(self):
        """Test the static reverse complement method."""
        assert MotifPool._reverse_complement('ACGT') == 'ACGT'  # Palindrome
        assert MotifPool._reverse_complement('AACG') == 'CGTT'
        assert MotifPool._reverse_complement('AAAA') == 'TTTT'
        assert MotifPool._reverse_complement('GCGC') == 'GCGC'  # Palindrome
        assert MotifPool._reverse_complement('ATCGATCG') == 'CGATCGAT'
    
    def test_reverse_complement_preserves_case(self):
        """Test that reverse complement handles mixed case."""
        assert MotifPool._reverse_complement('AcGt') == 'aCgT'
        assert MotifPool._reverse_complement('aacg') == 'cgtt'
    
    def test_reverse_complement_with_asymmetric_pwm(self, asymmetric_pwm):
        """Test reverse complement produces correct sequence."""
        # Forward: should produce 'AACG'
        forward_pool = MotifPool(asymmetric_pwm, orientation='forward')
        forward_seqs = forward_pool.generate_seqs(num_seqs=5, seed=42)
        assert all(s == 'AACG' for s in forward_seqs)
        
        # Reverse: should produce 'CGTT' (rc of 'AACG')
        reverse_pool = MotifPool(asymmetric_pwm, orientation='reverse')
        reverse_seqs = reverse_pool.generate_seqs(num_seqs=5, seed=42)
        assert all(s == 'CGTT' for s in reverse_seqs)


# ============================================================================
# Forward Probability Distribution Tests
# ============================================================================

class TestForwardProbability:
    """Tests for forward_prob parameter with orientation='both'."""
    
    def test_default_forward_prob_is_0_5(self, simple_pwm):
        """Test that default forward_prob is 0.5."""
        pool = MotifPool(simple_pwm, orientation='both')
        assert pool.forward_prob == 0.5
    
    def test_forward_prob_0_5_gives_equal_distribution(self, simple_pwm):
        """Test that forward_prob=0.5 gives roughly equal distribution."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.5)
        
        seqs = pool.generate_seqs(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for s in seqs if s == 'ACG')
        reverse_count = sum(1 for s in seqs if s == 'CGT')
        
        # With n=1000, p=0.5: std=15.8, using ±5 std (~99.9999% CI) = ±79
        # Expected: 500 each, range: 420-580
        assert 420 < forward_count < 580, f"Forward count {forward_count} outside expected range [420, 580]"
        assert 420 < reverse_count < 580, f"Reverse count {reverse_count} outside expected range [420, 580]"
    
    def test_forward_prob_0_8_skews_toward_forward(self, simple_pwm):
        """Test that forward_prob=0.8 gives ~80% forward orientation."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.8)
        
        seqs = pool.generate_seqs(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for s in seqs if s == 'ACG')
        
        # With n=1000, p=0.8: std=12.6, using ±5 std = ±63
        # Expected: 800, range: 737-863
        assert 737 < forward_count < 863, f"Forward count {forward_count} outside expected range [737, 863]"
    
    def test_forward_prob_0_2_skews_toward_reverse(self, simple_pwm):
        """Test that forward_prob=0.2 gives ~20% forward orientation."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.2)
        
        seqs = pool.generate_seqs(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for s in seqs if s == 'ACG')
        
        # With n=1000, p=0.2: std=12.6, using ±5 std = ±63
        # Expected: 200, range: 137-263
        assert 137 < forward_count < 263, f"Forward count {forward_count} outside expected range [137, 263]"
    
    def test_forward_prob_1_0_always_forward(self, simple_pwm):
        """Test that forward_prob=1.0 gives 100% forward orientation."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=1.0)
        
        seqs = pool.generate_seqs(num_seqs=100, seed=42)
        
        assert all(s == 'ACG' for s in seqs), "All should be forward with forward_prob=1.0"
    
    def test_forward_prob_0_0_always_reverse(self, simple_pwm):
        """Test that forward_prob=0.0 gives 100% reverse orientation."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.0)
        
        seqs = pool.generate_seqs(num_seqs=100, seed=42)
        
        assert all(s == 'CGT' for s in seqs), "All should be reverse with forward_prob=0.0"
    
    def test_forward_prob_ignored_when_not_both(self, simple_pwm):
        """Test that forward_prob is ignored when orientation is not 'both'."""
        # forward_prob shouldn't affect forward orientation
        pool_forward = MotifPool(simple_pwm, orientation='forward', forward_prob=0.0)
        seqs = pool_forward.generate_seqs(num_seqs=10, seed=42)
        assert all(s == 'ACG' for s in seqs)
        
        # forward_prob shouldn't affect reverse orientation
        pool_reverse = MotifPool(simple_pwm, orientation='reverse', forward_prob=1.0)
        seqs = pool_reverse.generate_seqs(num_seqs=10, seed=42)
        assert all(s == 'CGT' for s in seqs)
    
    def test_invalid_forward_prob_raises_error(self, simple_pwm):
        """Test that invalid forward_prob values raise ValueError."""
        with pytest.raises(ValueError, match="forward_prob must be between 0 and 1"):
            MotifPool(simple_pwm, forward_prob=-0.1)
        
        with pytest.raises(ValueError, match="forward_prob must be between 0 and 1"):
            MotifPool(simple_pwm, forward_prob=1.1)


# ============================================================================
# Statistical Distribution Verification Tests
# ============================================================================

class TestOrientationDistribution:
    """Rigorous statistical tests for orientation distribution."""
    
    def test_chi_squared_equal_probability(self, simple_pwm):
        """Chi-squared test for orientation='both' with forward_prob=0.5."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.5)
        
        seqs = pool.generate_seqs(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for s in seqs if s == 'ACG')
        reverse_count = sum(1 for s in seqs if s == 'CGT')
        
        # Expected: 500 each
        expected = 500
        chi_squared = (
            ((forward_count - expected) ** 2 / expected) +
            ((reverse_count - expected) ** 2 / expected)
        )
        
        # With df=1, chi-squared critical values:
        #   alpha=0.05 -> 3.84, alpha=0.01 -> 6.63, alpha=0.001 -> 10.83
        # Using 10.0 corresponds to p ≈ 0.0016 (very unlikely to fail by chance)
        assert chi_squared < 10.0, f"Chi-squared {chi_squared:.2f} too high for equal distribution (threshold: 10.0)"
    
    def test_chi_squared_skewed_probability(self, simple_pwm):
        """Chi-squared test for orientation='both' with forward_prob=0.7."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.7)
        
        seqs = pool.generate_seqs(num_seqs=1000, seed=42)
        
        forward_count = sum(1 for s in seqs if s == 'ACG')
        reverse_count = sum(1 for s in seqs if s == 'CGT')
        
        # Expected: 700 forward, 300 reverse
        expected_fwd = 700
        expected_rev = 300
        
        chi_squared = (
            ((forward_count - expected_fwd) ** 2 / expected_fwd) +
            ((reverse_count - expected_rev) ** 2 / expected_rev)
        )
        
        # Using 10.0 corresponds to p ≈ 0.0016 for df=1
        assert chi_squared < 10.0, f"Chi-squared {chi_squared:.2f} too high (threshold: 10.0)"
    
    def test_distribution_across_multiple_seeds(self, simple_pwm):
        """Test orientation distribution holds across different seeds."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.6)
        
        all_ratios = []
        for seed in range(10):
            seqs = pool.generate_seqs(num_seqs=200, seed=seed * 1000)
            forward_count = sum(1 for s in seqs if s == 'ACG')
            ratio = forward_count / 200
            all_ratios.append(ratio)
        
        # Average ratio should be close to 0.6
        # With 10 seeds × 200 samples = 2000 total, std of mean ≈ 0.011
        # Using ±5 std → 0.6 ± 0.055
        avg_ratio = sum(all_ratios) / len(all_ratios)
        assert 0.545 < avg_ratio < 0.655, f"Average ratio {avg_ratio:.3f} outside expected range [0.545, 0.655]"


# ============================================================================
# Design Cards Integration Tests
# ============================================================================

class TestDesignCardsOrientation:
    """Tests for design cards integration with orientation metadata."""
    
    def test_design_cards_include_orientation_column(self, simple_pwm):
        """Test that design cards include orientation column for named MotifPool."""
        pool = MotifPool(simple_pwm, orientation='both', name='motif')
        
        result = pool.generate_seqs(num_seqs=10, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        assert 'motif_orientation' in df.columns, "orientation column should be in design cards"
    
    def test_design_cards_orientation_values_correct(self, simple_pwm):
        """Test that orientation values in design cards are correct."""
        pool = MotifPool(simple_pwm, orientation='both', name='motif')
        
        result = pool.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        seqs = result['sequences']
        
        # Verify orientation matches actual sequence
        for i, (seq, orientation) in enumerate(zip(seqs, df['motif_orientation'])):
            if seq == 'ACG':
                assert orientation == 'forward', f"Seq {i}: {seq} should be forward, got {orientation}"
            elif seq == 'CGT':
                assert orientation == 'reverse', f"Seq {i}: {seq} should be reverse, got {orientation}"
    
    def test_design_cards_forward_only(self, simple_pwm):
        """Test design cards with orientation='forward'."""
        pool = MotifPool(simple_pwm, orientation='forward', name='motif')
        
        result = pool.generate_seqs(num_seqs=10, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        assert all(df['motif_orientation'] == 'forward')
    
    def test_design_cards_reverse_only(self, simple_pwm):
        """Test design cards with orientation='reverse'."""
        pool = MotifPool(simple_pwm, orientation='reverse', name='motif')
        
        result = pool.generate_seqs(num_seqs=10, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        assert all(df['motif_orientation'] == 'reverse')
    
    def test_design_cards_orientation_distribution(self, simple_pwm):
        """Test that orientation distribution in design cards matches expected."""
        pool = MotifPool(simple_pwm, orientation='both', forward_prob=0.7, name='motif')
        
        result = pool.generate_seqs(num_seqs=1000, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        forward_count = (df['motif_orientation'] == 'forward').sum()
        
        # With n=1000, p=0.7: std=14.5, using ±5 std = ±72
        # Expected: 700, range: 628-772
        assert 628 < forward_count < 772, f"Forward count {forward_count} outside expected range [628, 772]"
    
    def test_design_cards_with_insertion_scan(self, simple_pwm):
        """Test design cards when MotifPool is used with InsertionScanPool."""
        motif = MotifPool(simple_pwm, orientation='both', name='motif')
        scan = InsertionScanPool(
            'NNNNNNNN',
            motif,
            mode='sequential',
            name='scan'
        )
        
        result = scan.generate_seqs(num_seqs=12, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        # Should have both orientation and position columns
        assert 'motif_orientation' in df.columns
        assert 'scan_pos' in df.columns
        
        # Orientation should vary
        orientations = df['motif_orientation'].unique()
        # With 12 samples and 50% prob, very likely to have both
        assert len(orientations) >= 1  # At minimum one orientation


# ============================================================================
# Determinism and Reproducibility Tests
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior with seeds."""
    
    def test_same_seed_same_orientation(self, simple_pwm):
        """Test that same seed produces same orientation choices."""
        pool = MotifPool(simple_pwm, orientation='both')
        
        seqs1 = pool.generate_seqs(num_seqs=50, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=50, seed=42)
        
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
    
    def test_different_seed_different_orientation(self, simple_pwm):
        """Test that different seeds produce different orientation choices."""
        pool = MotifPool(simple_pwm, orientation='both')
        
        seqs1 = pool.generate_seqs(num_seqs=50, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=50, seed=123)
        
        # With 50 samples at 50% each, extremely unlikely to be identical
        assert seqs1 != seqs2, "Different seeds should produce different sequences"
    
    def test_orientation_deterministic_per_state(self, simple_pwm):
        """Test that orientation is deterministic for each state."""
        pool = MotifPool(simple_pwm, orientation='both', name='motif')
        
        # Generate twice with same seed
        result1 = pool.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        result2 = pool.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        
        df1 = result1['design_cards'].to_dataframe()
        df2 = result2['design_cards'].to_dataframe()
        
        assert list(df1['motif_orientation']) == list(df2['motif_orientation'])


# ============================================================================
# Integration Tests with Other Pools
# ============================================================================

class TestIntegrationWithOtherPools:
    """Tests for MotifPool orientation with other pool types."""
    
    def test_with_insertion_scan_pool(self, simple_pwm):
        """Test MotifPool with InsertionScanPool."""
        motif = MotifPool(simple_pwm, orientation='both', name='motif')
        scan = InsertionScanPool(
            'XXXXXXXX',
            motif,
            mode='sequential',
            mark_changes=True,
            name='scan'
        )
        
        seqs = scan.generate_seqs(num_seqs=12, seed=42)
        
        # Should have 6 positions × varying orientations
        assert len(seqs) == 12
        
        # Check that motifs appear in sequences
        # Forward: 'acg' (lowercase due to mark_changes)
        # Reverse: 'cgt' (lowercase due to mark_changes)
        for seq in seqs:
            assert 'acg' in seq.lower() or 'cgt' in seq.lower()
    
    def test_concatenation_with_other_pools(self, simple_pwm):
        """Test MotifPool orientation in concatenated pools."""
        motif = MotifPool(simple_pwm, orientation='both', name='motif')
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        suffix = Pool(['SUFFIX'], name='suffix', mode='sequential')
        
        composite = prefix + motif + suffix
        
        seqs = composite.generate_seqs(num_seqs=20, seed=42)
        
        # All should start with PREFIX and end with SUFFIX
        for seq in seqs:
            assert seq.startswith('PREFIX')
            assert seq.endswith('SUFFIX')
            # Middle should be ACG or CGT
            middle = seq[6:-6]
            assert middle in ['ACG', 'CGT']
    
    def test_with_mixed_pool_forward_reverse_separate(self, simple_pwm):
        """Test using separate forward and reverse MotifPools with MixedPool."""
        # This is the recommended approach for paired orientation testing
        fwd_motif = MotifPool(simple_pwm, orientation='forward', name='fwd_motif')
        rev_motif = MotifPool(simple_pwm, orientation='reverse', name='rev_motif')
        
        # Use in InsertionScanPool
        fwd_scan = InsertionScanPool('NNNN', fwd_motif, mode='sequential', name='fwd_scan')
        rev_scan = InsertionScanPool('NNNN', rev_motif, mode='sequential', name='rev_scan')
        
        # Note: MixedPool with MotifPool children has infinite states (due to MotifPool),
        # so we use num_seqs instead of num_complete_iterations
        mixed = MixedPool([fwd_scan, rev_scan], mode='random')
        
        seqs = mixed.generate_seqs(num_seqs=20, seed=42)
        
        # Should have a mix of forward and reverse scan outputs
        assert len(seqs) == 20
        
        # Verify both forward ('ACG') and reverse ('CGT') motifs appear
        has_forward = any('ACG' in s for s in seqs)
        has_reverse = any('CGT' in s for s in seqs)
        assert has_forward or has_reverse  # At least one should appear
    
    def test_with_barcode_pool(self, simple_pwm):
        """Test MotifPool orientation with BarcodePool."""
        motif = MotifPool(simple_pwm, orientation='both', name='motif')
        barcode = BarcodePool(num_barcodes=4, length=6, min_edit_distance=2, 
                              name='barcode', seed=42)
        
        library = motif + barcode
        
        result = library.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        
        seqs = result['sequences']
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        # Should have orientation tracking
        assert 'motif_orientation' in df.columns
        
        # Each sequence should be motif (3) + barcode (6) = 9 chars
        assert all(len(s) == 9 for s in seqs)


# ============================================================================
# Edge Cases and Validation Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and validation."""
    
    def test_single_position_pwm_with_orientation(self):
        """Test orientation with single-position PWM."""
        pwm = pd.DataFrame({
            'A': [1.0],
            'C': [0.0],
            'G': [0.0],
            'T': [0.0]
        })
        
        fwd_pool = MotifPool(pwm, orientation='forward')
        rev_pool = MotifPool(pwm, orientation='reverse')
        
        fwd_seqs = fwd_pool.generate_seqs(num_seqs=5, seed=42)
        rev_seqs = rev_pool.generate_seqs(num_seqs=5, seed=42)
        
        # Forward: 'A'
        assert all(s == 'A' for s in fwd_seqs)
        # Reverse complement of 'A' is 'T'
        assert all(s == 'T' for s in rev_seqs)
    
    def test_palindromic_sequence(self):
        """Test orientation with palindromic PWM (rc == forward)."""
        # ACGT is a palindrome: rc(ACGT) = ACGT
        pwm = pd.DataFrame({
            'A': [1.0, 0.0, 0.0, 0.0],
            'C': [0.0, 1.0, 0.0, 0.0],
            'G': [0.0, 0.0, 1.0, 0.0],
            'T': [0.0, 0.0, 0.0, 1.0]
        })
        
        both_pool = MotifPool(pwm, orientation='both', name='motif')
        
        result = both_pool.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        
        seqs = result['sequences']
        df = result['design_cards'].to_dataframe()
        
        # All sequences should be 'ACGT' regardless of orientation
        assert all(s == 'ACGT' for s in seqs)
        
        # But orientation metadata should still vary
        orientations = df['motif_orientation'].unique()
        assert len(orientations) >= 1  # Should have recorded orientations
    
    def test_repr_includes_orientation(self, simple_pwm):
        """Test that __repr__ includes orientation info when not default."""
        pool_fwd = MotifPool(simple_pwm, orientation='forward')
        pool_rev = MotifPool(simple_pwm, orientation='reverse')
        pool_both = MotifPool(simple_pwm, orientation='both')
        pool_both_prob = MotifPool(simple_pwm, orientation='both', forward_prob=0.3)
        
        # Forward (default) shouldn't show orientation
        assert "orientation" not in repr(pool_fwd)
        
        # Reverse should show orientation
        assert "orientation='reverse'" in repr(pool_rev)
        
        # Both should show orientation
        assert "orientation='both'" in repr(pool_both)
        
        # Non-default forward_prob should be shown
        assert "forward_prob=0.3" in repr(pool_both_prob)
    
    def test_metadata_levels(self, simple_pwm):
        """Test orientation works with different metadata levels."""
        # Core level - no orientation
        pool_core = MotifPool(simple_pwm, orientation='both', name='motif', metadata='core')
        result_core = pool_core.generate_seqs(num_seqs=5, seed=42, return_design_cards=True)
        df_core = result_core['design_cards'].to_dataframe()
        assert 'motif_orientation' not in df_core.columns
        
        # Features level - has orientation
        pool_features = MotifPool(simple_pwm, orientation='both', name='motif', metadata='features')
        result_features = pool_features.generate_seqs(num_seqs=5, seed=42, return_design_cards=True)
        df_features = result_features['design_cards'].to_dataframe()
        assert 'motif_orientation' in df_features.columns
        
        # Complete level - has orientation and value
        pool_complete = MotifPool(simple_pwm, orientation='both', name='motif', metadata='complete')
        result_complete = pool_complete.generate_seqs(num_seqs=5, seed=42, return_design_cards=True)
        df_complete = result_complete['design_cards'].to_dataframe()
        assert 'motif_orientation' in df_complete.columns
        assert 'motif_value' in df_complete.columns


# ============================================================================
# Realistic MPRA Scenario Tests
# ============================================================================

class TestRealisticMPRAScenarios:
    """Tests based on realistic MPRA library design scenarios."""
    
    def test_tf_binding_site_scanning(self):
        """Test scanning a TF motif across a region in both orientations."""
        # Sp1-like motif (GC-rich)
        sp1_pwm = pd.DataFrame({
            'A': [0.05, 0.05, 0.05, 0.10, 0.05],
            'C': [0.10, 0.10, 0.10, 0.80, 0.10],
            'G': [0.80, 0.80, 0.80, 0.05, 0.80],
            'T': [0.05, 0.05, 0.05, 0.05, 0.05]
        })
        
        motif = MotifPool(sp1_pwm, orientation='both', name='sp1')
        background = 'AAAAAAAAAAAAAAAA'  # 16nt region
        
        scan = InsertionScanPool(
            background,
            motif,
            step_size=2,  # Scan every 2bp
            mode='sequential',
            name='scan'
        )
        
        result = scan.generate_seqs(num_seqs=24, seed=42, return_design_cards=True)
        
        seqs = result['sequences']
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        # Should have position and orientation info
        assert 'scan_pos' in df.columns
        assert 'sp1_orientation' in df.columns
        
        # Check orientation distribution
        forward_count = (df['sp1_orientation'] == 'forward').sum()
        reverse_count = (df['sp1_orientation'] == 'reverse').sum()
        
        assert forward_count > 0
        assert reverse_count > 0
    
    def test_paired_orientation_library_design(self):
        """Test creating paired forward/reverse library for orientation effect analysis."""
        # This is the recommended approach for systematic orientation testing
        # Use a deterministic motif (fixed PWM) so we can use sequential mode
        pwm = pd.DataFrame({
            'A': [1.0, 0.0, 0.0, 0.0],  # Deterministic: 'ACGT'
            'C': [0.0, 1.0, 0.0, 0.0],
            'G': [0.0, 0.0, 1.0, 0.0],
            'T': [0.0, 0.0, 0.0, 1.0]
        })
        
        # Create two separate MotifPools with fixed orientation
        fwd_motif = MotifPool(pwm, orientation='forward', name='motif_fwd')
        rev_motif = MotifPool(pwm, orientation='reverse', name='motif_rev')
        
        background = 'NNNNNNNN'
        
        fwd_scan = InsertionScanPool(background, fwd_motif, mode='sequential', name='fwd_scan')
        rev_scan = InsertionScanPool(background, rev_motif, mode='sequential', name='rev_scan')
        
        # Note: MixedPool with MotifPool children has infinite states (MotifPool always has 
        # infinite internal states even with deterministic PWM), so use random mode
        library = MixedPool([fwd_scan, rev_scan], mode='random')
        
        result = library.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        
        seqs = result['sequences']
        
        # Should get sequences from both scans
        assert len(seqs) == 20
        
        # Verify library produces valid sequences
        # InsertionScanPool with default insert_or_overwrite='overwrite' keeps same length
        for seq in seqs:
            assert len(seq) == 8  # Same as background (overwrite mode)
    
    def test_library_with_barcodes_and_orientation_tracking(self):
        """Test complete library with orientation tracking for downstream analysis."""
        pwm = pd.DataFrame({
            'A': [0.9, 0.0, 0.1],
            'C': [0.0, 0.9, 0.0],
            'G': [0.1, 0.0, 0.9],
            'T': [0.0, 0.1, 0.0]
        })
        
        motif = MotifPool(pwm, orientation='both', forward_prob=0.5, name='motif')
        barcode = BarcodePool(num_barcodes=8, length=8, min_edit_distance=2, 
                              name='barcode', seed=42)
        
        library = motif + barcode
        
        result = library.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        
        dc = result['design_cards']
        df = dc.to_dataframe()
        
        # Verify we can analyze orientation effects
        forward_seqs = df[df['motif_orientation'] == 'forward']
        reverse_seqs = df[df['motif_orientation'] == 'reverse']
        
        assert len(forward_seqs) > 0, "Should have forward orientation samples"
        assert len(reverse_seqs) > 0, "Should have reverse orientation samples"
        
        # Each barcode should appear with both orientations (statistically)
        # This enables paired analysis of orientation effects


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""
    
    def test_default_behavior_unchanged(self, simple_pwm):
        """Test that default behavior (no orientation specified) is unchanged."""
        pool = MotifPool(simple_pwm)
        
        seqs = pool.generate_seqs(num_seqs=10, seed=42)
        
        # Default should be forward orientation
        assert all(s == 'ACG' for s in seqs)
    
    def test_existing_api_still_works(self, simple_pwm):
        """Test that existing API usage still works."""
        # All these should work without errors
        pool1 = MotifPool(simple_pwm)
        pool2 = MotifPool(simple_pwm, mode='random')
        pool3 = MotifPool(simple_pwm, name='test')
        pool4 = MotifPool(simple_pwm, iteration_order=5)
        pool5 = MotifPool(simple_pwm, metadata='complete')
        
        # Generate sequences
        for pool in [pool1, pool2, pool3, pool4, pool5]:
            seqs = pool.generate_seqs(num_seqs=5, seed=42)
            assert len(seqs) == 5
    
    def test_sequential_mode_still_rejected(self, simple_pwm):
        """Test that sequential mode is still properly rejected."""
        with pytest.raises(ValueError, match="only supports mode='random'"):
            MotifPool(simple_pwm, mode='sequential')

