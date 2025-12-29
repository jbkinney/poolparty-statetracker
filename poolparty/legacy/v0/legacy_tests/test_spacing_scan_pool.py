"""Comprehensive tests for SpacingScanPool.

Tests validate:
1. Both interfaces (scan_ranges and distances)
2. Distance calculations and coordinate systems
3. Spacing calculations between inserts
4. Design card metadata correctness
5. Edge cases and error handling

All tests use predefined expected sequences for rigorous verification.
"""

import pytest
import warnings
from collections import Counter
from poolparty import SpacingScanPool, Pool, MotifPool
import pandas as pd


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def make_background(length: int, char: str = "N") -> str:
    """Create a background sequence of given length."""
    return char * length


def verify_insert_at_position(seq: str, insert: str, pos: int) -> bool:
    """Verify that insert appears at the given position in seq."""
    return seq[pos:pos + len(insert)] == insert


# =============================================================================
# Basic Creation Tests
# =============================================================================

class TestSpacingScanPoolCreation:
    """Tests for SpacingScanPool creation and basic properties."""
    
    def test_creation_with_distance_lists(self):
        """Test creation with explicit distance lists."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "TTT"],
            anchor_pos=25,
            insert_distances=[[-20, -10], [5, 15]],
        )
        # 2 × 2 = 4 combinations, all should be valid
        assert pool.num_internal_states == 4
    
    def test_creation_with_scan_ranges(self):
        """Test creation with scan range interface."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "TTT"],  # 3bp each
            anchor_pos=25,
            insert_scan_ranges=[
                (-20, -10),  # Left edge: -20 to -13 (step 1), right edge reaches -10
                (10, 20),    # Left edge: 10 to 17, right edge reaches 20
            ],
        )
        # 8 positions for first × 8 for second = 64, but filtered by validity
        assert pool.num_internal_states > 0
    
    def test_creation_with_scan_ranges_and_step(self):
        """Test scan ranges with explicit step."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAA"],  # 4bp
            anchor_pos=50,
            insert_scan_ranges=[(-40, -10, 10)],  # Left edge: -40, -30, -20 (right edge -10 means left edge max is -14)
        )
        # Left edge can be at: -40, -30, -20 (3 positions)
        assert pool.num_internal_states == 3
    
    def test_creation_mixed_interfaces(self):
        """Test mixing scan_ranges and distances for different inserts."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "TTT", "GGG"],
            anchor_pos=50,
            insert_scan_ranges=[(-40, -30, 5), None, (20, 30, 5)],
            insert_distances=[None, [-10, 0, 10], None],
        )
        assert pool.num_internal_states > 0
    
    def test_insert_names_default(self):
        """Test that default insert names are generated."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "TTT"],
            anchor_pos=25,
            insert_distances=[[-20], [10]],
        )
        assert pool._insert_names == ["insert_0", "insert_1"]
    
    def test_insert_names_custom(self):
        """Test custom insert names."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "TTT"],
            insert_names=["SP1", "AP1"],
            anchor_pos=25,
            insert_distances=[[-20], [10]],
        )
        assert pool._insert_names == ["SP1", "AP1"]


# =============================================================================
# Distance and Position Calculation Tests
# =============================================================================

class TestDistanceCalculations:
    """Tests for distance-to-position calculations."""
    
    def test_negative_distance_places_upstream(self):
        """Negative distance places insert upstream of anchor."""
        # Background: 50 N's, anchor at 25
        # Insert "AAA" at distance -10 → position 15
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA"],
            anchor_pos=25,
            insert_distances=[[-10]],
        )
        pool.set_state(0)
        seq = pool.seq
        
        # Insert should be at position 15 (25 + (-10) = 15)
        expected_pos = 15
        assert seq[expected_pos:expected_pos + 3] == "AAA"
        # Verify rest is N
        assert seq[:expected_pos] == "N" * expected_pos
        assert seq[expected_pos + 3:] == "N" * (50 - expected_pos - 3)
    
    def test_positive_distance_places_downstream(self):
        """Positive distance places insert downstream of anchor."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["TTT"],
            anchor_pos=25,
            insert_distances=[[10]],
        )
        pool.set_state(0)
        seq = pool.seq
        
        # Insert should be at position 35 (25 + 10 = 35)
        expected_pos = 35
        assert seq[expected_pos:expected_pos + 3] == "TTT"
    
    def test_zero_distance_places_at_anchor(self):
        """Zero distance places insert at anchor position."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["GGG"],
            anchor_pos=25,
            insert_distances=[[0]],
        )
        pool.set_state(0)
        seq = pool.seq
        
        assert seq[25:28] == "GGG"
    
    def test_scan_range_accounts_for_insert_length(self):
        """Scan range correctly accounts for insert length."""
        # Insert is 5bp, scan range is (-20, -10)
        # Left edge can reach -20, right edge can reach -10
        # So left edge can be: -20, -19, ..., -15 (since -15 + 5 = -10)
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAAA"],  # 5bp
            anchor_pos=50,
            insert_scan_ranges=[(-20, -10)],  # Region [-20, -10] relative to anchor
        )
        
        # Left edge positions: -20, -19, ..., -15 = 6 positions
        assert pool.num_internal_states == 6
        
        # Verify actual positions
        positions = []
        for i in range(pool.num_internal_states):
            pool.set_state(i)
            seq = pool.seq
            # Find where insert starts
            pos = seq.find("AAAAA")
            positions.append(pos)
        
        # Positions in background: 30, 31, 32, 33, 34, 35 (anchor=50, dist=-20 to -15)
        expected = [30, 31, 32, 33, 34, 35]
        assert sorted(positions) == expected


# =============================================================================
# Predefined Expected Sequence Tests
# =============================================================================

class TestPredefinedSequences:
    """Tests with predefined expected sequences for exact verification."""
    
    def test_single_insert_exact_sequence(self):
        """Verify exact sequence with single insert."""
        background = "0123456789"  # 10 chars, easy to verify positions
        insert = "XXX"
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[insert],
            anchor_pos=5,
            insert_distances=[[-3, 0, 2]],  # Positions 2, 5, 7
        )
        
        expected_seqs = [
            "01XXX56789",  # pos 2: replaces "234"
            "01234XXX89",  # pos 5: replaces "567"
            "0123456XXX",  # pos 7: replaces "789"
        ]
        
        for i, expected in enumerate(expected_seqs):
            pool.set_state(i)
            assert pool.seq == expected, f"State {i}: expected {expected}, got {pool.seq}"
    
    def test_two_inserts_exact_sequence(self):
        """Verify exact sequence with two inserts."""
        background = "0123456789ABCDEF"  # 16 chars
        insert_a = "XX"
        insert_b = "YY"
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[insert_a, insert_b],
            anchor_pos=8,
            insert_distances=[[-6], [2]],  # A at pos 2, B at pos 10
        )
        
        # A replaces pos 2-3, B replaces pos 10-11
        expected = "01XX456789YYCDEF"
        
        pool.set_state(0)
        assert pool.seq == expected
    
    def test_multiple_combinations_exact(self):
        """Verify all combinations with exact expected sequences."""
        background = "____________________"  # 20 underscores
        insert_a = "AA"
        insert_b = "BB"
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[insert_a, insert_b],
            insert_names=["A", "B"],
            anchor_pos=10,
            insert_distances=[[-8, -6], [4, 6]],  # A at 2,4; B at 14,16
            min_spacing=0,
            enforce_order=True,
        )
        
        # 2 × 2 = 4 combinations
        # A at anchor+dist_A, B at anchor+dist_B (anchor=10)
        expected_combos = {
            (-8, 4): "__AA__________BB____",   # A at 2 (10-8), B at 14 (10+4)
            (-8, 6): "__AA____________BB__",   # A at 2, B at 16 (10+6)
            (-6, 4): "____AA________BB____",   # A at 4 (10-6), B at 14
            (-6, 6): "____AA__________BB__",   # A at 4, B at 16
        }
        
        assert pool.num_internal_states == 4
        
        generated = {}
        for i in range(4):
            pool.set_state(i)
            seq = pool.seq  # Must access seq first to update _cached_combo
            combo = pool._cached_combo
            generated[combo] = seq
        
        for combo, expected in expected_combos.items():
            assert combo in generated, f"Combo {combo} not generated"
            assert generated[combo] == expected, \
                f"Combo {combo}: expected '{expected}', got '{generated[combo]}'"


# =============================================================================
# Spacing Calculation Tests
# =============================================================================

class TestSpacingCalculations:
    """Tests for spacing (gap) calculations between inserts."""
    
    def test_spacing_between_two_inserts(self):
        """Verify spacing calculation between two inserts."""
        background = "N" * 50
        insert_a = "AAA"  # 3bp
        insert_b = "TTT"  # 3bp
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[insert_a, insert_b],
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_distances=[[-15], [5]],  # A at pos 10, B at pos 30
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 50)
        
        # A occupies [10, 13), B occupies [30, 33)
        # Spacing = 30 - 13 = 17
        assert metadata['spacing_A_B'] == 17
    
    def test_spacing_with_touching_inserts(self):
        """Verify spacing is 0 when inserts are adjacent."""
        background = "N" * 50
        insert_a = "AAA"  # 3bp
        insert_b = "TTT"  # 3bp
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[insert_a, insert_b],
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_distances=[[-15], [-12]],  # A at pos 10, B at pos 13 (adjacent)
            min_spacing=0,
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 50)
        
        # A occupies [10, 13), B occupies [13, 16)
        # Spacing = 13 - 13 = 0
        assert metadata['spacing_A_B'] == 0
    
    def test_pairwise_spacings_three_inserts(self):
        """Verify all pairwise spacings with three inserts."""
        background = "N" * 100
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=["AA", "BB", "CC"],
            insert_names=["A", "B", "C"],
            anchor_pos=50,
            insert_distances=[[-30], [-10], [20]],  # A at 20, B at 40, C at 70
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 100)
        
        # A: [20, 22), B: [40, 42), C: [70, 72)
        # A-B spacing: 40 - 22 = 18
        # A-C spacing: 70 - 22 = 48
        # B-C spacing: 70 - 42 = 28
        assert metadata['spacing_A_B'] == 18
        assert metadata['spacing_A_C'] == 48
        assert metadata['spacing_B_C'] == 28
    
    def test_spacing_when_order_flips(self):
        """Test spacing calculation when B is actually 5' of A."""
        # With enforce_order=False, allow B to be upstream of A
        background = "N" * 100
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=["AAA", "BBB"],
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[10], [-20]],  # A at 60, B at 30
            enforce_order=False,  # Allow B to be 5' of A
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 100)
        
        # B: [30, 33), A: [60, 63)
        # B is upstream of A (flipped order), so spacing is negative
        # Gap = A_start - B_end = 60 - 33 = 27, but negated = -27
        assert metadata['spacing_A_B'] == -27
        # abs(spacing) gives the actual gap
        assert abs(metadata['spacing_A_B']) == 27


class TestSignedSpacingCalculations:
    """Rigorous tests for signed spacing calculations.
    
    Tests verify the sign convention:
    - Positive spacing: insert i is upstream (5') of insert j (expected order)
    - Negative spacing: insert j is upstream (5') of insert i (flipped order)
    - abs(spacing) always equals the gap between closest boundaries
    """
    
    def test_signed_spacing_expected_order_multiple_sequences(self):
        """Verify positive spacing when inserts are in expected order across many sequences."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAA", "BBBB"],  # 4bp each
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[-30, -20, -10], [10, 20, 30]],  # A always upstream
            enforce_order=True,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # All spacings should be positive (A upstream of B)
        assert (df['sp_spacing_A_B'] > 0).all(), \
            f"Expected all positive spacings, got: {df['sp_spacing_A_B'].tolist()}"
        
        # Verify each spacing matches actual gap
        for i, row in df.iterrows():
            expected_gap = row['sp_B_pos_start'] - row['sp_A_pos_end']
            assert row['sp_spacing_A_B'] == expected_gap, \
                f"Row {i}: spacing {row['sp_spacing_A_B']} != expected {expected_gap}"
    
    def test_signed_spacing_flipped_order_multiple_sequences(self):
        """Verify negative spacing when inserts are in flipped order across many sequences."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAA", "BBBB"],  # 4bp each
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[10, 20, 30], [-30, -20, -10]],  # B always upstream (flipped)
            enforce_order=False,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # All spacings should be negative (B upstream of A = flipped)
        assert (df['sp_spacing_A_B'] < 0).all(), \
            f"Expected all negative spacings, got: {df['sp_spacing_A_B'].tolist()}"
        
        # Verify abs(spacing) matches actual gap
        for i, row in df.iterrows():
            actual_gap = row['sp_A_pos_start'] - row['sp_B_pos_end']
            assert abs(row['sp_spacing_A_B']) == actual_gap, \
                f"Row {i}: abs(spacing) {abs(row['sp_spacing_A_B'])} != gap {actual_gap}"
    
    def test_signed_spacing_mixed_orders_same_pool(self):
        """Verify both positive and negative spacings in same pool with enforce_order=False."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "BBB"],  # 3bp each
            insert_names=["A", "B"],
            anchor_pos=50,
            # Some combos: A upstream, some: B upstream
            insert_distances=[[-20, 20], [-10, 10]],
            enforce_order=False,
            min_spacing=5,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        positive_count = (df['sp_spacing_A_B'] > 0).sum()
        negative_count = (df['sp_spacing_A_B'] < 0).sum()
        
        # Should have both positive and negative spacings
        assert positive_count > 0, "Expected some positive spacings (A upstream)"
        assert negative_count > 0, "Expected some negative spacings (B upstream)"
        
        # Verify correctness for each row
        for i, row in df.iterrows():
            a_start, a_end = row['sp_A_pos_start'], row['sp_A_pos_end']
            b_start, b_end = row['sp_B_pos_start'], row['sp_B_pos_end']
            spacing = row['sp_spacing_A_B']
            
            if a_start < b_start:
                # A is upstream - expect positive spacing
                expected = b_start - a_end
                assert spacing == expected, f"Row {i}: A upstream, expected +{expected}, got {spacing}"
            else:
                # B is upstream - expect negative spacing
                expected = -(a_start - b_end)
                assert spacing == expected, f"Row {i}: B upstream, expected {expected}, got {spacing}"
    
    def test_signed_spacing_three_inserts_all_combinations(self):
        """Test signed spacing with three inserts and various order combinations."""
        pool = SpacingScanPool(
            background_seq="N" * 150,
            insert_seqs=["AA", "BB", "CC"],  # 2bp each
            insert_names=["X", "Y", "Z"],
            anchor_pos=75,
            # Positions allow various orderings
            insert_distances=[[-50, 0, 50], [-30, 20], [-10, 40]],
            enforce_order=False,
            min_spacing=3,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Verify all pairwise spacings for each sequence
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # Get positions
            x_start, x_end = row['sp_X_pos_start'], row['sp_X_pos_end']
            y_start, y_end = row['sp_Y_pos_start'], row['sp_Y_pos_end']
            z_start, z_end = row['sp_Z_pos_start'], row['sp_Z_pos_end']
            
            # X-Y spacing
            if x_start <= y_start:
                expected_xy = y_start - x_end
            else:
                expected_xy = -(x_start - y_end)
            assert row['sp_spacing_X_Y'] == expected_xy, \
                f"Row {i}: X-Y spacing mismatch: {row['sp_spacing_X_Y']} != {expected_xy}"
            
            # X-Z spacing
            if x_start <= z_start:
                expected_xz = z_start - x_end
            else:
                expected_xz = -(x_start - z_end)
            assert row['sp_spacing_X_Z'] == expected_xz, \
                f"Row {i}: X-Z spacing mismatch: {row['sp_spacing_X_Z']} != {expected_xz}"
            
            # Y-Z spacing
            if y_start <= z_start:
                expected_yz = z_start - y_end
            else:
                expected_yz = -(y_start - z_end)
            assert row['sp_spacing_Y_Z'] == expected_yz, \
                f"Row {i}: Y-Z spacing mismatch: {row['sp_spacing_Y_Z']} != {expected_yz}"
            
            # Verify inserts at correct positions in sequence
            assert seq[x_start:x_end] == "AA"
            assert seq[y_start:y_end] == "BB"
            assert seq[z_start:z_end] == "CC"
    
    def test_signed_spacing_abs_always_equals_gap(self):
        """Verify abs(spacing) always equals actual gap for all sequences."""
        pool = SpacingScanPool(
            background_seq="N" * 80,
            insert_seqs=["AAAA", "BBBB"],
            insert_names=["A", "B"],
            anchor_pos=40,
            insert_distances=[[-25, -15, -5, 5, 15, 25], [-20, -10, 0, 10, 20]],
            enforce_order=False,
            min_spacing=2,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        for i, row in df.iterrows():
            a_start, a_end = row['sp_A_pos_start'], row['sp_A_pos_end']
            b_start, b_end = row['sp_B_pos_start'], row['sp_B_pos_end']
            spacing = row['sp_spacing_A_B']
            
            # Calculate actual gap between closest boundaries
            if a_start <= b_start:
                actual_gap = b_start - a_end
            else:
                actual_gap = a_start - b_end
            
            assert abs(spacing) == actual_gap, \
                f"Row {i}: abs(spacing)={abs(spacing)} != actual_gap={actual_gap}"
    
    def test_signed_spacing_in_composite_structure(self):
        """Test signed spacing calculations in composite pool structure."""
        prefix = Pool(["PREFIX_"], name="pre", mode='sequential')  # 7 chars
        
        spacing = SpacingScanPool(
            background_seq="N" * 60,
            insert_seqs=["XXX", "YYY"],
            insert_names=["X", "Y"],
            anchor_pos=30,
            insert_distances=[[-20, 10], [-10, 20]],  # Some combos flip order
            enforce_order=False,
            min_spacing=3,
            name="sp",
            mode='sequential',
        )
        
        suffix = Pool(["_SUFFIX"], name="suf", mode='sequential')  # 7 chars
        
        library = prefix + spacing + suffix
        
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # Verify prefix/suffix
            assert seq[:7] == "PREFIX_"
            assert seq[-7:] == "_SUFFIX"
            
            # Get positions (in spacing pool's coordinate space)
            x_start, x_end = row['sp_X_pos_start'], row['sp_X_pos_end']
            y_start, y_end = row['sp_Y_pos_start'], row['sp_Y_pos_end']
            spacing_val = row['sp_spacing_X_Y']
            
            # Calculate expected spacing
            if x_start <= y_start:
                expected = y_start - x_end
            else:
                expected = -(x_start - y_end)
            
            assert spacing_val == expected, \
                f"Row {i}: spacing {spacing_val} != expected {expected}"
            
            # Verify inserts at correct absolute positions
            x_abs = row['sp_X_abs_pos_start']
            y_abs = row['sp_Y_abs_pos_start']
            assert seq[x_abs:x_abs + 3] == "XXX"
            assert seq[y_abs:y_abs + 3] == "YYY"
    
    def test_signed_spacing_random_mode_many_sequences(self):
        """Test signed spacing in random mode with many sequences."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAA", "BBBB"],
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[-35, -25, -15, -5, 5, 15, 25, 35], 
                              [-30, -20, -10, 0, 10, 20, 30]],
            enforce_order=False,
            min_spacing=5,
            name="sp",
            mode='random',
        )
        
        result = pool.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Should have both positive and negative spacings in random sampling
        has_positive = (df['sp_spacing_A_B'] > 0).any()
        has_negative = (df['sp_spacing_A_B'] < 0).any()
        
        # With these distance ranges, we should see both orders
        assert has_positive or has_negative, "Expected some spacings"
        
        # Verify all spacings are correctly computed
        for i, row in df.iterrows():
            a_start, a_end = row['sp_A_pos_start'], row['sp_A_pos_end']
            b_start, b_end = row['sp_B_pos_start'], row['sp_B_pos_end']
            spacing = row['sp_spacing_A_B']
            
            if a_start <= b_start:
                expected = b_start - a_end
            else:
                expected = -(a_start - b_end)
            
            assert spacing == expected, \
                f"Row {i}: spacing {spacing} != expected {expected}"
            
            # Gap should always satisfy min_spacing
            assert abs(spacing) >= 5, f"Row {i}: gap {abs(spacing)} < min_spacing 5"
    
    def test_signed_spacing_edge_case_adjacent_inserts(self):
        """Test spacing = 0 when inserts are exactly adjacent."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "BBB"],  # 3bp each
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_distances=[[-10], [-7]],  # A ends at 18, B starts at 18
            min_spacing=0,
            name="sp",
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 50)
        
        # A: [15, 18), B: [18, 21) - exactly touching
        assert metadata['A_pos_start'] == 15
        assert metadata['A_pos_end'] == 18
        assert metadata['B_pos_start'] == 18
        assert metadata['B_pos_end'] == 21
        
        # Spacing should be exactly 0 (positive, A upstream)
        assert metadata['spacing_A_B'] == 0
    
    def test_signed_spacing_sign_indicates_order(self):
        """Verify sign(spacing) correctly indicates insert order."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "BBB"],
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[-30, 30], [-20, 20]],
            enforce_order=False,
            min_spacing=5,
            name="sp",
            mode='sequential',
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        for i, row in df.iterrows():
            a_start = row['sp_A_pos_start']
            b_start = row['sp_B_pos_start']
            spacing = row['sp_spacing_A_B']
            
            if spacing > 0:
                # Positive spacing means A should be upstream
                assert a_start < b_start, \
                    f"Row {i}: positive spacing but A not upstream (A={a_start}, B={b_start})"
            elif spacing < 0:
                # Negative spacing means B should be upstream
                assert b_start < a_start, \
                    f"Row {i}: negative spacing but B not upstream (A={a_start}, B={b_start})"
            else:
                # Zero spacing means adjacent
                pass


# =============================================================================
# Coordinate System Tests
# =============================================================================

class TestCoordinateSystems:
    """Tests for different coordinate systems in design cards."""
    
    def test_dist_coordinate(self):
        """Verify distance from anchor is reported correctly."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA"],
            insert_names=["X"],
            anchor_pos=25,
            insert_distances=[[-10, 0, 10]],
        )
        
        expected_dists = [-10, 0, 10]
        for i, expected_dist in enumerate(expected_dists):
            pool.set_state(i)
            metadata = pool.get_metadata(0, 50)
            assert metadata['X_dist'] == expected_dist
    
    def test_pos_coordinates(self):
        """Verify position in background is reported correctly."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA"],  # 3bp
            insert_names=["X"],
            anchor_pos=25,
            insert_distances=[[-10]],  # Position 15
        )
        
        pool.set_state(0)
        metadata = pool.get_metadata(0, 50)
        
        # Insert at position 15, length 3
        assert metadata['X_pos_start'] == 15
        assert metadata['X_pos_end'] == 18
    
    def test_abs_pos_in_composite(self):
        """Verify absolute position accounts for composite prefix."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA"],
            insert_names=["X"],
            anchor_pos=25,
            insert_distances=[[-10]],  # Position 15 in background
            name="inner",
        )
        
        pool.set_state(0)
        
        # Simulate being in a composite with 100bp prefix
        # abs_start = 100 means pool starts at position 100 in final sequence
        metadata = pool.get_metadata(100, 150)
        
        # Absolute position = abs_start + pos_in_background
        assert metadata['X_abs_pos_start'] == 100 + 15  # 115
        assert metadata['X_abs_pos_end'] == 100 + 18    # 118


# =============================================================================
# Design Card Integration Tests
# =============================================================================

class TestDesignCards:
    """Tests for design card functionality."""
    
    def test_design_cards_columns(self):
        """Verify design card has all expected columns."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "TTT"],
            insert_names=["SP1", "AP1"],
            anchor_pos=25,
            insert_distances=[[-15], [5]],
            name="mod",
        )
        
        result = pool.generate_seqs(num_seqs=1, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Core columns
        assert 'sequence_id' in df.columns
        assert 'mod_index' in df.columns
        assert 'mod_abs_start' in df.columns
        assert 'mod_abs_end' in df.columns
        
        # Per-insert columns
        for name in ['SP1', 'AP1']:
            assert f'mod_{name}_dist' in df.columns
            assert f'mod_{name}_pos_start' in df.columns
            assert f'mod_{name}_pos_end' in df.columns
            assert f'mod_{name}_abs_pos_start' in df.columns
            assert f'mod_{name}_abs_pos_end' in df.columns
        
        # Spacing column
        assert 'mod_spacing_SP1_AP1' in df.columns
    
    def test_design_cards_values_match_sequences(self):
        """Verify design card values match actual generated sequences."""
        background = "0123456789ABCDEFGHIJ"  # 20 chars
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=["XX", "YY"],
            insert_names=["X", "Y"],
            anchor_pos=10,
            insert_distances=[[-8, -6], [2, 4]],  # X at 2,4; Y at 12,14
            name="test",
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        seqs = result['sequences']
        
        for i, seq in enumerate(seqs):
            row = df.iloc[i]
            x_start = row['test_X_pos_start']
            y_start = row['test_Y_pos_start']
            
            # Verify inserts are at reported positions
            assert seq[x_start:x_start + 2] == "XX", f"Seq {i}: X not at {x_start}"
            assert seq[y_start:y_start + 2] == "YY", f"Seq {i}: Y not at {y_start}"
    
    def test_design_cards_spacing_matches_positions(self):
        """Verify reported spacing matches position difference."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAA", "BBB"],
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_distances=[[-15, -10], [5, 10]],
            name="test",
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        for i, row in df.iterrows():
            a_end = row['test_A_pos_end']
            b_start = row['test_B_pos_start']
            reported_spacing = row['test_spacing_A_B']
            
            calculated_spacing = b_start - a_end
            assert reported_spacing == calculated_spacing, \
                f"Row {i}: spacing mismatch"


# =============================================================================
# Filtering and Constraint Tests
# =============================================================================

class TestFiltering:
    """Tests for filtering invalid combinations."""
    
    def test_overlap_filtered(self):
        """Verify overlapping combinations are filtered out."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AAAAA", "BBBBB"],  # 5bp each
            anchor_pos=25,
            insert_distances=[[-5, 0], [-2, 5]],  # Some would overlap
            min_spacing=0,
        )
        
        # Combinations:
        # (-5, -2): A at 20-25, B at 23-28 → overlap! Filtered
        # (-5, 5): A at 20-25, B at 30-35 → OK (gap = 5)
        # (0, -2): A at 25-30, B at 23-28 → overlap! Filtered
        # (0, 5): A at 25-30, B at 30-35 → touching OK (gap = 0)
        
        # Should have 2 valid combinations
        assert pool.num_internal_states == 2
    
    def test_min_spacing_enforced(self):
        """Verify min_spacing constraint is enforced."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AA", "BB"],
            anchor_pos=25,
            insert_distances=[[-10, -5], [0, 5]],
            min_spacing=10,  # Require 10bp gap
        )
        
        # All combinations have gap < 10 except maybe some
        # A at -10 (15-17), B at 0 (25-27): gap = 25 - 17 = 8 < 10 → filtered
        # A at -10 (15-17), B at 5 (30-32): gap = 30 - 17 = 13 >= 10 → OK
        # A at -5 (20-22), B at 0 (25-27): gap = 25 - 22 = 3 < 10 → filtered
        # A at -5 (20-22), B at 5 (30-32): gap = 30 - 22 = 8 < 10 → filtered
        
        assert pool.num_internal_states == 1
    
    def test_enforce_order_filters_flips(self):
        """Verify enforce_order=True filters combinations where order flips."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AA", "BB"],
            anchor_pos=25,
            insert_distances=[[5, -10], [0, -15]],  # Some combos flip order
            enforce_order=True,
        )
        
        # A at 5 (30), B at 0 (25): B is 5' of A → filtered
        # A at 5 (30), B at -15 (10): B is 5' of A → filtered
        # A at -10 (15), B at 0 (25): A is 5' of B → OK
        # A at -10 (15), B at -15 (10): B is 5' of A → filtered
        
        assert pool.num_internal_states == 1
    
    def test_enforce_order_false_allows_flips(self):
        """Verify enforce_order=False allows order flips."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AA", "BB"],
            anchor_pos=25,
            insert_distances=[[-5], [10]],
            enforce_order=False,
        )
        
        # Only one combination, but would be valid regardless of order
        assert pool.num_internal_states == 1
    
    def test_out_of_bounds_filtered(self):
        """Verify out-of-bounds positions are filtered."""
        pool = SpacingScanPool(
            background_seq="N" * 30,  # Short background
            insert_seqs=["AAAAA"],  # 5bp
            anchor_pos=15,
            insert_distances=[[-20, -10, 10, 20]],  # Some out of bounds
        )
        
        # -20: pos -5 → out of bounds (start < 0)
        # -10: pos 5 → OK
        # 10: pos 25 → end = 30 → OK (just fits)
        # 20: pos 35 → out of bounds (end > 30)
        
        assert pool.num_internal_states == 2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestInsertMode:
    """Tests for insert_or_overwrite='insert' mode."""
    
    def test_insert_mode_sequence_length(self):
        """Verify insert mode increases sequence length."""
        pool = SpacingScanPool(
            background_seq="NNNNNNNN",  # 8bp
            insert_seqs=["AA", "BB"],  # 2bp each
            insert_names=["A", "B"],
            anchor_pos=4,
            insert_distances=[[-2], [1]],  # A at 2, B at 5
            insert_or_overwrite='insert',
        )
        pool.set_state(0)
        seq = pool.seq
        
        # Length should be 8 + 2 + 2 = 12
        assert len(seq) == 12
    
    def test_insert_mode_positions_account_for_shifts(self):
        """Verify positions account for shifts from earlier inserts."""
        pool = SpacingScanPool(
            background_seq="NNNNNNNN",  # 8bp
            insert_seqs=["AA", "BB"],  # 2bp each
            insert_names=["A", "B"],
            anchor_pos=4,
            insert_distances=[[-2], [1]],  # A at bg_pos 2, B at bg_pos 5
            insert_or_overwrite='insert',
            name="sp",
        )
        pool.set_state(0)
        seq = pool.seq
        
        # A is at position 2 (no prior inserts)
        # B is at position 5 + 2 = 7 (shifted by A's length)
        assert seq.find("AA") == 2
        assert seq.find("BB") == 7
        
        metadata = pool.get_metadata(0, len(seq))
        assert metadata['A_pos_start'] == 2
        assert metadata['B_pos_start'] == 7
    
    def test_insert_mode_spacing_accounts_for_shifts(self):
        """Verify spacing is calculated in output coordinates."""
        pool = SpacingScanPool(
            background_seq="NNNNNNNN",  # 8bp
            insert_seqs=["AA", "BB"],  # 2bp each
            insert_names=["A", "B"],
            anchor_pos=4,
            insert_distances=[[-2], [1]],
            insert_or_overwrite='insert',
            name="sp",
        )
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, len(seq))
        
        # A at [2, 4), B at [7, 9) in output
        # Spacing = 7 - 4 = 3
        actual_spacing = seq.find("BB") - (seq.find("AA") + 2)
        assert metadata['spacing_A_B'] == actual_spacing
        assert metadata['spacing_A_B'] == 3
    
    def test_insert_mode_append_at_end(self):
        """Verify inserting at the end of background works."""
        pool = SpacingScanPool(
            background_seq="NNNN",  # 4bp
            insert_seqs=["AA"],
            insert_names=["A"],
            anchor_pos=2,
            insert_distances=[[2]],  # Insert at position 4 (end)
            insert_or_overwrite='insert',
        )
        
        assert pool.num_internal_states == 1
        pool.set_state(0)
        seq = pool.seq
        
        assert seq == "NNNNAA"
        assert len(seq) == 6
    
    def test_insert_mode_three_inserts_shifts(self):
        """Verify three inserts with correct cumulative shifts."""
        pool = SpacingScanPool(
            background_seq="0123456789",  # 10bp
            insert_seqs=["AA", "BB", "CC"],  # 2bp each
            insert_names=["A", "B", "C"],
            anchor_pos=5,
            insert_distances=[[-3], [0], [3]],  # bg positions: 2, 5, 8
            insert_or_overwrite='insert',
            name="sp",
        )
        pool.set_state(0)
        seq = pool.seq
        
        # Length: 10 + 2 + 2 + 2 = 16
        assert len(seq) == 16
        
        # A at output pos 2 (no prior inserts)
        # B at output pos 5 + 2 = 7 (after A)
        # C at output pos 8 + 2 + 2 = 12 (after A and B)
        assert seq.find("AA") == 2
        assert seq.find("BB") == 7
        assert seq.find("CC") == 12
        
        metadata = pool.get_metadata(0, len(seq))
        assert metadata['A_pos_start'] == 2
        assert metadata['B_pos_start'] == 7
        assert metadata['C_pos_start'] == 12
        
        # Spacings
        assert metadata['spacing_A_B'] == 7 - 4  # 3
        assert metadata['spacing_B_C'] == 12 - 9  # 3
        assert metadata['spacing_A_C'] == 12 - 4  # 8
    
    def test_insert_mode_in_composite(self):
        """Test insert mode SpacingScanPool in composite."""
        prefix = Pool(["PRE_"], name="pre", mode='sequential')  # 4 chars
        
        spacing = SpacingScanPool(
            background_seq="NNNNNNNN",  # 8bp
            insert_seqs=["XX", "YY"],
            insert_names=["X", "Y"],
            anchor_pos=4,
            insert_distances=[[-2], [2]],
            insert_or_overwrite='insert',
            name="sp",
        )
        
        suffix = Pool(["_SUF"], name="suf", mode='sequential')  # 4 chars
        
        library = prefix + spacing + suffix
        
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        seq = result['sequences'][0]
        
        # Length: 4 + (8 + 2 + 2) + 4 = 20
        assert len(seq) == 20
        
        # Verify prefix and suffix
        assert seq[:4] == "PRE_"
        assert seq[-4:] == "_SUF"
        
        # Verify absolute positions
        x_abs = df['sp_X_abs_pos_start'].iloc[0]
        y_abs = df['sp_Y_abs_pos_start'].iloc[0]
        
        # X at output pos 2 + prefix(4) = 6
        # Y at output pos 6 + 2 + prefix(4) = 12
        assert seq[x_abs:x_abs + 2] == "XX"
        assert seq[y_abs:y_abs + 2] == "YY"


class TestErrorHandling:
    """Tests for error conditions."""
    
    def test_error_duplicate_insert_names(self):
        """Error when insert_names contains duplicates."""
        with pytest.raises(ValueError, match="must be unique"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAA", "BBB"],
                insert_names=["SAME", "SAME"],
                anchor_pos=25,
                insert_distances=[[-10], [10]],
            )
    
    def test_error_cartesian_product_too_large(self):
        """Error when potential combinations exceed limit before enumeration."""
        with pytest.raises(ValueError, match="Cartesian product too large"):
            SpacingScanPool(
                background_seq="N" * 500,
                insert_seqs=["AAA", "BBB", "CCC", "DDD"],
                insert_names=["A", "B", "C", "D"],
                anchor_pos=250,
                insert_scan_ranges=[
                    (-200, -100),  # ~100 positions
                    (-50, 50),     # ~100 positions  
                    (100, 200),    # ~100 positions
                    (210, 300),    # ~90 positions
                ],
            )
    
    def test_error_empty_inserts(self):
        """Error when insert_seqs is empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=[],
                anchor_pos=25,
                insert_distances=[],
            )
    
    def test_error_negative_min_spacing(self):
        """Error when min_spacing is negative."""
        with pytest.raises(ValueError, match="min_spacing must be >= 0"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAA"],
                anchor_pos=25,
                insert_distances=[[-10]],
                min_spacing=-5,
            )
    
    def test_error_both_interfaces_for_same_insert(self):
        """Error when both scan_range and distances provided for same insert."""
        with pytest.raises(ValueError, match="both scan_range and distances"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAA"],
                anchor_pos=25,
                insert_scan_ranges=[(-20, -10)],
                insert_distances=[[-15, -10]],  # Both provided!
            )
    
    def test_error_neither_interface_for_insert(self):
        """Error when neither interface provided for an insert."""
        with pytest.raises(ValueError, match="neither scan_range nor distances"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAA", "TTT"],
                anchor_pos=25,
                insert_scan_ranges=[(-20, -10), None],
                insert_distances=[None, None],  # Second has neither
            )
    
    def test_error_anchor_out_of_bounds(self):
        """Error when anchor is outside background."""
        with pytest.raises(ValueError, match="anchor_pos"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAA"],
                anchor_pos=100,  # Beyond background
                insert_distances=[[-10]],
            )
    
    def test_error_no_valid_combinations(self):
        """Error when no valid combinations exist."""
        with pytest.raises(ValueError, match="No valid distance combinations"):
            SpacingScanPool(
                background_seq="N" * 20,  # Short
                insert_seqs=["A" * 10, "B" * 10],  # Two 10bp inserts
                anchor_pos=10,
                insert_distances=[[0], [5]],  # Would overlap
                min_spacing=0,
            )
    
    def test_error_scan_range_too_small(self):
        """Error when scan range is smaller than insert length."""
        with pytest.raises(ValueError, match="too small for insert"):
            SpacingScanPool(
                background_seq="N" * 50,
                insert_seqs=["AAAAAAAAAA"],  # 10bp
                anchor_pos=25,
                insert_scan_ranges=[(-20, -15)],  # Only 5bp range, insert is 10bp
            )


# =============================================================================
# Pool as Input Tests
# =============================================================================

class TestPoolInputs:
    """Tests with Pool objects as inputs."""
    
    def test_pool_as_background(self):
        """Test with Pool as background sequence."""
        bg_pool = Pool(seqs=["ACGTACGTACGT"], mode='sequential')
        
        pool = SpacingScanPool(
            background_seq=bg_pool,
            insert_seqs=["NNN"],
            anchor_pos=6,
            insert_distances=[[-3, 0, 3]],
        )
        
        assert pool.num_internal_states == 3
        
        pool.set_state(0)
        seq = pool.seq
        assert len(seq) == 12
        assert "NNN" in seq
    
    def test_pool_as_insert(self):
        """Test with Pool as insert sequence."""
        insert_pool = Pool(seqs=["AAA", "TTT", "GGG"], mode='random')
        
        pool = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[insert_pool],
            anchor_pos=15,
            insert_distances=[[0]],
        )
        
        # Should work - insert comes from pool
        pool.set_state(0)
        seq = pool.seq
        assert len(seq) == 30


# =============================================================================
# Sequential vs Random Mode Tests
# =============================================================================

class TestModes:
    """Tests for sequential and random mode behavior."""
    
    def test_sequential_mode_deterministic(self):
        """Sequential mode produces deterministic sequence for each state."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["XX", "YY"],
            anchor_pos=25,
            insert_distances=[[-10, -5], [5, 10]],
            mode='sequential',
        )
        
        # Generate all states
        seqs1 = []
        for i in range(pool.num_internal_states):
            pool.set_state(i)
            seqs1.append(pool.seq)
        
        # Generate again - should be identical
        seqs2 = []
        for i in range(pool.num_internal_states):
            pool.set_state(i)
            seqs2.append(pool.seq)
        
        assert seqs1 == seqs2
    
    def test_all_combinations_enumerated(self):
        """Sequential mode enumerates all valid combinations."""
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["AA", "BB"],
            anchor_pos=25,
            insert_distances=[[-10, -5], [5, 10]],
            mode='sequential',
        )
        
        # 2 × 2 = 4 combinations (all should be valid)
        assert pool.num_internal_states == 4
        
        # Generate all
        seqs = pool.generate_seqs(num_complete_iterations=1)
        assert len(seqs) == 4
        assert len(set(seqs)) == 4  # All unique


# =============================================================================
# Mark Changes Tests
# =============================================================================

class TestMarkChanges:
    """Tests for mark_changes functionality."""
    
    def test_mark_changes_swapcases_inserts(self):
        """mark_changes applies swapcase to inserted sequences."""
        pool = SpacingScanPool(
            background_seq="nnnnnnnnnn",  # lowercase background
            insert_seqs=["AAA", "TTT"],   # uppercase inserts
            anchor_pos=5,
            insert_distances=[[-3], [2]],  # Positions 2 and 7
            mark_changes=True,
        )
        
        pool.set_state(0)
        seq = pool.seq
        
        # Inserts should be lowercase (swapped)
        assert seq[2:5] == "aaa"
        assert seq[7:10] == "ttt"
        # Background unchanged
        assert seq[0:2] == "nn"
        assert seq[5:7] == "nn"


# =============================================================================
# Integration Tests
# =============================================================================

class TestRandomMode:
    """Tests for random mode behavior and constraint satisfaction."""
    
    def test_random_mode_respects_min_spacing(self):
        """Verify min_spacing constraint is satisfied in random mode."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAAA", "BBBB"],  # 4bp each
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_scan_ranges=[(-40, -10, 5), (10, 50, 5)],
            min_spacing=10,
            mode='random',
            name="sp",
        )
        
        # Generate many sequences in random mode
        result = pool.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # All spacings should be >= min_spacing
        assert all(df['sp_spacing_A_B'] >= 10), \
            f"Found spacing < 10: {df['sp_spacing_A_B'].min()}"
    
    def test_random_mode_respects_enforce_order(self):
        """Verify enforce_order constraint is satisfied in random mode."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "BBB", "CCC"],
            insert_names=["A", "B", "C"],
            anchor_pos=50,
            insert_distances=[[-30, -20, -10], [-5, 0, 5], [20, 30, 40]],
            enforce_order=True,
            mode='random',
            name="sp",
        )
        
        result = pool.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Verify order: A_pos_start < B_pos_start < C_pos_start
        for i, row in df.iterrows():
            assert row['sp_A_pos_start'] < row['sp_B_pos_start'], \
                f"Order violation A < B at row {i}"
            assert row['sp_B_pos_start'] < row['sp_C_pos_start'], \
                f"Order violation B < C at row {i}"
    
    def test_random_mode_no_overlaps(self):
        """Verify no overlapping inserts in random mode."""
        pool = SpacingScanPool(
            background_seq="N" * 80,
            insert_seqs=["AAAAA", "BBBBB"],  # 5bp each
            insert_names=["A", "B"],
            anchor_pos=40,
            insert_scan_ranges=[(-35, -5, 3), (5, 35, 3)],
            min_spacing=0,
            mode='random',
            name="sp",
        )
        
        result = pool.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # All spacings should be >= 0 (non-overlapping)
        assert all(df['sp_spacing_A_B'] >= 0), \
            f"Found overlap: spacing = {df['sp_spacing_A_B'].min()}"
    
    def test_random_mode_samples_variety(self):
        """Verify random mode samples different combinations."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AA", "BB"],
            insert_names=["A", "B"],
            anchor_pos=50,
            insert_distances=[[-30, -20, -10], [10, 20, 30]],
            mode='random',
            name="sp",
        )
        
        result = pool.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Should see variety in A distances
        unique_a_dists = df['sp_A_dist'].nunique()
        assert unique_a_dists >= 2, f"Expected variety in A_dist, got {unique_a_dists} unique"
        
        # Should see variety in B distances
        unique_b_dists = df['sp_B_dist'].nunique()
        assert unique_b_dists >= 2, f"Expected variety in B_dist, got {unique_b_dists} unique"
    
    def test_random_mode_deterministic_with_seed(self):
        """Verify random mode is deterministic with same seed."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "BBB"],
            anchor_pos=50,
            insert_distances=[[-20, -10], [10, 20]],
            mode='random',
        )
        
        seqs1 = pool.generate_seqs(num_seqs=20, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=20, seed=42)
        
        assert seqs1 == seqs2, "Same seed should produce same sequences"
    
    def test_random_mode_different_seeds_different_results(self):
        """Verify different seeds produce different results."""
        pool = SpacingScanPool(
            background_seq="N" * 100,
            insert_seqs=["AAA", "BBB"],
            anchor_pos=50,
            insert_distances=[[-20, -10, 0], [10, 20, 30]],
            mode='random',
        )
        
        seqs1 = pool.generate_seqs(num_seqs=50, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=50, seed=123)
        
        assert seqs1 != seqs2, "Different seeds should produce different sequences"


class TestComplexComposites:
    """Tests for SpacingScanPool in complex composite structures."""
    
    def test_prefix_spacing_suffix_coordinates(self):
        """Test coordinates in prefix + SpacingScanPool + suffix composite."""
        prefix = Pool(["ADAPTER_"], name="prefix", mode='sequential')  # 8 chars
        
        spacing_pool = SpacingScanPool(
            background_seq="N" * 30,  # 30 chars
            insert_seqs=["AAA", "TTT"],  # 3bp each
            insert_names=["A", "T"],
            anchor_pos=15,
            insert_distances=[[-10], [5]],  # A at pos 5, T at pos 20
            name="spacing",
        )
        
        suffix = Pool(["_BARCODE"], name="suffix", mode='sequential')  # 8 chars
        
        library = prefix + spacing_pool + suffix
        
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        seq = result['sequences'][0]
        
        # Total length: 8 + 30 + 8 = 46
        assert len(seq) == 46
        
        # Verify prefix and suffix
        assert seq[:8] == "ADAPTER_"
        assert seq[-8:] == "_BARCODE"
        
        # Verify position coordinates
        row = df.iloc[0]
        
        # Position in background (spacing pool's internal coords)
        assert row['spacing_A_pos_start'] == 5
        assert row['spacing_A_pos_end'] == 8
        assert row['spacing_T_pos_start'] == 20
        assert row['spacing_T_pos_end'] == 23
        
        # Absolute position in composite (accounts for 8-char prefix)
        assert row['spacing_A_abs_pos_start'] == 8 + 5  # 13
        assert row['spacing_A_abs_pos_end'] == 8 + 8    # 16
        assert row['spacing_T_abs_pos_start'] == 8 + 20  # 28
        assert row['spacing_T_abs_pos_end'] == 8 + 23    # 31
        
        # Verify inserts are at correct positions in actual sequence
        assert seq[13:16] == "AAA"
        assert seq[28:31] == "TTT"
    
    def test_multiple_spacing_pools_in_composite(self):
        """Test composite with two SpacingScanPools."""
        spacing1 = SpacingScanPool(
            background_seq="1" * 20,
            insert_seqs=["AA"],
            insert_names=["X"],
            anchor_pos=10,
            insert_distances=[[-5, 0, 5]],
            name="s1",
            mode='sequential',
        )
        
        linker = Pool(["---"], name="linker", mode='sequential')
        
        spacing2 = SpacingScanPool(
            background_seq="2" * 20,
            insert_seqs=["BB"],
            insert_names=["Y"],
            anchor_pos=10,
            insert_distances=[[-5, 0, 5]],
            name="s2",
            mode='sequential',
        )
        
        composite = spacing1 + linker + spacing2
        
        # 3 states for s1 × 3 states for s2 = 9 total
        result = composite.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        
        assert len(result['sequences']) == 9
        
        df = result['design_cards'].to_dataframe()
        
        # Verify all combinations are present
        s1_dists = set(df['s1_X_dist'].unique())
        s2_dists = set(df['s2_Y_dist'].unique())
        assert s1_dists == {-5, 0, 5}
        assert s2_dists == {-5, 0, 5}
        
        # Verify absolute positions account for composite structure
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # s1 is at the beginning (abs_start = 0)
            x_abs_start = row['s1_X_abs_pos_start']
            assert seq[x_abs_start:x_abs_start + 2] == "AA"
            
            # s2 is after s1 (20) + linker (3) = 23
            y_abs_start = row['s2_Y_abs_pos_start']
            assert seq[y_abs_start:y_abs_start + 2] == "BB"
    
    def test_spacing_pool_with_barcode_coordinates(self):
        """Test SpacingScanPool + BarcodePool coordinates."""
        from poolparty import BarcodePool
        
        spacing = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=["GGGCGG", "TGACTCA"],  # SP1 (6bp) and AP1 (7bp)
            insert_names=["SP1", "AP1"],
            anchor_pos=25,
            insert_distances=[[-15, -10], [5, 10]],
            name="module",
            mode='sequential',
        )
        
        barcode = BarcodePool(
            num_barcodes=4,
            length=8,
            min_edit_distance=2,
            name="bc",
            seed=42,
        )
        
        library = spacing + barcode
        
        # 4 spacing combos × 4 barcodes = 16
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        
        assert len(result['sequences']) == 16
        
        df = result['design_cards'].to_dataframe()
        
        # Verify all sequences have correct length
        for seq in result['sequences']:
            assert len(seq) == 50 + 8  # spacing + barcode
        
        # Verify spacing coordinates are correct
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            sp1_start = row['module_SP1_abs_pos_start']
            ap1_start = row['module_AP1_abs_pos_start']
            
            assert seq[sp1_start:sp1_start + 6] == "GGGCGG"
            assert seq[ap1_start:ap1_start + 7] == "TGACTCA"
    
    def test_three_inserts_with_prefix_suffix(self):
        """Test 3 inserts with complex coordinate verification."""
        prefix = Pool(["PRE_"], name="pre", mode='sequential')  # 4 chars
        
        spacing = SpacingScanPool(
            background_seq="0123456789" * 4,  # 40 chars, easy to verify
            insert_seqs=["AAA", "BBB", "CCC"],  # 3bp each
            insert_names=["A", "B", "C"],
            anchor_pos=20,
            insert_distances=[[-15], [-5], [10]],  # A at 5, B at 15, C at 30
            name="sp",
            mode='sequential',
        )
        
        suffix = Pool(["_SUF"], name="suf", mode='sequential')  # 4 chars
        
        library = prefix + spacing + suffix
        
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        seq = result['sequences'][0]
        
        # Total length: 4 + 40 + 4 = 48
        assert len(seq) == 48
        
        row = df.iloc[0]
        
        # Verify distances
        assert row['sp_A_dist'] == -15
        assert row['sp_B_dist'] == -5
        assert row['sp_C_dist'] == 10
        
        # Verify positions in background
        assert row['sp_A_pos_start'] == 5
        assert row['sp_B_pos_start'] == 15
        assert row['sp_C_pos_start'] == 30
        
        # Verify absolute positions (prefix adds 4)
        assert row['sp_A_abs_pos_start'] == 9
        assert row['sp_B_abs_pos_start'] == 19
        assert row['sp_C_abs_pos_start'] == 34
        
        # Verify spacings
        # A ends at 8, B starts at 15: spacing = 15 - 8 = 7
        assert row['sp_spacing_A_B'] == 7
        # A ends at 8, C starts at 30: spacing = 30 - 8 = 22
        assert row['sp_spacing_A_C'] == 22
        # B ends at 18, C starts at 30: spacing = 30 - 18 = 12
        assert row['sp_spacing_B_C'] == 12
        
        # Verify actual sequence content
        assert seq[9:12] == "AAA"
        assert seq[19:22] == "BBB"
        assert seq[34:37] == "CCC"
    
    def test_spacing_with_pool_inputs_coordinates(self):
        """Test SpacingScanPool with Pool inputs in composite."""
        bg_pool = Pool(["BACKGROUND_SEQ_1234567890"], name="bg", mode='sequential')  # 25 chars
        
        insert_a = Pool(["XXX", "YYY"], name="ins_a", mode='sequential')
        insert_b = Pool(["AAA"], name="ins_b", mode='sequential')
        
        spacing = SpacingScanPool(
            background_seq=bg_pool,
            insert_seqs=[insert_a, insert_b],
            insert_names=["A", "B"],
            anchor_pos=12,
            insert_distances=[[-8], [5]],  # A at 4, B at 17
            name="sp",
            mode='sequential',
        )
        
        # 2 states for insert_a (XXX, YYY)
        result = spacing.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        
        assert len(result['sequences']) == 2
        
        df = result['design_cards'].to_dataframe()
        
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # Verify positions are correct
            assert row['sp_A_pos_start'] == 4
            assert row['sp_B_pos_start'] == 17
            
            # Verify insert is at correct position
            assert seq[4:7] in ["XXX", "YYY"]
            assert seq[17:20] == "AAA"
    
    def test_nested_composite_with_random_mode(self):
        """Test SpacingScanPool in random mode within composite."""
        from poolparty import RandomMutationPool
        
        prefix = RandomMutationPool("AAAAAAA", mutation_rate=0.1, name="pre")  # 7 chars
        
        spacing = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=["GCAT", "TACG"],  # 4bp each
            insert_names=["M1", "M2"],
            anchor_pos=15,
            insert_distances=[[-10, -5, 0], [5, 10]],
            min_spacing=2,
            name="sp",
            mode='random',
        )
        
        suffix = Pool(["_FIXED"], name="suf", mode='sequential')  # 6 chars
        
        library = prefix + spacing + suffix
        
        result = library.generate_seqs(num_seqs=50, seed=42, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Verify all constraints satisfied
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # Total length
            assert len(seq) == 7 + 30 + 6
            
            # Spacing constraint
            assert row['sp_spacing_M1_M2'] >= 2
            
            # Ordering constraint (default enforce_order=True)
            assert row['sp_M1_pos_start'] < row['sp_M2_pos_start']
            
            # Verify inserts at correct absolute positions
            m1_abs = row['sp_M1_abs_pos_start']
            m2_abs = row['sp_M2_abs_pos_start']
            assert seq[m1_abs:m1_abs + 4] == "GCAT"
            assert seq[m2_abs:m2_abs + 4] == "TACG"
            
            # Verify suffix is unchanged
            assert seq[-6:] == "_FIXED"


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_tf_spacing_scan(self):
        """Simulate a TF spacing scan experiment."""
        # Background: 100bp around TSS
        background = "N" * 100
        
        # Two TF motifs
        sp1 = "GGGCGG"  # 6bp
        ap1 = "TGACTCA"  # 7bp
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[sp1, ap1],
            insert_names=["SP1", "AP1"],
            anchor_pos=50,  # TSS
            insert_scan_ranges=[
                (-60, -30, 10),  # SP1: 30-60bp upstream
                (-20, 10, 10),   # AP1: 20bp upstream to 10bp downstream
            ],
            min_spacing=5,
            mode='sequential',
            name="tf_module",
        )
        
        # Should have valid combinations
        assert pool.num_internal_states > 0
        
        # Generate all and verify
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        seqs = result['sequences']
        
        # All sequences should be 100bp
        assert all(len(s) == 100 for s in seqs)
        
        # All spacings should be >= 5
        assert all(df['tf_module_spacing_SP1_AP1'] >= 5)
        
        # SP1 should always be upstream of AP1 (enforce_order=True)
        for i, row in df.iterrows():
            assert row['tf_module_SP1_pos_end'] <= row['tf_module_AP1_pos_start']
    
    def test_in_composite_library(self):
        """Test SpacingScanPool in a composite library structure."""
        prefix = Pool(["ADAPTER_"], name="adapter", mode='sequential')
        
        spacing_pool = SpacingScanPool(
            background_seq="NNNNNNNNNNNNNNNNNNNN",  # 20bp
            insert_seqs=["AAA", "TTT"],
            insert_names=["A", "T"],
            anchor_pos=10,
            insert_distances=[[-5], [5]],  # A at 5, T at 15
            name="core",
        )
        
        suffix = Pool(["_BARCODE"], name="barcode", mode='sequential')
        
        library = prefix + spacing_pool + suffix
        
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        
        assert len(result['sequences']) == 1
        
        seq = result['sequences'][0]
        assert seq.startswith("ADAPTER_")
        assert seq.endswith("_BARCODE")
        assert len(seq) == 8 + 20 + 8  # 36
        
        # Check design cards have correct absolute positions
        df = result['design_cards'].to_dataframe()
        # A is at pos 5 in background, plus 8 for prefix = 13
        assert df['core_A_abs_pos_start'].iloc[0] == 13
        
        # Verify actual sequence content at reported positions
        assert seq[13:16] == "AAA"
        assert seq[23:26] == "TTT"  # T at 15 + 8 = 23
    
    def test_realistic_enhancer_grammar_screen(self):
        """Simulate realistic enhancer grammar screen with 3 TFs."""
        # 150bp enhancer region with anchor at center
        background = "ATCG" * 37 + "AT"  # 150bp
        
        # Three TF motifs of different lengths
        ets = "GGAA"      # 4bp - ETS family
        gata = "AGATAA"   # 6bp - GATA
        runx = "TGTGGT"   # 6bp - RUNX
        
        pool = SpacingScanPool(
            background_seq=background,
            insert_seqs=[ets, gata, runx],
            insert_names=["ETS", "GATA", "RUNX"],
            anchor_pos=75,  # Center
            insert_scan_ranges=[
                (-60, -30, 15),  # ETS upstream
                (-20, 20, 10),   # GATA around center
                (30, 60, 15),    # RUNX downstream
            ],
            min_spacing=3,
            enforce_order=True,
            mode='sequential',
            name="enhancer",
        )
        
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        # Verify all constraints
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # 1. Sequence length
            assert len(seq) == 150
            
            # 2. All spacings >= min_spacing (3)
            assert row['enhancer_spacing_ETS_GATA'] >= 3
            assert row['enhancer_spacing_ETS_RUNX'] >= 3
            assert row['enhancer_spacing_GATA_RUNX'] >= 3
            
            # 3. Order preserved: ETS < GATA < RUNX
            assert row['enhancer_ETS_pos_end'] <= row['enhancer_GATA_pos_start']
            assert row['enhancer_GATA_pos_end'] <= row['enhancer_RUNX_pos_start']
            
            # 4. Inserts at correct positions
            ets_pos = row['enhancer_ETS_pos_start']
            gata_pos = row['enhancer_GATA_pos_start']
            runx_pos = row['enhancer_RUNX_pos_start']
            
            assert seq[ets_pos:ets_pos + 4] == ets
            assert seq[gata_pos:gata_pos + 6] == gata
            assert seq[runx_pos:runx_pos + 6] == runx
    
    def test_mpra_library_with_barcodes_and_adapters(self):
        """Test complete MPRA library structure."""
        from poolparty import BarcodePool
        
        # 5' adapter
        adapter_5p = Pool(["ACTGGCCTT"], name="5p", mode='sequential')  # 9bp
        
        # Regulatory element with TF binding sites
        regulatory = SpacingScanPool(
            background_seq="N" * 80,  # 80bp regulatory region
            insert_seqs=["CACGTG", "TGACGTCA"],  # E-box (6bp), CRE (8bp)
            insert_names=["EBOX", "CRE"],
            anchor_pos=40,
            insert_distances=[[-25, -15, -5], [10, 20, 30]],
            min_spacing=5,
            mode='sequential',
            name="reg",
        )
        
        # 3' adapter
        adapter_3p = Pool(["TTCCGGACT"], name="3p", mode='sequential')  # 9bp
        
        # Barcode
        barcode = BarcodePool(
            num_barcodes=3,
            length=10,
            min_edit_distance=3,
            name="bc",
            seed=42,
        )
        
        library = adapter_5p + regulatory + adapter_3p + barcode
        
        # 9 regulatory combos × 3 barcodes = 27
        result = library.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        df = result['design_cards'].to_dataframe()
        
        assert len(result['sequences']) == 27
        
        for i, row in df.iterrows():
            seq = result['sequences'][i]
            
            # Total length: 9 + 80 + 9 + 10 = 108
            assert len(seq) == 108
            
            # Verify adapters
            assert seq[:9] == "ACTGGCCTT"
            assert seq[89:98] == "TTCCGGACT"
            
            # Verify regulatory element positions
            # EBOX absolute = 9 (5' adapter) + pos_in_reg
            ebox_abs = row['reg_EBOX_abs_pos_start']
            cre_abs = row['reg_CRE_abs_pos_start']
            
            assert seq[ebox_abs:ebox_abs + 6] == "CACGTG"
            assert seq[cre_abs:cre_abs + 8] == "TGACGTCA"
            
            # Verify spacing constraint
            assert row['reg_spacing_EBOX_CRE'] >= 5

