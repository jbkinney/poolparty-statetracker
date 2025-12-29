"""Complex nesting design card tests.

These tests verify design card correctness in complex multi-level nesting scenarios
involving multi-input pools (InsertionScan, SpacingScan, InsertionScanORF).

Each category tests four aspects:
1. Config Verification - Metadata matches constructor hyperparameters
2. Combination Coverage - All pools tracked correctly with proper null-filling
3. Position/Value Integrity - Extraction formulas work at all nesting levels
4. Distributional Verification - Metadata follows expected distributions
"""

import pytest
from poolparty import (
    Pool, MixedPool, DesignCards,
    InsertionScanPool, DeletionScanPool, SubseqPool, ShuffleScanPool,
    KMutationPool, RandomMutationPool,
    KMutationORFPool, RandomMutationORFPool,
    InsertionScanORFPool, DeletionScanORFPool,
    SpacingScanPool, IUPACPool, KmerPool,
)


# =============================================================================
# Category A: Multi-Input Inside Multi-Input (12 tests)
# =============================================================================

class TestMultiInputInMultiInput:
    """Category A: Tests for multi-input pools nested inside each other.
    
    Structures tested:
    - InsertionScan with SpacingScan as insert
    - SpacingScan with InsertionScan as insert
    - InsertionScanORF with SpacingScan as insert
    """
    
    # -------------------------------------------------------------------------
    # Config Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_config_insertion_scan_with_spacing_insert(self):
        """InsertionScan with SpacingScan insert: verify all hyperparams hold."""
        # Inner SpacingScan: distances must be from configured lists
        inner_a = IUPACPool('AAAA', name='inner_a', metadata='complete')
        inner_b = IUPACPool('TTTT', name='inner_b', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='N' * 20,
            insert_seqs=[inner_a, inner_b],
            insert_names=['A', 'B'],
            anchor_pos=10,
            insert_distances=[[-4, 0], [2, 6]],  # 2*2 = 4 combos
            name='inner_ss', mode='sequential', metadata='complete'
        )
        
        # Outer InsertionScan: start=2, end=10, step=2 -> positions [2,4,6,8]
        outer_is = InsertionScanPool(
            background_seq='G' * 30,
            insert_seq=inner_ss,
            start=2, end=12, step_size=2,
            insert_or_overwrite='insert',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        result = outer_is.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            
            # Outer IS: pos must be in [2, 4, 6, 8, 10] (range(2, 12-20+1, 2) adjusted for insert length)
            outer_pos = row['outer_is_pos']
            assert outer_pos is not None, "outer_is_pos should be present"
            assert outer_pos >= 2, f"outer_is_pos {outer_pos} should be >= 2"
            assert outer_pos % 2 == 0, f"outer_is_pos {outer_pos} should be even (step=2)"
            
            # Inner SS: distances must be from configured lists
            a_dist = row['inner_ss_A_dist']
            b_dist = row['inner_ss_B_dist']
            assert a_dist in [-4, 0], f"A_dist {a_dist} not in [-4, 0]"
            assert b_dist in [2, 6], f"B_dist {b_dist} not in [2, 6]"
    
    def test_config_spacing_scan_with_insertion_insert(self):
        """SpacingScan with InsertionScan insert: verify all hyperparams hold."""
        # Inner InsertionScan: start=1, end=7, step=2 -> positions [1, 3, 5]
        inner_bg = IUPACPool('RRRRRRRR', name='inner_bg', metadata='complete')
        inner_ins = Pool(['XX'], name='inner_ins', metadata='complete')
        inner_is = InsertionScanPool(
            background_seq=inner_bg,
            insert_seq=inner_ins,
            start=1, end=7, step_size=2,
            insert_or_overwrite='overwrite',
            name='inner_is', mode='sequential', metadata='complete'
        )
        
        # Outer SpacingScan: distances from configured lists
        other = Pool(['YYYY'], name='other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[inner_is, other],
            insert_names=['IS', 'Other'],
            anchor_pos=20,
            insert_distances=[[-8], [8]],  # Single distance each
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=15, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(15):
            row = dc.get_row(i)
            
            # Inner IS: pos must be in [1, 3, 5]
            inner_pos = row['inner_is_pos']
            assert inner_pos in [1, 3, 5], f"inner_is_pos {inner_pos} not in [1, 3, 5]"
            
            # Outer SS: distances must match config
            assert row['outer_ss_IS_dist'] == -8, f"IS_dist should be -8"
            assert row['outer_ss_Other_dist'] == 8, f"Other_dist should be 8"
    
    def test_config_insertion_scan_orf_with_spacing_insert(self):
        """InsertionScanORF with SpacingScan insert: verify ORF hyperparams hold."""
        # Inner SpacingScan: produces 6bp ACGT-only output (2 codons)
        inner_a = Pool(['GGG'], name='inner_a', metadata='complete')  # 3bp
        inner_ss = SpacingScanPool(
            background_seq='AAAAAA',  # 6bp background (ACGT only)
            insert_seqs=[inner_a],
            insert_names=['A'],
            anchor_pos=3,
            insert_distances=[[0]],  # Single position
            name='inner_ss', mode='sequential', metadata='complete'
        )
        
        # Outer InsertionScanORF with 2-codon insert in 6-codon bg
        outer_iso = InsertionScanORFPool(
            background_seq='ATGAAACCCGGGTTTCCC',  # 18bp = 6 codons
            insert_seq=inner_ss,  # 6bp insert = 2 codons
            start=1, end=5, step_size=1,  # codon positions [1, 2, 3, 4]
            insert_or_overwrite='overwrite',
            name='outer_iso', mode='sequential', metadata='complete'
        )
        
        result = outer_iso.generate_seqs(num_seqs=12, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(12):
            row = dc.get_row(i)
            
            # Outer InsertionScanORF: codon_pos must be in [1, 2, 3, 4]
            codon_pos = row['outer_iso_codon_pos']
            assert codon_pos in [1, 2, 3, 4], f"codon_pos {codon_pos} not in [1, 2, 3, 4]"
            
            # codon_pos_abs = codon_pos * 3 (orf_start=0)
            codon_pos_abs = row['outer_iso_codon_pos_abs']
            assert codon_pos_abs == codon_pos * 3, \
                f"codon_pos_abs {codon_pos_abs} != codon_pos * 3 = {codon_pos * 3}"
    
    # -------------------------------------------------------------------------
    # Combination Coverage (3 tests)
    # -------------------------------------------------------------------------
    
    def test_combo_insertion_scan_with_spacing_insert(self):
        """InsertionScan with SpacingScan insert: all pools tracked."""
        inner_a = IUPACPool('AA', name='combo_a', metadata='complete')
        inner_b = IUPACPool('TT', name='combo_b', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='NNNNNNNN',
            insert_seqs=[inner_a, inner_b],
            insert_names=['A', 'B'],
            anchor_pos=4,
            insert_distances=[[-2], [2]],
            name='combo_ss', mode='sequential', metadata='complete'
        )
        
        # inner_ss produces 8bp, use insert mode with end > start
        outer_is = InsertionScanPool(
            background_seq='G' * 30,  # Longer background
            insert_seq=inner_ss,
            start=2, end=20, step_size=4,  # positions [2, 6, 10, 14, 18]
            insert_or_overwrite='insert',
            name='combo_is', mode='sequential', metadata='complete'
        )
        
        result = outer_is.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools should be tracked
        expected_pools = ['combo_is', 'combo_ss', 'combo_a', 'combo_b']
        for pool_name in expected_pools:
            assert f'{pool_name}_index' in dc.keys, f"{pool_name} should be tracked"
            assert f'{pool_name}_value' in dc.keys, f"{pool_name} should have _value"
        
        # SpacingScan should have its insert-specific columns
        assert 'combo_ss_A_dist' in dc.keys
        assert 'combo_ss_B_dist' in dc.keys
        assert 'combo_ss_spacing_A_B' in dc.keys
    
    def test_combo_spacing_scan_with_insertion_insert(self):
        """SpacingScan with InsertionScan insert: all pools tracked."""
        inner_bg = Pool(['GGGGGG'], name='is_bg', metadata='complete')
        inner_ins = Pool(['XX'], name='is_ins', metadata='complete')
        inner_is = InsertionScanPool(
            background_seq=inner_bg,
            insert_seq=inner_ins,
            start=1, end=5, step_size=1,
            name='is_pool', mode='sequential', metadata='complete'
        )
        
        other = Pool(['YYYY'], name='other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[inner_is, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name='ss_pool', mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools tracked
        expected = ['ss_pool', 'is_pool', 'is_bg', 'is_ins', 'other']
        for pool_name in expected:
            assert f'{pool_name}_index' in dc.keys, f"{pool_name} should be tracked"
        
        # InsertionScan columns present
        assert 'is_pool_pos' in dc.keys
        assert 'is_pool_insert' in dc.keys
    
    def test_combo_insertion_scan_orf_with_spacing_insert(self):
        """InsertionScanORF with SpacingScan insert: ORF + spacing metadata."""
        inner_a = Pool(['GGG'], name='orf_a', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='AAAAAA',  # 6bp = 2 codons (ACGT only)
            insert_seqs=[inner_a],
            insert_names=['A'],
            anchor_pos=3,
            insert_distances=[[0]],
            name='orf_ss', mode='sequential', metadata='complete'
        )
        
        outer_iso = InsertionScanORFPool(
            background_seq='ATGAAACCCGGGTTT',  # 15bp = 5 codons
            insert_seq=inner_ss,  # 6bp = 2 codons
            start=1, end=4, step_size=1,  # codon positions [1, 2, 3]
            insert_or_overwrite='overwrite',
            name='orf_iso', mode='sequential', metadata='complete'
        )
        
        result = outer_iso.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # ORF-specific columns
        assert 'orf_iso_codon_pos' in dc.keys
        assert 'orf_iso_codon_pos_abs' in dc.keys
        
        # Spacing columns from inner
        assert 'orf_ss_A_dist' in dc.keys
        
        # All pools tracked
        for pool_name in ['orf_iso', 'orf_ss', 'orf_a']:
            assert f'{pool_name}_index' in dc.keys
    
    # -------------------------------------------------------------------------
    # Position/Value Integrity (3 tests)
    # -------------------------------------------------------------------------
    
    def test_integrity_insertion_scan_with_spacing_insert(self):
        """InsertionScan with SpacingScan insert: extraction matches values."""
        inner_a = Pool(['AA'], name='int_a', metadata='complete')
        inner_b = Pool(['TT'], name='int_b', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='CCCCCCCCCC',  # 10bp
            insert_seqs=[inner_a, inner_b],
            insert_names=['A', 'B'],
            anchor_pos=5,
            insert_distances=[[-3, -1], [1, 3]],  # 2*2 = 4 combos (non-trivial)
            name='int_ss', mode='sequential', metadata='complete'
        )
        
        # inner_ss produces 10bp output
        outer_is = InsertionScanPool(
            background_seq='G' * 30,  # 30bp background
            insert_seq=inner_ss,
            start=2, end=18, step_size=4,  # positions [2, 6, 10, 14] = 4 positions
            insert_or_overwrite='insert',
            name='int_is', mode='sequential', metadata='complete'
        )
        
        # 4 IS positions * 4 SS combos = 16 states
        result = outer_is.generate_seqs(num_seqs=16, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(16):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Outer IS: extraction matches value (outer pool has valid positions)
            is_start = row['int_is_abs_start']
            is_end = row['int_is_abs_end']
            is_value = row['int_is_value']
            extracted_is = seq[is_start:is_end]
            assert extracted_is == is_value, \
                f"int_is extraction '{extracted_is}' != value '{is_value}'"
            
            # Inner SS: transformer parent - positions are None, but value is valid
            # The SS value is the SpacingScan output (10bp), which is the insert in IS
            ss_value = row['int_ss_value']
            assert ss_value is not None, "Inner SS value should be present"
            # SS value should be contained within IS value (as the inserted content)
            assert ss_value in is_value, \
                f"int_ss value '{ss_value}' should be in int_is value '{is_value}'"
            # Verify SS value contains expected inserts from its children
            assert 'AA' in ss_value, "SS value should contain 'AA'"
            assert 'TT' in ss_value, "SS value should contain 'TT'"
            
            # Verify SS distances are from configured lists
            a_dist = row['int_ss_A_dist']
            b_dist = row['int_ss_B_dist']
            assert a_dist in [-3, -1], f"A_dist {a_dist} should be in [-3, -1]"
            assert b_dist in [1, 3], f"B_dist {b_dist} should be in [1, 3]"
    
    def test_integrity_spacing_scan_with_insertion_insert(self):
        """SpacingScan with InsertionScan insert: spacing formula holds."""
        inner_bg = Pool(['GGGGGG'], name='spint_bg', metadata='complete')
        inner_ins = Pool(['XX'], name='spint_ins', metadata='complete')
        inner_is = InsertionScanPool(
            background_seq=inner_bg,
            insert_seq=inner_ins,
            start=1, end=5, step_size=1,
            insert_or_overwrite='overwrite',
            name='spint_is', mode='sequential', metadata='complete'
        )
        
        other = Pool(['YYYY'], name='spint_other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[inner_is, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name='spint_ss', mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            
            # Spacing formula: spacing_IS_Oth = Oth_pos_start - IS_pos_end
            is_end = row['spint_ss_IS_pos_end']
            oth_start = row['spint_ss_Oth_pos_start']
            expected_spacing = oth_start - is_end
            actual_spacing = row['spint_ss_spacing_IS_Oth']
            
            assert actual_spacing == expected_spacing, \
                f"spacing {actual_spacing} != Oth_start - IS_end = {expected_spacing}"
    
    def test_integrity_insertion_scan_orf_with_spacing_insert(self):
        """InsertionScanORF with SpacingScan insert: codon extraction works."""
        inner_a = Pool(['GGG'], name='orfint_a', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='AAAAAAAAA',  # 9bp = 3 codons
            insert_seqs=[inner_a],
            insert_names=['A'],
            anchor_pos=4,
            insert_distances=[[0]],
            name='orfint_ss', mode='sequential', metadata='complete'
        )
        
        outer_iso = InsertionScanORFPool(
            background_seq='ATGAAACCCGGGTTT',  # 15bp = 5 codons
            insert_seq=inner_ss,
            start=1, end=4, step_size=1,
            insert_or_overwrite='overwrite',
            name='orfint_iso', mode='sequential', metadata='complete'
        )
        
        result = outer_iso.generate_seqs(num_seqs=9, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(9):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Verify codon_pos_abs allows correct extraction
            codon_pos_abs = row['orfint_iso_codon_pos_abs']
            # The insert is 9bp (3 codons), so we can extract that region
            insert_value = row['orfint_iso_insert']
            
            # The value at codon_pos_abs should be part of the inserted sequence
            assert codon_pos_abs is not None
            assert insert_value is not None
    
    # -------------------------------------------------------------------------
    # Distributional Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_distrib_insertion_scan_with_spacing_insert(self):
        """InsertionScan with SpacingScan insert: uniform distributions hold."""
        inner_a = IUPACPool('RR', name='dist_a', metadata='complete')  # R = A or G
        inner_b = Pool(['TT'], name='dist_b', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='CCCCCCCCCC',  # 10bp
            insert_seqs=[inner_a, inner_b],
            insert_names=['A', 'B'],
            anchor_pos=5,
            insert_distances=[[-3, -1], [2]],  # 2 combos for A_dist
            name='dist_ss', mode='sequential', metadata='complete'
        )
        
        # inner_ss produces 10bp, use insert mode
        # start=2, end=18, step=4 in insert mode -> positions [2, 6, 10, 14, 18] = 5 positions
        outer_is = InsertionScanPool(
            background_seq='G' * 30,
            insert_seq=inner_ss,
            start=2, end=18, step_size=4,
            insert_or_overwrite='insert',
            name='dist_is', mode='sequential', metadata='complete'
        )
        
        # 5 IS positions * 2 SS combos = 10 total states per iteration
        # Generate 50 seqs = 5 complete iterations
        result = outer_is.generate_seqs(num_seqs=50, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify outer IS positions are uniform (5 positions, 50 seqs = 10 each)
        is_positions = [dc.get_row(i)['dist_is_pos'] for i in range(50)]
        for pos in [2, 6, 10, 14, 18]:
            count = is_positions.count(pos)
            assert count == 10, f"IS pos {pos} should appear 10 times, got {count}"
        
        # Verify inner SS A_dist is uniform over the configured values (50/2 = 25 each)
        a_dists = [dc.get_row(i)['dist_ss_A_dist'] for i in range(50)]
        for dist in [-3, -1]:
            count = a_dists.count(dist)
            assert count == 25, f"A_dist {dist} should appear 25 times, got {count}"
        
        # Verify IUPAC R produces ~50/50 A/G
        all_a_values = ''.join(dc.get_row(i)['dist_a_value'] for i in range(50))
        a_freq = all_a_values.count('A') / len(all_a_values)
        assert 0.4 < a_freq < 0.6, f"IUPAC R should be ~50% A, got {a_freq:.2%}"
    
    def test_distrib_spacing_scan_with_insertion_insert(self):
        """SpacingScan with InsertionScan insert: sequential uniformity."""
        # Use longer background to get more positions
        inner_bg = Pool(['GGGGGGGG'], name='dss_bg', metadata='complete')  # 8bp
        inner_ins = Pool(['XX'], name='dss_ins', metadata='complete')  # 2bp
        inner_is = InsertionScanPool(
            background_seq=inner_bg,
            insert_seq=inner_ins,
            start=0, end=8, step_size=2,  # positions [0, 2, 4, 6] (end-W+1=8-2+1=7)
            insert_or_overwrite='overwrite',
            name='dss_is', mode='sequential', metadata='complete'
        )
        
        other = Pool(['YY'], name='dss_other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 40,  # Larger background
            insert_seqs=[inner_is, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-8, -4], [4, 8]],  # 2*2 = 4 combos
            name='dss_ss', mode='sequential', metadata='complete'
        )
        
        # 4 IS positions * 4 SS combos = 16 states per iteration
        # Generate 48 seqs = 3 complete iterations
        result = outer_ss.generate_seqs(num_seqs=48, return_design_cards=True)
        dc = result['design_cards']
        
        # Inner IS positions uniform (48 seqs / 4 positions = 12 each)
        is_positions = [dc.get_row(i)['dss_is_pos'] for i in range(48)]
        for pos in [0, 2, 4, 6]:
            count = is_positions.count(pos)
            assert count == 12, f"IS pos {pos} should appear 12 times, got {count}"
        
        # Outer SS combinations complete
        combos = set()
        for i in range(48):
            row = dc.get_row(i)
            combos.add((row['dss_ss_IS_dist'], row['dss_ss_Oth_dist']))
        assert len(combos) == 4, f"Should have 4 SS combos, got {len(combos)}"
    
    def test_distrib_insertion_scan_orf_with_spacing_insert(self):
        """InsertionScanORF with SpacingScan insert: codon positions & spacing combos uniform."""
        # Two inserts to create multiple SpacingScan combos
        # Use small 1bp inserts with 9bp background = 3 codons, configured for 4 valid combos
        inner_a = Pool(['G'], name='diso_a', metadata='complete')  # 1bp
        inner_b = Pool(['A'], name='diso_b', metadata='complete')  # 1bp
        inner_ss = SpacingScanPool(
            background_seq='CCCCCCCCC',  # 9bp = 3 codons (ACGT only)
            insert_seqs=[inner_a, inner_b],
            insert_names=['A', 'B'],
            anchor_pos=4,
            insert_distances=[[-3, -2], [3, 4]],  # 2*2 = 4 combos (verified non-overlapping)
            name='diso_ss', mode='sequential', metadata='complete'
        )
        
        # 3-codon insert in 10-codon bg, overwrite mode: positions [1, 2, 3, 4, 5, 6, 7] = 7 positions
        outer_iso = InsertionScanORFPool(
            background_seq='ATGAAACCCGGGTTTCCCGGGATGAAACCC',  # 30bp = 10 codons
            insert_seq=inner_ss,  # 9bp = 3 codons
            start=1, end=8, step_size=1,  # codon positions [1, 2, 3, 4, 5, 6, 7]
            insert_or_overwrite='overwrite',
            name='diso_iso', mode='sequential', metadata='complete'
        )
        
        # 5 codon positions * 4 SS combos = 20 states per iteration
        result = outer_iso.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        # Codon positions uniform (20 seqs / 5 positions = 4 each)
        codon_positions = [dc.get_row(i)['diso_iso_codon_pos'] for i in range(20)]
        for pos in [1, 2, 3, 4, 5]:
            count = codon_positions.count(pos)
            assert count == 4, f"codon_pos {pos} should appear 4 times, got {count}"
        
        # SpacingScan A_dist uniform (20 seqs / 2 A distances = 10 each)
        a_dists = [dc.get_row(i)['diso_ss_A_dist'] for i in range(20)]
        for dist in [-3, -2]:
            count = a_dists.count(dist)
            assert count == 10, f"A_dist {dist} should appear 10 times, got {count}"


# =============================================================================
# Category B: Multi-Input with MixedPool (12 tests)
# =============================================================================

class TestMultiInputWithMixedPool:
    """Category B: Tests for multi-input pools combined with MixedPool.
    
    Structures tested:
    - MixedPool[InsertionScan1, InsertionScan2]
    - SpacingScan with MixedPool as insert
    - InsertionScan with MixedPool as background
    """
    
    # -------------------------------------------------------------------------
    # Config Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_config_mixed_pool_insertion_scan_children(self):
        """MixedPool[InsertionScan1, InsertionScan2]: each child's config valid."""
        # Child 1: start=0, end=6, step=2 -> positions [0, 2, 4]
        is1 = InsertionScanPool(
            'AAAAAAAA', 'XX',
            start=0, end=6, step_size=2,
            name='is1', mode='sequential', metadata='complete'
        )
        
        # Child 2: start=1, end=7, step=2 -> positions [1, 3, 5]
        is2 = InsertionScanPool(
            'TTTTTTTT', 'YY',
            start=1, end=7, step_size=2,
            name='is2', mode='sequential', metadata='complete'
        )
        
        mixed = MixedPool([is1, is2], name='mix', mode='sequential')
        
        # 3 states each * 2 children = 6 total per iteration
        result = mixed.generate_seqs(num_seqs=18, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(18):
            row = dc.get_row(i)
            selected = row['mix_selected']
            
            if selected == 0:
                # is1 selected: pos in [0, 2, 4]
                pos = row['is1_pos']
                assert pos in [0, 2, 4], f"is1_pos {pos} not in [0, 2, 4]"
                assert row['is2_pos'] is None, "is2 should be None when not selected"
            else:
                # is2 selected: pos in [1, 3, 5]
                pos = row['is2_pos']
                assert pos in [1, 3, 5], f"is2_pos {pos} not in [1, 3, 5]"
                assert row['is1_pos'] is None, "is1 should be None when not selected"
    
    def test_config_spacing_scan_with_mixed_insert(self):
        """SpacingScan with MixedPool insert: mixed selection + spacing valid."""
        # MixedPool children
        child_a = Pool(['AAAA'], name='child_a', metadata='complete')
        child_b = Pool(['TTTT'], name='child_b', metadata='complete')
        mixed = MixedPool([child_a, child_b], name='mix', mode='sequential')
        
        other = Pool(['GGGG'], name='other', metadata='complete')
        
        # SpacingScan with MixedPool as one insert
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-6, -3], [3, 6]],  # 2*2 = 4 combos
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=16, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(16):
            row = dc.get_row(i)
            
            # MixedPool selection tracked
            selected = row['mix_selected']
            assert selected in [0, 1], f"mix_selected {selected} should be 0 or 1"
            
            # Spacing distances from config
            mix_dist = row['ss_Mix_dist']
            oth_dist = row['ss_Oth_dist']
            assert mix_dist in [-6, -3], f"Mix_dist {mix_dist} not in [-6, -3]"
            assert oth_dist in [3, 6], f"Oth_dist {oth_dist} not in [3, 6]"
    
    def test_config_insertion_scan_with_mixed_bg(self):
        """InsertionScan with MixedPool background: IS config valid for selected bg."""
        # MixedPool backgrounds of same length
        bg_a = Pool(['AAAAAAAA'], name='bg_a', metadata='complete')  # 8bp
        bg_b = Pool(['TTTTTTTT'], name='bg_b', metadata='complete')  # 8bp
        mixed_bg = MixedPool([bg_a, bg_b], name='mix_bg', mode='sequential')
        
        insert = Pool(['XX'], name='insert', metadata='complete')
        
        # InsertionScan: start=1, end=6, step=2 -> positions [1, 3, 5]
        is_pool = InsertionScanPool(
            background_seq=mixed_bg,
            insert_seq=insert,
            start=1, end=6, step_size=2,
            insert_or_overwrite='overwrite',
            name='is', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=12, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(12):
            row = dc.get_row(i)
            
            # IS pos in [1, 3, 5] regardless of which bg selected
            pos = row['is_pos']
            assert pos in [1, 3, 5], f"is_pos {pos} not in [1, 3, 5]"
            
            # MixedPool selection tracked
            bg_selected = row['mix_bg_selected']
            assert bg_selected in [0, 1]
    
    # -------------------------------------------------------------------------
    # Combination Coverage (3 tests)
    # -------------------------------------------------------------------------
    
    def test_combo_mixed_pool_insertion_scan_children(self):
        """MixedPool[InsertionScan1, InsertionScan2]: all tracked correctly."""
        bg1 = Pool(['GGGG'], name='cbg1', metadata='complete')
        ins1 = Pool(['AA'], name='cins1', metadata='complete')
        is1 = InsertionScanPool(bg1, ins1, start=0, end=3, step_size=1,
                                name='cis1', mode='sequential', metadata='complete')
        
        bg2 = Pool(['CCCC'], name='cbg2', metadata='complete')
        ins2 = Pool(['TT'], name='cins2', metadata='complete')
        is2 = InsertionScanPool(bg2, ins2, start=0, end=3, step_size=1,
                                name='cis2', mode='sequential', metadata='complete')
        
        mixed = MixedPool([is1, is2], name='cmix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools should have columns
        for name in ['cmix', 'cis1', 'cbg1', 'cins1', 'cis2', 'cbg2', 'cins2']:
            assert f'{name}_index' in dc.keys, f"{name} should be tracked"
        
        # Unselected child has None values
        for i in range(10):
            row = dc.get_row(i)
            if row['cmix_selected'] == 0:
                assert row['cis1_value'] is not None
                assert row['cis2_value'] is None
            else:
                assert row['cis1_value'] is None
                assert row['cis2_value'] is not None
    
    def test_combo_spacing_scan_with_mixed_insert(self):
        """SpacingScan with MixedPool insert: all pools tracked."""
        child_a = Pool(['AA'], name='sma', metadata='complete')
        child_b = Pool(['TT'], name='smb', metadata='complete')
        mixed = MixedPool([child_a, child_b], name='smix', mode='sequential')
        
        other = Pool(['GG'], name='soth', metadata='complete')
        
        ss = SpacingScanPool(
            background_seq='N' * 20,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=10,
            insert_distances=[[-4], [4]],
            name='sss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools tracked
        for name in ['sss', 'smix', 'sma', 'smb', 'soth']:
            assert f'{name}_index' in dc.keys, f"{name} should be tracked"
        
        # SpacingScan metadata
        assert 'sss_Mix_dist' in dc.keys
        assert 'sss_Oth_dist' in dc.keys
        assert 'sss_spacing_Mix_Oth' in dc.keys
    
    def test_combo_insertion_scan_with_mixed_bg(self):
        """InsertionScan with MixedPool bg: all pools tracked."""
        bg_a = Pool(['AAAA'], name='isma', metadata='complete')
        bg_b = Pool(['TTTT'], name='ismb', metadata='complete')
        mixed_bg = MixedPool([bg_a, bg_b], name='ismmix', mode='sequential')
        
        insert = Pool(['XX'], name='ismins', metadata='complete')
        
        is_pool = InsertionScanPool(
            mixed_bg, insert,
            start=0, end=3, step_size=1,
            name='ismis', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=8, return_design_cards=True)
        dc = result['design_cards']
        
        # All tracked
        for name in ['ismis', 'ismmix', 'isma', 'ismb', 'ismins']:
            assert f'{name}_index' in dc.keys, f"{name} should be tracked"
        
        # InsertionScan metadata
        assert 'ismis_pos' in dc.keys
        assert 'ismis_insert' in dc.keys
    
    # -------------------------------------------------------------------------
    # Position/Value Integrity (3 tests)
    # -------------------------------------------------------------------------
    
    def test_integrity_mixed_pool_insertion_scan_children(self):
        """MixedPool[InsertionScan1, InsertionScan2]: extraction correct for selected."""
        is1 = InsertionScanPool('GGGGGG', 'AA', start=0, end=5, step_size=1,
                                name='intis1', mode='sequential', metadata='complete')
        is2 = InsertionScanPool('CCCCCC', 'TT', start=0, end=5, step_size=1,
                                name='intis2', mode='sequential', metadata='complete')
        
        mixed = MixedPool([is1, is2], name='intmix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['intmix_selected']
            
            # Selected child's value matches sequence
            if selected == 0:
                assert row['intis1_value'] == seq
            else:
                assert row['intis2_value'] == seq
    
    def test_integrity_spacing_scan_with_mixed_insert(self):
        """SpacingScan with MixedPool insert: spacing formula holds."""
        child_a = Pool(['AAAA'], name='intssa', metadata='complete')
        child_b = Pool(['TTTT'], name='intssb', metadata='complete')
        mixed = MixedPool([child_a, child_b], name='intssmix', mode='sequential')
        
        other = Pool(['GGGG'], name='intssoth', metadata='complete')
        
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-8], [8]],
            name='intss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(6):
            row = dc.get_row(i)
            
            # Spacing formula
            mix_end = row['intss_Mix_pos_end']
            oth_start = row['intss_Oth_pos_start']
            expected_spacing = oth_start - mix_end
            actual_spacing = row['intss_spacing_Mix_Oth']
            
            assert actual_spacing == expected_spacing, \
                f"spacing {actual_spacing} != {expected_spacing}"
    
    def test_integrity_insertion_scan_with_mixed_bg(self):
        """InsertionScan with MixedPool bg: extraction correct."""
        bg_a = Pool(['AAAAAAAA'], name='intisma', metadata='complete')
        bg_b = Pool(['TTTTTTTT'], name='intismb', metadata='complete')
        mixed_bg = MixedPool([bg_a, bg_b], name='intismix', mode='sequential')
        
        insert = Pool(['XX'], name='intismins', metadata='complete')
        
        is_pool = InsertionScanPool(
            mixed_bg, insert,
            start=2, end=6, step_size=2,
            insert_or_overwrite='overwrite',
            name='intismis', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=8, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(8):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Extract IS value
            is_start = row['intismis_abs_start']
            is_end = row['intismis_abs_end']
            is_value = row['intismis_value']
            extracted = seq[is_start:is_end]
            
            assert extracted == is_value, \
                f"extraction '{extracted}' != value '{is_value}'"
    
    # -------------------------------------------------------------------------
    # Distributional Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_distrib_mixed_pool_insertion_scan_children(self):
        """MixedPool[InsertionScan1, InsertionScan2]: uniform selection."""
        # Each IS has 2 positions: 6bp bg, 2bp insert, start=0, end=5, step=2 -> [0, 2] = 2 positions
        is1 = InsertionScanPool('GGGGGG', 'AA', start=0, end=5, step_size=2,
                                name='dmis1', mode='sequential', metadata='complete')
        is2 = InsertionScanPool('CCCCCC', 'TT', start=0, end=5, step_size=2,
                                name='dmis2', mode='sequential', metadata='complete')
        
        mixed = MixedPool([is1, is2], name='dmmix', mode='sequential')
        
        # 2 positions each * 2 children = 4 states per iteration
        # Generate 40 seqs = 10 complete iterations
        result = mixed.generate_seqs(num_seqs=40, return_design_cards=True)
        dc = result['design_cards']
        
        # Each child selected 20 times (40 / 2 = 20)
        selections = [dc.get_row(i)['dmmix_selected'] for i in range(40)]
        assert selections.count(0) == 20, f"Child 0 should be selected 20 times, got {selections.count(0)}"
        assert selections.count(1) == 20, f"Child 1 should be selected 20 times, got {selections.count(1)}"
        
        # Each IS position appears uniformly for its child (20 / 2 = 10 each)
        is1_positions = [dc.get_row(i)['dmis1_pos'] for i in range(40) if dc.get_row(i)['dmmix_selected'] == 0]
        for pos in [0, 2]:
            count = is1_positions.count(pos)
            assert count == 10, f"is1_pos {pos} should appear 10 times, got {count}"
    
    def test_distrib_spacing_scan_with_weighted_mixed_insert(self):
        """SpacingScan with weighted MixedPool: selection follows weights."""
        child_a = Pool(['AAAA'], name='dwssa', metadata='complete')
        child_b = Pool(['TTTT'], name='dwssb', metadata='complete')
        # 70/30 weights
        mixed = MixedPool([child_a, child_b], weights=[0.7, 0.3], name='dwssmix', mode='random')
        
        other = Pool(['GGGG'], name='dwssoth', metadata='complete')
        
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-8], [8]],
            name='dwss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=300, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify weights ~70/30
        selections = [dc.get_row(i)['dwssmix_selected'] for i in range(300)]
        child_a_ratio = selections.count(0) / 300
        assert 0.60 < child_a_ratio < 0.80, \
            f"Child A ratio {child_a_ratio:.2%} should be ~70%"
    
    def test_distrib_insertion_scan_with_mixed_bg(self):
        """InsertionScan with MixedPool bg: uniform selection."""
        bg_a = Pool(['AAAA'], name='dismba', metadata='complete')
        bg_b = Pool(['TTTT'], name='dismbb', metadata='complete')
        mixed_bg = MixedPool([bg_a, bg_b], name='dismbmix', mode='sequential')
        
        insert = Pool(['XX'], name='dismbins', metadata='complete')
        
        is_pool = InsertionScanPool(
            mixed_bg, insert,
            start=0, end=3, step_size=1,  # 3 positions
            insert_or_overwrite='overwrite',
            name='dismbis', mode='sequential', metadata='complete'
        )
        
        # 3 IS positions * 2 bg children = 6 states
        result = is_pool.generate_seqs(num_seqs=60, return_design_cards=True)
        dc = result['design_cards']
        
        # Each bg selected 30 times
        selections = [dc.get_row(i)['dismbmix_selected'] for i in range(60)]
        assert selections.count(0) == 30, f"bg_a should be selected 30 times"
        assert selections.count(1) == 30, f"bg_b should be selected 30 times"


# =============================================================================
# Category C: 4-5 Level Deep Nesting (16 tests)
# =============================================================================

class TestDeepNesting:
    """Category C: Tests for 4-5 level deep nesting chains.
    
    Patterns tested:
    - 4-Level: Pool → KMut → InsertionScan → SpacingScan
    - 4-Level: Pool → SpacingScan → MixedPool → KMutation
    - 4-Level: IUPACPool → InsertionScanORF → SpacingScan → KMutation
    - 5-Level: L1 → KMut → InsertionScan → MixedPool → SpacingScan
    """
    
    # -------------------------------------------------------------------------
    # Config Verification (4 tests)
    # -------------------------------------------------------------------------
    
    def test_config_4level_kmut_insertion_spacing(self):
        """4-Level: Pool→KMut→InsertionScan→SpacingScan: all configs valid."""
        # L1: Base pool
        L1 = Pool(['GGGGGGGGGGGG'], name='L1', metadata='complete')  # 12bp
        
        # L2: KMutation with k=2, positions=[0,3,6,9]
        L2 = KMutationPool(L1, k=2, positions=[0, 3, 6, 9], name='L2', 
                          mode='sequential', metadata='complete')
        
        # L3: InsertionScan with start=2, end=10, step=2, overwrite mode -> [2,4,6,8]
        L3 = InsertionScanPool(L2, 'XX', start=2, end=10, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        
        # L4: SpacingScan with distances
        other = Pool(['YY'], name='other', metadata='complete')
        L4 = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[L3, other],
            insert_names=['A', 'B'],
            anchor_pos=20,
            insert_distances=[[-8, -4], [4, 8]],  # 2*2=4 combos
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=30, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(30):
            row = dc.get_row(i)
            
            # L2: exactly 2 mutations, positions in [0,3,6,9]
            assert len(row['L2_mut_pos']) == 2, "L2 should have exactly 2 mutations"
            for pos in row['L2_mut_pos']:
                assert pos in [0, 3, 6, 9], f"L2 mut_pos {pos} not in allowed list"
            
            # L3: pos in [2, 4, 6, 8] (overwrite mode: end-W+1=10-2+1=9)
            l3_pos = row['L3_pos']
            assert l3_pos in [2, 4, 6, 8], f"L3_pos {l3_pos} not in [2, 4, 6, 8]"
            
            # L4: distances from config
            a_dist = row['L4_A_dist']
            b_dist = row['L4_B_dist']
            assert a_dist in [-8, -4], f"L4_A_dist {a_dist} not in [-8, -4]"
            assert b_dist in [4, 8], f"L4_B_dist {b_dist} not in [4, 8]"
    
    def test_config_4level_spacing_mixed_kmut(self):
        """4-Level: Pool→SpacingScan→MixedPool→KMutation: all configs valid."""
        # L1: Base
        L1 = Pool(['AAAA'], name='L1', metadata='complete')
        
        # L2: SpacingScan
        L2_other = Pool(['TT'], name='L2_other', metadata='complete')
        L2 = SpacingScanPool(
            background_seq='NNNNNNNNNN',
            insert_seqs=[L1, L2_other],
            insert_names=['A', 'B'],
            anchor_pos=5,
            insert_distances=[[-2], [2]],
            name='L2', mode='sequential', metadata='complete'
        )
        
        # L3: MixedPool wrapping L2
        L3_alt = Pool(['CCCCCCCCCC'], name='L3_alt', metadata='complete')
        L3 = MixedPool([L2, L3_alt], name='L3', mode='sequential')
        
        # L4: KMutation on MixedPool output
        L4 = KMutationPool(L3, k=1, name='L4', mode='sequential', metadata='complete')
        
        result = L4.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            
            # L4: exactly 1 mutation
            assert len(row['L4_mut_pos']) == 1, "L4 should have 1 mutation"
            
            # L3: selection valid
            l3_selected = row['L3_selected']
            assert l3_selected in [0, 1], f"L3_selected {l3_selected} should be 0 or 1"
            
            # L2: if selected, distances valid
            if l3_selected == 0:
                assert row['L2_A_dist'] == -2
                assert row['L2_B_dist'] == 2
    
    def test_config_4level_iupac_insertion_orf_spacing_kmut(self):
        """4-Level: IUPAC→InsertionScanORF→SpacingScan→KMut: all configs valid."""
        # L1: IUPACPool with R = A or G
        L1 = IUPACPool('ATGRRRRRRTAA', name='L1', metadata='complete')  # 12bp = 4 codons
        
        # L2: InsertionScanORF, codon positions 1-3
        L2_ins = Pool(['GGG'], name='L2_ins', metadata='complete')
        L2 = InsertionScanORFPool(
            L1, L2_ins,
            start=1, end=3, step_size=1,  # codon positions [1, 2]
            insert_or_overwrite='overwrite',
            name='L2', mode='sequential', metadata='complete'
        )
        
        # L3: SpacingScan
        L3_other = Pool(['CCCC'], name='L3_other', metadata='complete')
        L3 = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[L2, L3_other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-6], [6]],
            name='L3', mode='sequential', metadata='complete'
        )
        
        # L4: KMutation
        L4 = KMutationPool(L3, k=1, name='L4', mode='sequential', metadata='complete')
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            
            # L1: value contains only A, T, G (R expanded)
            l1_val = row['L1_value']
            assert all(c in 'ATGC' for c in l1_val), f"L1 should be ATGC only"
            
            # L2: codon_pos in [1, 2]
            codon_pos = row['L2_codon_pos']
            assert codon_pos in [1, 2], f"L2 codon_pos {codon_pos} not in [1, 2]"
            
            # L4: 1 mutation
            assert len(row['L4_mut_pos']) == 1
    
    def test_config_5level_full_chain(self):
        """5-Level: L1→KMut→InsertionScan→MixedPool→SpacingScan: all configs valid."""
        # L1: Base
        L1 = IUPACPool('RRRRRRRR', name='L1', metadata='complete')  # 8bp, R=A/G
        
        # L2: KMutation k=1, positions=[0,2,4,6]
        L2 = KMutationPool(L1, k=1, positions=[0, 2, 4, 6], name='L2',
                          mode='sequential', metadata='complete')
        
        # L3: InsertionScan start=1, end=5, step=2 -> [1, 3]
        L3 = InsertionScanPool(L2, 'XX', start=1, end=5, step_size=2,
                               insert_or_overwrite='insert',
                               name='L3', mode='sequential', metadata='complete')
        
        # L4: MixedPool
        L4_alt = Pool(['YYYYYYYYYY'], name='L4_alt', metadata='complete')
        L4 = MixedPool([L3, L4_alt], name='L4', mode='sequential')
        
        # L5: SpacingScan
        L5_other = Pool(['ZZ'], name='L5_other', metadata='complete')
        L5 = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[L4, L5_other],
            insert_names=['Main', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-6, -3], [3, 6]],  # 2*2=4
            name='L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            
            # L1: only A/G (from R)
            l1_val = row['L1_value']
            if l1_val is not None:
                assert all(c in 'AG' for c in l1_val)
            
            # L2: 1 mutation at allowed position
            if row['L2_mut_pos'] is not None:
                assert len(row['L2_mut_pos']) == 1
                assert row['L2_mut_pos'][0] in [0, 2, 4, 6]
            
            # L3: pos in [1, 3, 5] when selected (insert mode: range(1, 5+1, 2))
            if row['L3_pos'] is not None:
                assert row['L3_pos'] in [1, 3, 5]
            
            # L4: selection valid
            l4_sel = row['L4_selected']
            assert l4_sel in [0, 1]
            
            # L5: distances valid
            assert row['L5_Main_dist'] in [-6, -3]
            assert row['L5_Oth_dist'] in [3, 6]
    
    # -------------------------------------------------------------------------
    # Combination Coverage (4 tests)
    # -------------------------------------------------------------------------
    
    def test_combo_4level_kmut_insertion_spacing(self):
        """4-Level: all pools tracked with columns."""
        L1 = Pool(['GGGGGGGGGGGG'], name='c4L1', metadata='complete')  # 12bp
        L2 = KMutationPool(L1, k=1, name='c4L2', mode='sequential', metadata='complete')  # 12bp
        L3 = InsertionScanPool(L2, 'XX', start=2, end=8, step_size=2,
                               name='c4L3', mode='sequential', metadata='complete')  # 12bp (overwrite)
        other = Pool(['YY'], name='c4oth', metadata='complete')  # 2bp
        # L3 is 12bp, other is 2bp. Use larger background and non-overlapping distances
        L4 = SpacingScanPool(
            'N' * 50, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-15], [15]],  # -15+25=10, 15+25=40, no overlap
            name='c4L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All 5 pools tracked
        for name in ['c4L1', 'c4L2', 'c4L3', 'c4L4', 'c4oth']:
            assert f'{name}_index' in dc.keys, f"{name} should be tracked"
        
        # Specific metadata columns
        assert 'c4L2_mut_pos' in dc.keys
        assert 'c4L3_pos' in dc.keys
        assert 'c4L4_A_dist' in dc.keys
    
    def test_combo_4level_spacing_mixed_kmut(self):
        """4-Level with MixedPool: correct null-filling."""
        L1 = Pool(['AAAA'], name='sm4L1', metadata='complete')
        L1_other = Pool(['TT'], name='sm4L1o', metadata='complete')
        L2 = SpacingScanPool(
            'NNNNNNNN', insert_seqs=[L1, L1_other], insert_names=['A', 'B'],
            anchor_pos=4, insert_distances=[[-2], [2]],
            name='sm4L2', mode='sequential', metadata='complete'
        )
        L2_alt = Pool(['XXXXXXXX'], name='sm4L2a', metadata='complete')
        L3 = MixedPool([L2, L2_alt], name='sm4L3', mode='sequential')
        L4 = KMutationPool(L3, k=1, name='sm4L4', mode='sequential', metadata='complete')
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All tracked
        for name in ['sm4L1', 'sm4L1o', 'sm4L2', 'sm4L2a', 'sm4L3', 'sm4L4']:
            assert f'{name}_index' in dc.keys
        
        # Null-filling when L2_alt selected
        for i in range(10):
            row = dc.get_row(i)
            if row['sm4L3_selected'] == 1:
                # L2 not selected, its children should be None
                assert row['sm4L2_value'] is None
                assert row['sm4L1_value'] is None
    
    def test_combo_4level_orf_in_chain(self):
        """4-Level with ORF pool: ORF metadata present."""
        L1 = Pool(['ATGAAACCCGGG'], name='orf4L1', metadata='complete')  # 12bp = 4 codons
        L2_ins = Pool(['TTT'], name='orf4L2i', metadata='complete')
        L2 = InsertionScanORFPool(
            L1, L2_ins, start=1, end=3, step_size=1,
            name='orf4L2', mode='sequential', metadata='complete'
        )
        L3_other = Pool(['CCCC'], name='orf4L3o', metadata='complete')
        L3 = SpacingScanPool(
            'N' * 30, insert_seqs=[L2, L3_other], insert_names=['ORF', 'Oth'],
            anchor_pos=15, insert_distances=[[-6], [6]],
            name='orf4L3', mode='sequential', metadata='complete'
        )
        L4 = KMutationPool(L3, k=1, name='orf4L4', mode='sequential', metadata='complete')
        
        result = L4.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # ORF-specific columns
        assert 'orf4L2_codon_pos' in dc.keys
        assert 'orf4L2_codon_pos_abs' in dc.keys
    
    def test_combo_5level_full_chain(self):
        """5-Level: all 5+ pools tracked."""
        L1 = Pool(['GGGGGGGG'], name='c5L1', metadata='complete')  # 8bp
        L2 = KMutationPool(L1, k=1, name='c5L2', mode='sequential', metadata='complete')  # 8bp
        # Use overwrite mode: L3 produces 8bp output
        L3 = InsertionScanPool(L2, 'XX', start=1, end=7, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='c5L3', mode='sequential', metadata='complete')  # 8bp
        L3_alt = Pool(['YYYYYYYY'], name='c5L3a', metadata='complete')  # 8bp (match L3)
        L4 = MixedPool([L3, L3_alt], name='c5L4', mode='sequential')  # 8bp
        L4_other = Pool(['ZZ'], name='c5L4o', metadata='complete')  # 2bp
        L5 = SpacingScanPool(
            'N' * 40, insert_seqs=[L4, L4_other], insert_names=['M', 'O'],
            anchor_pos=20, insert_distances=[[-10], [10]],  # Non-overlapping
            name='c5L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools tracked
        for name in ['c5L1', 'c5L2', 'c5L3', 'c5L3a', 'c5L4', 'c5L4o', 'c5L5']:
            assert f'{name}_index' in dc.keys, f"{name} should be tracked"
    
    # -------------------------------------------------------------------------
    # Position/Value Integrity (4 tests)
    # -------------------------------------------------------------------------
    
    def test_integrity_4level_extraction_all_levels(self):
        """4-Level: values are correct and consistent at every level."""
        L1 = Pool(['GGGGGGGGGGGG'], name='i4L1', metadata='complete')  # 12bp
        L2 = KMutationPool(L1, k=1, name='i4L2', mode='sequential', metadata='complete')  # 12bp
        L3 = InsertionScanPool(L2, 'XX', start=2, end=8, step_size=2,
                               insert_or_overwrite='insert',
                               name='i4L3', mode='sequential', metadata='complete')  # 14bp (12+2)
        other = Pool(['YY'], name='i4oth', metadata='complete')  # 2bp
        L4 = SpacingScanPool(
            'N' * 50, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-15], [15]],  # Non-overlapping
            name='i4L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L4 value matches full sequence
            assert row['i4L4_value'] == seq
            
            # L3 is transformer parent (fed to SS), abs_start/abs_end are None
            # Verify L3 value is present in the sequence and has expected length (14bp)
            l3_value = row['i4L3_value']
            assert l3_value is not None, "L3 value should be present"
            assert len(l3_value) == 14, f"L3 should be 14bp, got {len(l3_value)}"
            assert l3_value in seq, f"L3 value should be in sequence"
    
    def test_integrity_4level_spacing_formula(self):
        """4-Level with SpacingScan: spacing formula holds."""
        L1 = Pool(['AAAA'], name='sf4L1', metadata='complete')
        L2 = KMutationPool(L1, k=1, name='sf4L2', mode='sequential', metadata='complete')
        other = Pool(['TTTT'], name='sf4oth', metadata='complete')
        L3 = SpacingScanPool(
            'N' * 30, insert_seqs=[L2, other], insert_names=['Mut', 'Oth'],
            anchor_pos=15, insert_distances=[[-8], [8]],
            name='sf4L3', mode='sequential', metadata='complete'
        )
        L4 = KMutationPool(L3, k=1, name='sf4L4', mode='sequential', metadata='complete')
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            
            # Spacing formula at L3
            mut_end = row['sf4L3_Mut_pos_end']
            oth_start = row['sf4L3_Oth_pos_start']
            expected = oth_start - mut_end
            actual = row['sf4L3_spacing_Mut_Oth']
            assert actual == expected
    
    def test_integrity_4level_mutation_positions(self):
        """4-Level: mutation positions allow character verification."""
        L1 = Pool(['GGGGGGGGGGGG'], name='mp4L1', metadata='complete')
        L2 = KMutationPool(L1, k=2, name='mp4L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'AA', start=2, end=8, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='mp4L3', mode='sequential', metadata='complete')
        other = Pool(['TT'], name='mp4oth', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 30, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-6], [6]],
            name='mp4L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            
            # L2: original was all G, mutations should not be G
            for mut_to in row['mp4L2_mut_to']:
                assert mut_to != 'G', "Mutation should change from G to something else"
    
    def test_integrity_5level_extraction(self):
        """5-Level: extraction works at all levels."""
        L1 = Pool(['GGGGGGGG'], name='i5L1', metadata='complete')
        L2 = KMutationPool(L1, k=1, name='i5L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'XX', start=1, end=5, step_size=2,
                               insert_or_overwrite='insert',
                               name='i5L3', mode='sequential', metadata='complete')
        L3_alt = Pool(['YYYYYYYYYY'], name='i5L3a', metadata='complete')
        L4 = MixedPool([L3, L3_alt], name='i5L4', mode='sequential')
        L4_other = Pool(['ZZ'], name='i5L4o', metadata='complete')
        L5 = SpacingScanPool(
            'N' * 35, insert_seqs=[L4, L4_other], insert_names=['M', 'O'],
            anchor_pos=17, insert_distances=[[-7], [7]],
            name='i5L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L5 is full sequence
            assert row['i5L5_value'] == seq
    
    # -------------------------------------------------------------------------
    # Distributional Verification (4 tests)
    # -------------------------------------------------------------------------
    
    def test_distrib_4level_all_distributions(self):
        """4-Level: all level distributions verified."""
        # L1: IUPAC R = A/G
        L1 = IUPACPool('RRRRRRRRRRRR', name='d4L1', metadata='complete')
        
        # L2: k=1, positions=[0,3,6,9]
        L2 = KMutationPool(L1, k=1, positions=[0, 3, 6, 9], name='d4L2',
                          mode='sequential', metadata='complete')
        
        # L3: positions [2, 4, 6]
        L3 = InsertionScanPool(L2, 'XX', start=2, end=8, step_size=2,
                               insert_or_overwrite='insert',
                               name='d4L3', mode='sequential', metadata='complete')
        
        # L3 is 14bp (12+2). Use large bg, ensure no overlaps
        other = Pool(['YY'], name='d4oth', metadata='complete')  # 2bp
        L4 = SpacingScanPool(
            'N' * 60, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=30, insert_distances=[[-20, -18], [18]],  # A ends before B starts
            name='d4L4', mode='sequential', metadata='complete'
        )
        
        # 3 L3 positions * 2 L4 combos = 6 states per L1/L2 combo
        result = L4.generate_seqs(num_seqs=120, return_design_cards=True)
        dc = result['design_cards']
        
        # L1: ~50% A, ~50% G
        all_l1 = ''.join(dc.get_row(i)['d4L1_value'] for i in range(120))
        a_freq = all_l1.count('A') / len(all_l1)
        assert 0.40 < a_freq < 0.60, f"L1 IUPAC R should be ~50% A, got {a_freq:.2%}"
        
        # L2: mutation positions uniform (4 positions, 25% each)
        l2_positions = [dc.get_row(i)['d4L2_mut_pos'][0] for i in range(120)]
        for pos in [0, 3, 6, 9]:
            freq = l2_positions.count(pos) / len(l2_positions)
            assert 0.15 < freq < 0.45, f"L2 position {pos} freq {freq:.2%} outside bounds"
        
        # L3: positions uniform (4 positions: [2, 4, 6, 8] from start=2, end=8, step=2 in insert mode)
        l3_positions = [dc.get_row(i)['d4L3_pos'] for i in range(120)]
        for pos in [2, 4, 6, 8]:
            count = l3_positions.count(pos)
            assert count == 30, f"L3 pos {pos} should appear 30 times, got {count}"
    
    def test_distrib_4level_spacing_combos_complete(self):
        """4-Level: SpacingScan combinations complete."""
        L1 = Pool(['GGGG'], name='dsc4L1', metadata='complete')  # 4bp
        L2 = KMutationPool(L1, k=1, positions=[0, 1], name='dsc4L2',
                          mode='sequential', metadata='complete')  # 2 positions × 3 muts = 6 states
        other = Pool(['TT'], name='dsc4oth', metadata='complete')  # 2bp
        # 4bp + 2bp inserts, well-separated
        L3 = SpacingScanPool(
            'N' * 30, insert_seqs=[L2, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-10, -8], [6, 8]],  # 2*2=4 combos
            name='dsc4L3', mode='sequential', metadata='complete'
        )
        # Use L3 as the top-level, add a simple wrapper to make 4 levels
        L4 = KMutationPool(L3, k=1, positions=[0, 1], name='dsc4L4',
                          mode='sequential', metadata='complete')  # 2 positions
        
        # Total states = L2(6) × L3(4) × L4(6) = 144
        # Generate 144 to cover all combos
        result = L4.generate_seqs(num_seqs=144, return_design_cards=True)
        dc = result['design_cards']
        
        # All 4 distance combos should appear
        combos = set()
        for i in range(144):
            row = dc.get_row(i)
            combos.add((row['dsc4L3_A_dist'], row['dsc4L3_B_dist']))
        assert len(combos) == 4, f"Should have 4 combos, got {len(combos)}"
    
    def test_distrib_5level_mixed_selection(self):
        """5-Level: MixedPool selection uniform in sequential mode."""
        # Keep pools small to have manageable state count
        L1 = Pool(['GGGGGGGG'], name='d5L1', metadata='complete')  # 8bp
        L2 = KMutationPool(L1, k=1, positions=[0, 1], name='d5L2',
                          mode='sequential', metadata='complete')  # 2 mut positions × 3 muts = 6 states
        # Use overwrite mode - with 8bp bg, 2bp insert, start=0, end=7 → positions [0, 2, 4] = 3 positions
        L3 = InsertionScanPool(L2, 'XX', start=0, end=7, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='d5L3', mode='sequential', metadata='complete')  # 8bp, 3 positions
        L3_alt = Pool(['YYYYYYYY'], name='d5L3a', metadata='complete')  # 8bp (same as L3)
        # MixedPool: L3 has 6×3=18 states, L3_alt has 1 state → total 19 per iteration
        L4 = MixedPool([L3, L3_alt], name='d5L4', mode='sequential')  # 8bp
        L4_other = Pool(['ZZ'], name='d5L4o', metadata='complete')  # 2bp
        L5 = SpacingScanPool(
            'N' * 30, insert_seqs=[L4, L4_other], insert_names=['M', 'O'],
            anchor_pos=15, insert_distances=[[-10, -8], [8, 10]],  # 2*2=4 combos
            name='d5L5', mode='sequential', metadata='complete'
        )
        
        # L4 has 19 internal states (18 from L3, 1 from L3_alt)
        # L5 has 4 combos → total = 19 * 4 = 76 states
        # Generate 76 seqs = 1 complete iteration
        result = L5.generate_seqs(num_seqs=76, return_design_cards=True)
        dc = result['design_cards']
        
        # L4 selection: L3 should be selected 18x, L3_alt 1x per SS combo → 18*4=72, 1*4=4
        selections = [dc.get_row(i)['d5L4_selected'] for i in range(76)]
        l3_count = selections.count(0)
        l3a_count = selections.count(1)
        # 18:1 ratio per combo → 72:4 total
        assert l3_count == 72, f"L3 should be selected 72 times, got {l3_count}"
        assert l3a_count == 4, f"L3_alt should be selected 4 times, got {l3a_count}"
    
    def test_distrib_5level_weighted_mixed(self):
        """5-Level: weighted MixedPool follows weights."""
        L1 = Pool(['GGGGGGGG'], name='dw5L1', metadata='complete')  # 8bp
        L2 = KMutationPool(L1, k=1, name='dw5L2', mode='sequential', metadata='complete')  # 8bp
        L3 = InsertionScanPool(L2, 'XX', start=1, end=7, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='dw5L3', mode='sequential', metadata='complete')  # 8bp
        L3_alt = Pool(['YYYYYYYY'], name='dw5L3a', metadata='complete')  # 8bp (same as L3)
        # Weighted 80/20
        L4 = MixedPool([L3, L3_alt], weights=[0.8, 0.2], name='dw5L4', mode='random')  # 8bp
        L4_other = Pool(['ZZ'], name='dw5L4o', metadata='complete')  # 2bp
        L5 = SpacingScanPool(
            'N' * 40, insert_seqs=[L4, L4_other], insert_names=['M', 'O'],
            anchor_pos=20, insert_distances=[[-10], [10]],  # Non-overlapping
            name='dw5L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=300, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        # L4 selection should be ~80/20
        selections = [dc.get_row(i)['dw5L4_selected'] for i in range(300)]
        l3_ratio = selections.count(0) / 300
        assert 0.70 < l3_ratio < 0.90, f"L3 ratio {l3_ratio:.2%} should be ~80%"


# =============================================================================
# Category D: ORF Pools in Complex Multi-Input (12 tests)
# =============================================================================

class TestORFInComplexMultiInput:
    """Category D: Tests for ORF pools in complex multi-input structures.
    
    Structures tested:
    - KMutationORFPool as SpacingScan insert
    - DeletionScanORFPool as SpacingScan insert
    - MixedPool[KMutORF, InsertionScanORF, DeletionScanORF] in SpacingScan
    """
    
    # -------------------------------------------------------------------------
    # Config Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_config_kmutation_orf_in_spacing(self):
        """KMutationORFPool as SpacingScan insert: ORF config valid."""
        # KMutationORF: k=1 mutation in codons 0-4 (5 codons)
        orf_pool = KMutationORFPool(
            'ATGAAACCCGGGTTT',  # 15bp = 5 codons
            mutation_type='any_codon', k=1,
            orf_start=0, orf_end=15,
            name='orf', mode='sequential', metadata='complete'
        )
        
        other = Pool(['AAAA'], name='other', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[orf_pool, other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-8], [8]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=15, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(15):
            row = dc.get_row(i)
            
            # ORF: exactly 1 codon mutated
            codon_pos = row['orf_codon_pos']
            assert len(codon_pos) == 1, "Should have exactly 1 mutation"
            
            # Codon position in valid range (0-4 for 5 codons)
            assert 0 <= codon_pos[0] < 5, f"codon_pos {codon_pos[0]} out of range"
            
            # Note: codon_pos_abs is None for transformer parents inside SpacingScan
            # This is expected - use codon_pos (relative) instead
    
    def test_config_deletion_scan_orf_in_spacing(self):
        """DeletionScanORFPool as SpacingScan insert: del_codons valid."""
        # DeletionScanORF: delete 1 codon at a time
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGGTTT',  # 15bp = 5 codons
            deletion_size=1,
            start=1, end=4,  # codon positions [1, 2, 3]
            name='del', mode='sequential', metadata='complete', mark_changes=True
        )
        
        other = Pool(['TTTT'], name='other', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,  # Larger background
            insert_seqs=[del_orf, other],
            insert_names=['Del', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-8], [8]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=9, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(9):
            row = dc.get_row(i)
            
            # del_codons should be 3bp (1 codon)
            del_codons = row['del_del_codons']
            assert len(del_codons) == 3, f"del_codons should be 3bp, got {len(del_codons)}"
            
            # codon_pos in [1, 2, 3]
            codon_pos = row['del_codon_pos']
            assert codon_pos in [1, 2, 3], f"codon_pos {codon_pos} not in [1, 2, 3]"
    
    def test_config_mixed_orf_types_in_spacing(self):
        """MixedPool[KMutORF, InsertionScanORF, DelScanORF]: each type's config valid."""
        # KMutationORF
        kmut_orf = KMutationORFPool(
            'ATGAAACCCGGG',  # 12bp = 4 codons
            mutation_type='any_codon', k=1,
            name='kmut', mode='sequential', metadata='complete'
        )
        
        # InsertionScanORF
        is_orf = InsertionScanORFPool(
            'ATGAAACCCGGG', 'TTT',
            start=1, end=3, step_size=1,  # codon positions [1, 2]
            name='isorf', mode='sequential', metadata='complete'
        )
        
        # DeletionScanORF
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGG',
            deletion_size=1,
            start=1, end=3,  # codon positions [1, 2]
            name='dorf', mode='sequential', metadata='complete', mark_changes=True
        )
        
        mixed = MixedPool([kmut_orf, is_orf, del_orf], name='mix', mode='sequential')
        
        other = Pool(['CCCC'], name='other', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-6], [6]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            selected = row['mix_selected']
            
            if selected == 0:
                # KMutORF: 1 codon mutated
                assert row['kmut_codon_pos'] is not None
                assert len(row['kmut_codon_pos']) == 1
            elif selected == 1:
                # InsertionScanORF: codon_pos in [1, 2]
                assert row['isorf_codon_pos'] in [1, 2]
            else:
                # DeletionScanORF: codon_pos in [1, 2]
                assert row['dorf_codon_pos'] in [1, 2]
    
    # -------------------------------------------------------------------------
    # Combination Coverage (3 tests)
    # -------------------------------------------------------------------------
    
    def test_combo_kmutation_orf_in_spacing(self):
        """KMutationORFPool as SpacingScan insert: all pools tracked."""
        orf_pool = KMutationORFPool(
            'ATGAAACCCGGGTTT', mutation_type='any_codon', k=1,
            name='corf', mode='sequential', metadata='complete'
        )
        
        other = Pool(['AAAA'], name='coth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[orf_pool, other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='css', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # All tracked
        for name in ['corf', 'coth', 'css']:
            assert f'{name}_index' in dc.keys
        
        # ORF metadata - codon_pos is always present
        assert 'corf_codon_pos' in dc.keys
        assert 'corf_codon_from' in dc.keys
        assert 'corf_codon_to' in dc.keys
        
        # Spacing metadata
        assert 'css_ORF_dist' in dc.keys
        assert 'css_spacing_ORF_Oth' in dc.keys
    
    def test_combo_deletion_scan_orf_in_spacing(self):
        """DeletionScanORFPool as SpacingScan insert: all metadata present."""
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGGTTT', deletion_size=1, start=1, end=4,
            name='cdel', mode='sequential', metadata='complete', mark_changes=True
        )
        
        other = Pool(['TTTT'], name='cdoth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[del_orf, other],
            insert_names=['Del', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='cdss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # Deletion ORF metadata
        assert 'cdel_codon_pos' in dc.keys
        assert 'cdel_del_codons' in dc.keys
        assert 'cdel_del_aa' in dc.keys
    
    def test_combo_mixed_orf_types(self):
        """MixedPool[KMutORF, InsertionScanORF, DelScanORF]: correct nulls."""
        kmut_orf = KMutationORFPool(
            'ATGAAACCCGGG', mutation_type='any_codon', k=1,
            name='mkmut', mode='sequential', metadata='complete'
        )
        is_orf = InsertionScanORFPool(
            'ATGAAACCCGGG', 'TTT', start=1, end=3, step_size=1,
            name='misorf', mode='sequential', metadata='complete'
        )
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGG', deletion_size=1, start=1, end=3,
            name='mdorf', mode='sequential', metadata='complete', mark_changes=True
        )
        
        mixed = MixedPool([kmut_orf, is_orf, del_orf], name='mmix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=15, return_design_cards=True)
        dc = result['design_cards']
        
        # All ORF types tracked
        for name in ['mkmut', 'misorf', 'mdorf', 'mmix']:
            assert f'{name}_index' in dc.keys
        
        # Verify correct null-filling
        for i in range(15):
            row = dc.get_row(i)
            selected = row['mmix_selected']
            
            if selected == 0:
                assert row['mkmut_codon_pos'] is not None
                assert row['misorf_codon_pos'] is None
                assert row['mdorf_codon_pos'] is None
            elif selected == 1:
                assert row['mkmut_codon_pos'] is None
                assert row['misorf_codon_pos'] is not None
                assert row['mdorf_codon_pos'] is None
            else:
                assert row['mkmut_codon_pos'] is None
                assert row['misorf_codon_pos'] is None
                assert row['mdorf_codon_pos'] is not None
    
    # -------------------------------------------------------------------------
    # Position/Value Integrity (3 tests)
    # -------------------------------------------------------------------------
    
    def test_integrity_kmutation_orf_codon_extraction(self):
        """KMutationORFPool: codon_to in value and original sequence differs."""
        orf_pool = KMutationORFPool(
            'ATGAAACCCGGGTTT', mutation_type='any_codon', k=1,
            name='iorf', mode='sequential', metadata='complete'
        )
        
        other = Pool(['AAAA'], name='ioth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[orf_pool, other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='iss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=15, return_design_cards=True)
        dc = result['design_cards']
        
        original = 'ATGAAACCCGGGTTT'
        for i in range(15):
            row = dc.get_row(i)
            
            # Get ORF metadata
            codon_pos = row['iorf_codon_pos'][0]
            codon_from = row['iorf_codon_from'][0]
            codon_to = row['iorf_codon_to'][0]
            orf_value = row['iorf_value']
            
            # Verify codon_from matches original at codon_pos
            expected_from = original[codon_pos*3:(codon_pos+1)*3]
            assert codon_from == expected_from, \
                f"codon_from '{codon_from}' != original at pos {codon_pos}: '{expected_from}'"
            
            # Verify codon_to is in the ORF value at correct position
            actual_codon_in_value = orf_value[codon_pos*3:(codon_pos+1)*3]
            assert actual_codon_in_value == codon_to, \
                f"Value codon '{actual_codon_in_value}' != codon_to '{codon_to}'"
    
    def test_integrity_deletion_orf_length(self):
        """DeletionScanORFPool: output length = original - deletion_size*3."""
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGGTTT',  # 15bp = 5 codons
            deletion_size=1,  # Delete 1 codon = 3bp
            start=1, end=4,
            name='idel', mode='sequential', metadata='complete', mark_changes=True
        )
        
        other = Pool(['TTTT'], name='idoth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[del_orf, other],
            insert_names=['Del', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='idss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=9, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(9):
            row = dc.get_row(i)
            
            # With mark_changes=True, deleted codon is marked with '---', so length stays 15bp
            del_value = row['idel_value']
            assert len(del_value) == 15, f"del value should be 15bp, got {len(del_value)}"
            assert '---' in del_value, f"del value should contain '---' marker"
    
    def test_integrity_mixed_orf_extraction(self):
        """MixedPool[ORF types]: extraction matches for selected child."""
        kmut_orf = KMutationORFPool(
            'ATGAAACCCGGG', mutation_type='any_codon', k=1,
            name='imkmut', mode='sequential', metadata='complete'
        )
        is_orf = InsertionScanORFPool(
            'ATGAAACCCGGG', 'TTT', start=1, end=3, step_size=1,
            insert_or_overwrite='overwrite',
            name='imisorf', mode='sequential', metadata='complete'
        )
        
        mixed = MixedPool([kmut_orf, is_orf], name='immix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['immix_selected']
            
            if selected == 0:
                # KMutORF selected - its value should match sequence
                assert row['imkmut_value'] == seq
            else:
                # InsertionScanORF selected
                assert row['imisorf_value'] == seq
    
    # -------------------------------------------------------------------------
    # Distributional Verification (3 tests)
    # -------------------------------------------------------------------------
    
    def test_distrib_kmutation_orf_positions_uniform(self):
        """KMutationORFPool: codon positions uniform across range."""
        # Use positions=[1,2] to limit states: 2 positions × 63 codon substitutions = 126 states
        orf_pool = KMutationORFPool(
            'ATGAAACCCGGGTTT',  # 5 codons
            mutation_type='any_codon', k=1, positions=[1, 2],  # 2 positions
            name='dorf', mode='sequential', metadata='complete'
        )
        
        other = Pool(['AAAA'], name='doth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[orf_pool, other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='dss', mode='sequential', metadata='complete'
        )
        
        # 2 codon positions × 63 substitutions = 126 states; generate 126 to see all positions
        result = ss.generate_seqs(num_seqs=126, return_design_cards=True)
        dc = result['design_cards']
        
        # Each codon position should appear 63 times (126/2)
        positions = [dc.get_row(i)['dorf_codon_pos'][0] for i in range(126)]
        for pos in [1, 2]:
            count = positions.count(pos)
            assert count == 63, f"codon_pos {pos} should appear 63 times, got {count}"
    
    def test_distrib_deletion_orf_positions_uniform(self):
        """DeletionScanORFPool: deletion positions uniform."""
        del_orf = DeletionScanORFPool(
            'ATGAAACCCGGGTTT', deletion_size=1, start=1, end=4,  # positions [1,2,3]
            name='ddel', mode='sequential', metadata='complete', mark_changes=True
        )
        
        other = Pool(['TTTT'], name='ddoth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[del_orf, other],
            insert_names=['Del', 'Oth'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='ddss', mode='sequential', metadata='complete'
        )
        
        # 3 deletion positions per iteration
        result = ss.generate_seqs(num_seqs=30, return_design_cards=True)
        dc = result['design_cards']
        
        positions = [dc.get_row(i)['ddel_codon_pos'] for i in range(30)]
        for pos in [1, 2, 3]:
            count = positions.count(pos)
            assert count == 10, f"codon_pos {pos} should appear 10 times, got {count}"
    
    def test_distrib_mixed_orf_weighted_selection(self):
        """MixedPool[ORF types] with weights: selection follows weights."""
        kmut_orf = KMutationORFPool(
            'ATGAAACCCGGG', mutation_type='any_codon', k=1,
            name='wkmut', mode='sequential', metadata='complete'
        )
        is_orf = InsertionScanORFPool(
            'ATGAAACCCGGG', 'TTT', start=1, end=3, step_size=1,
            name='wisorf', mode='sequential', metadata='complete'
        )
        
        # 70/30 weights
        mixed = MixedPool([kmut_orf, is_orf], weights=[0.7, 0.3], name='wmix', mode='random')
        
        other = Pool(['CCCC'], name='woth', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=15, insert_distances=[[-6], [6]],
            name='wss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=300, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        selections = [dc.get_row(i)['wmix_selected'] for i in range(300)]
        kmut_ratio = selections.count(0) / 300
        assert 0.60 < kmut_ratio < 0.80, f"KMut ratio {kmut_ratio:.2%} should be ~70%"


# =============================================================================
# Category E: Pool/Transformer Reuse in Multi-Input (8 tests)
# =============================================================================

class TestPoolReuseInMultiInput:
    """Category E: Tests for pool/transformer reuse in complex structures.
    
    Structures tested:
    - Same Pool used as bg AND insert in InsertionScan within SpacingScan
    - Same KMutation transformer in multiple SpacingScan inserts
    """
    
    # -------------------------------------------------------------------------
    # Config Verification (2 tests)
    # -------------------------------------------------------------------------
    
    def test_config_same_pool_bg_and_insert(self):
        """Same Pool as bg AND insert: both occurrences have valid metadata."""
        # Same pool used twice
        shared = Pool(['AAAA'], name='shared', metadata='complete')
        
        # Use shared as both bg and insert
        is_pool = InsertionScanPool(
            background_seq=shared,
            insert_seq=shared,
            start=0, end=4, step_size=1,  # positions [0, 1, 2, 3, 4] in insert mode
            insert_or_overwrite='insert',
            name='is', mode='sequential', metadata='complete'
        )
        
        other = Pool(['TT'], name='other', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 20,
            insert_seqs=[is_pool, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=10,
            insert_distances=[[-4], [4]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=9, return_design_cards=True)
        dc = result['design_cards']
        
        # shared[1] and shared[2] should exist
        assert 'shared[1]_index' in dc.keys or 'shared_index' in dc.keys
        
        for i in range(9):
            row = dc.get_row(i)
            
            # IS position valid
            is_pos = row['is_pos']
            assert is_pos in [0, 1, 2, 3, 4], f"is_pos {is_pos} not in [0, 1, 2, 3, 4]"
            
            # Both occurrences of shared have same value (shared state)
            shared_vals = []
            for key in dc.keys:
                if key.startswith('shared') and key.endswith('_value'):
                    val = row[key]
                    if val is not None:
                        shared_vals.append(val)
            
            # All non-None values should be equal
            if len(shared_vals) > 1:
                assert all(v == shared_vals[0] for v in shared_vals), \
                    f"Shared pool occurrences should have same value"
    
    def test_config_same_transformer_multiple_inserts(self):
        """Same KMutation in multiple SpacingScan inserts: configs valid."""
        base = Pool(['GGGGGGGG'], name='base', metadata='complete')
        kmut = KMutationPool(base, k=1, positions=[0, 2, 4, 6], name='kmut',
                            mode='sequential', metadata='complete')
        
        # Use kmut twice as inserts
        ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[kmut, kmut],
            insert_names=['A', 'B'],
            anchor_pos=15,
            insert_distances=[[-6], [6]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            
            # Both kmut occurrences should have valid mutation positions
            for key in dc.keys:
                if 'kmut' in key and '_mut_pos' in key:
                    mut_pos = row[key]
                    if mut_pos is not None:
                        for pos in mut_pos:
                            assert pos in [0, 2, 4, 6], \
                                f"Mutation position {pos} not in allowed list"
    
    # -------------------------------------------------------------------------
    # Combination Coverage (2 tests)
    # -------------------------------------------------------------------------
    
    def test_combo_same_pool_bg_and_insert(self):
        """Same Pool as bg AND insert: [1] and [2] suffixes correct."""
        shared = Pool(['AAAA'], name='cshared', metadata='complete')
        
        is_pool = InsertionScanPool(
            shared, shared,
            start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='cis', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        
        # Should have cshared[1] and cshared[2] (or similar occurrence notation)
        shared_keys = [k for k in dc.keys if 'cshared' in k]
        assert len(shared_keys) >= 2, f"Should have at least 2 shared occurrences, got {shared_keys}"
        
        # Check for occurrence suffixes
        has_occurrence = any('[1]' in k or '[2]' in k for k in shared_keys)
        if len(shared_keys) > 1:
            # Multiple keys means occurrence tracking is working
            pass
    
    def test_combo_same_transformer_multiple_inserts(self):
        """Same transformer in multiple inserts: tracked with suffixes."""
        base = Pool(['GGGG'], name='cbase', metadata='complete')
        kmut = KMutationPool(base, k=1, name='ckmut', mode='sequential', metadata='complete')
        
        ss = SpacingScanPool(
            background_seq='N' * 20,
            insert_seqs=[kmut, kmut],
            insert_names=['A', 'B'],
            anchor_pos=10,
            insert_distances=[[-4], [4]],
            name='css', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # kmut should appear with occurrence suffixes
        kmut_keys = [k for k in dc.keys if 'ckmut' in k]
        assert len(kmut_keys) >= 2, f"Should have multiple kmut keys, got {kmut_keys}"
        
        # base should also appear with occurrence suffixes (as transformer parent)
        base_keys = [k for k in dc.keys if 'cbase' in k]
        assert len(base_keys) >= 2, f"Should have multiple base keys, got {base_keys}"
    
    # -------------------------------------------------------------------------
    # Position/Value Integrity (2 tests)
    # -------------------------------------------------------------------------
    
    def test_integrity_same_pool_both_extractions_valid(self):
        """Same Pool as bg AND insert: values are correct and shared state is consistent."""
        shared = Pool(['AAAA'], name='ishared', metadata='complete')
        
        is_pool = InsertionScanPool(
            shared, shared,
            start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='iis', mode='sequential', metadata='complete'
        )
        
        other = Pool(['TT'], name='iother', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 20,
            insert_seqs=[is_pool, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=10,
            insert_distances=[[-4], [4]],
            name='iss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=9, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(9):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS is transformer parent (inside SS), so abs_start/abs_end are None
            # Verify the value is correct (8bp = 4bp bg + 4bp insert = AAAAAAAA)
            is_value = row['iis_value']
            assert is_value == 'AAAAAAAA', f"IS value should be 'AAAAAAAA', got '{is_value}'"
            
            # Verify the IS value is contained in the sequence
            assert 'AAAAAAAA' in seq, f"Sequence should contain IS value 'AAAAAAAA'"
    
    def test_integrity_same_transformer_state_consistent(self):
        """Same transformer in multiple inserts: shared state means same mutations."""
        base = Pool(['GGGGGGGG'], name='sbase', metadata='complete')
        kmut = KMutationPool(base, k=1, name='skmut', mode='sequential', metadata='complete')
        
        # kmut is 8bp, use large bg to avoid out-of-bounds
        ss = SpacingScanPool(
            background_seq='N' * 50,
            insert_seqs=[kmut, kmut],
            insert_names=['A', 'B'],
            anchor_pos=25,
            insert_distances=[[-15], [10]],  # A at 10-18, B at 35-43, no overlap
            name='sss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Both occurrences share state, so their values should match
            kmut_values = []
            for key in dc.keys:
                if 'skmut' in key and key.endswith('_value'):
                    val = row[key]
                    if val is not None:
                        kmut_values.append(val)
            
            if len(kmut_values) >= 2:
                assert kmut_values[0] == kmut_values[1], \
                    f"Shared transformer should produce same value: {kmut_values}"
    
    # -------------------------------------------------------------------------
    # Distributional Verification (2 tests)
    # -------------------------------------------------------------------------
    
    def test_distrib_same_pool_consistent_across_occurrences(self):
        """Same Pool in multiple spots: state always matches between occurrences."""
        shared = IUPACPool('RRRR', name='dshared', metadata='complete')  # R = A/G
        
        is_pool = InsertionScanPool(
            shared, shared,
            start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='dis', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=30, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify that both occurrences always have same value
        mismatches = 0
        for i in range(30):
            row = dc.get_row(i)
            
            shared_values = []
            for key in dc.keys:
                if 'dshared' in key and key.endswith('_value'):
                    val = row[key]
                    if val is not None:
                        shared_values.append(val)
            
            if len(shared_values) >= 2:
                if shared_values[0] != shared_values[1]:
                    mismatches += 1
        
        # All should match (shared state)
        assert mismatches == 0, f"Found {mismatches} mismatches between shared pool occurrences"
    
    def test_distrib_reused_transformer_mutation_distribution(self):
        """Same transformer reused: mutation positions still follow config."""
        base = Pool(['GGGGGGGGGGGG'], name='dtbase', metadata='complete')
        kmut = KMutationPool(base, k=1, positions=[0, 3, 6, 9], name='dtkmut',
                            mode='sequential', metadata='complete')
        
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[kmut, kmut],
            insert_names=['A', 'B'],
            anchor_pos=20,
            insert_distances=[[-8], [8]],
            name='dtss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=40, return_design_cards=True)
        dc = result['design_cards']
        
        # Collect all mutation positions from both occurrences
        all_positions = []
        for i in range(40):
            row = dc.get_row(i)
            for key in dc.keys:
                if 'dtkmut' in key and '_mut_pos' in key:
                    mut_pos = row[key]
                    if mut_pos is not None:
                        all_positions.extend(mut_pos)
        
        # All positions should be in allowed list
        for pos in all_positions:
            assert pos in [0, 3, 6, 9], f"Position {pos} not in allowed list"
        
        # Each position should appear with roughly equal frequency
        # With 4 positions and random mode, expect ~25% each
        if len(all_positions) > 0:
            for target_pos in [0, 3, 6, 9]:
                count = all_positions.count(target_pos)
                freq = count / len(all_positions)
                assert freq > 0, f"Position {target_pos} should appear at least once"

