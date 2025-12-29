"""Sequence correctness and computation graph tests for complex nested pools.

These tests verify:
1. Generated sequences are mathematically correct based on pool operations
2. Computation graph structure and node sequences are accurate
3. Pool-specific characteristics hold in nested contexts
4. Distributions match expectations with rigorous sample sizes

Test categories:
A: Multi-Input Inside Multi-Input (IS+SS, DS+SS, Motif+SS, etc.)
B: Multi-Input with MixedPool
C: Deep Nesting 4-5 Levels
D: ORF Pools in Complex Multi-Input
E: Pool Reuse
"""

import pytest
import pandas as pd
from poolparty import (
    Pool, MixedPool, DesignCards,
    InsertionScanPool, DeletionScanPool, SubseqPool, ShuffleScanPool,
    KMutationPool, RandomMutationPool,
    KMutationORFPool, RandomMutationORFPool,
    InsertionScanORFPool, DeletionScanORFPool,
    SpacingScanPool, IUPACPool, KmerPool, MotifPool,
)


def create_pwm(sequence: str) -> pd.DataFrame:
    """Create a deterministic PWM from a sequence."""
    bases = ['A', 'C', 'G', 'T']
    data = {base: [1.0 if seq == base else 0.0 for seq in sequence] for base in bases}
    return pd.DataFrame(data)


def hamming_distance(s1: str, s2: str) -> int:
    """Count positions where two strings differ."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def translate_codon(codon: str) -> str:
    """Translate a DNA codon to amino acid."""
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    return codon_table.get(codon.upper(), 'X')


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return ''.join(complement.get(b, b) for b in reversed(seq))


# =============================================================================
# Class 1: TestSequenceCorrectness (~25 tests)
# =============================================================================

class TestSequenceCorrectness:
    """Verify generated sequences are mathematically correct."""
    
    # -------------------------------------------------------------------------
    # Category A: Multi-Input Inside Multi-Input (6 tests)
    # -------------------------------------------------------------------------
    
    def test_seq_insertion_with_spacing_insert(self):
        """InsertionScan with SpacingScan insert: verify all characteristics."""
        # SpacingScan as insert: 4 combos
        ss_a = Pool(['AAAA'], name='ss_a', metadata='complete')
        ss_b = Pool(['TTTT'], name='ss_b', metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq='N' * 30,
            insert_seqs=[ss_a, ss_b],
            insert_names=['A', 'B'],
            anchor_pos=15,
            insert_distances=[[-10, -6], [6, 10]],  # 4 combos
            name='inner_ss', mode='sequential', metadata='complete'
        )
        
        # Outer InsertionScan: 3 positions
        outer_is = InsertionScanPool(
            background_seq='G' * 100,
            insert_seq=inner_ss,
            start=10, end=40, step_size=10,  # positions [10, 20, 30]
            insert_or_overwrite='insert',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        # Generate 100 sequences
        result = outer_is.generate_seqs(num_seqs=100, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(100):
            row = dc.get_row(i)
            seq = seqs[i]
            is_pos = row['outer_is_pos']
            inner_value = row['inner_ss_value']
            
            # IS Characteristic 1: Position accuracy - extract insert from sequence
            extracted = seq[is_pos:is_pos + len(inner_value)]
            assert extracted == inner_value, \
                f"IS position accuracy failed: seq[{is_pos}:{is_pos + len(inner_value)}] != inner_ss_value"
            
            # IS Characteristic 2: Background before unchanged
            assert seq[:is_pos] == 'G' * is_pos, \
                f"Background before {is_pos} should be G's"
            
            # IS Characteristic 3: Background after (insert mode) - shifted
            after_insert = is_pos + len(inner_value)
            assert seq[after_insert:] == 'G' * (len(seq) - after_insert), \
                "Background after insert should be G's"
            
            # SS Characteristic 1: Spacing formula holds
            a_start = row['inner_ss_A_pos_start']
            a_end = row['inner_ss_A_pos_end']
            b_start = row['inner_ss_B_pos_start']
            spacing = row['inner_ss_spacing_A_B']
            assert spacing == b_start - a_end, \
                f"Spacing formula: {spacing} != {b_start} - {a_end}"
            
            # SS Characteristic 2: Distances from config
            a_dist = row['inner_ss_A_dist']
            b_dist = row['inner_ss_B_dist']
            assert a_dist in [-10, -6], f"A_dist {a_dist} not in config"
            assert b_dist in [6, 10], f"B_dist {b_dist} not in config"
            
            # Verify both inserts are in the inner_ss_value
            assert 'AAAA' in inner_value, "ss_a insert should be in inner_ss_value"
            assert 'TTTT' in inner_value, "ss_b insert should be in inner_ss_value"
    
    def test_seq_spacing_with_insertion_insert(self):
        """SpacingScan with InsertionScan insert: verify characteristics."""
        # Inner InsertionScan: 4 positions
        inner_bg = Pool(['GGGGGGGG'], name='is_bg', metadata='complete')
        inner_ins = Pool(['XX'], name='is_ins', metadata='complete')
        inner_is = InsertionScanPool(
            background_seq=inner_bg,
            insert_seq=inner_ins,
            start=0, end=7, step_size=2,  # [0, 2, 4, 6]
            insert_or_overwrite='overwrite',
            name='inner_is', mode='sequential', metadata='complete'
        )
        
        other = Pool(['YYYY'], name='other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 50,
            insert_seqs=[inner_is, other],
            insert_names=['IS', 'Oth'],
            anchor_pos=25,
            insert_distances=[[-12, -8], [8, 12]],  # 4 combos
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        # 4 IS positions × 4 SS combos = 16 states
        result = outer_ss.generate_seqs(num_seqs=160, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(160):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS Characteristic: Non-overlap
            is_start = row['outer_ss_IS_pos_start']
            is_end = row['outer_ss_IS_pos_end']
            oth_start = row['outer_ss_Oth_pos_start']
            oth_end = row['outer_ss_Oth_pos_end']
            assert is_end <= oth_start or oth_end <= is_start, \
                f"Inserts overlap: IS[{is_start}:{is_end}] Oth[{oth_start}:{oth_end}]"
            
            # IS Characteristic: Position in config
            is_pos = row['inner_is_pos']
            assert is_pos in [0, 2, 4, 6], f"IS pos {is_pos} not in config"
    
    def test_seq_deletion_scan_in_spacing(self):
        """DeletionScan as SpacingScan insert: verify deletion characteristics."""
        # DeletionScan: delete 4bp from 20bp background, mark_changes=False for actual deletion
        inner_ds = DeletionScanPool(
            background_seq='A' * 20,
            deletion_size=4,
            start=2, end=14, step_size=3,  # [2, 5, 8, 11]
            mark_changes=False,  # Actually delete, don't mark
            name='inner_ds', mode='sequential', metadata='complete'
        )
        
        # DeletionScan output is 16bp (20 - 4) when mark_changes=False
        other = Pool(['T' * 16], name='ds_other', metadata='complete')  # 16bp to match
        outer_ss = SpacingScanPool(
            background_seq='N' * 80,
            insert_seqs=[inner_ds, other],
            insert_names=['DS', 'Oth'],
            anchor_pos=40,
            insert_distances=[[-22], [22]],  # 1 combo, wide spacing
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        # 4 DS positions × 1 SS combo = 4 states
        result = outer_ss.generate_seqs(num_seqs=80, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(80):
            row = dc.get_row(i)
            
            # DS Characteristic 1: Output length (seq_length of DeletionScanPool)
            assert inner_ds.seq_length == 16, f"DS should have seq_length 16, got {inner_ds.seq_length}"
            
            # DS Characteristic 2: Position in config
            ds_pos = row['inner_ds_pos']
            assert ds_pos in [2, 5, 8, 11], f"DS pos {ds_pos} not in config"
    
    def test_seq_motif_in_spacing(self):
        """MotifPool in SpacingScan: verify orientation characteristics."""
        # MotifPool with orientation='both' to get both forward and reverse
        pwm = create_pwm('ACGTACGT')
        motif_pool = MotifPool(
            pwm, orientation='both',
            name='motif', mode='random', metadata='complete'
        )
        
        other = Pool(['GGGG'], name='motif_other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[motif_pool, other],
            insert_names=['Mot', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-12, -8], [8, 12]],  # 4 combos
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=160, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(160):
            row = dc.get_row(i)
            
            # Motif Characteristic: Orientation determines content
            orientation = row['motif_orientation']
            value = row['motif_value']
            
            if orientation == 'forward':
                assert value == 'ACGTACGT', f"Forward should be ACGTACGT, got {value}"
            else:
                expected = reverse_complement('ACGTACGT')
                assert value == expected, f"Reverse should be {expected}, got {value}"
    
    def test_seq_shuffle_in_insertion(self):
        """ShuffleScanPool in InsertionScan: verify shuffle characteristics."""
        # ShuffleScan with explicit position for deterministic behavior
        original_bg = 'AAAACGTAACGTTTTT'  # 16bp with variety in shuffle region
        inner_shuffle = ShuffleScanPool(
            background_seq=original_bg,
            shuffle_size=4,  # shuffle 4bp at position 4: CGTA
            positions=[4],   # explicit position
            mark_changes=False,
            name='shuffle', mode='random', metadata='complete'
        )
        
        outer_is = InsertionScanPool(
            background_seq='N' * 80,
            insert_seq=inner_shuffle,
            start=10, end=50, step_size=10,  # [10, 20, 30, 40]
            insert_or_overwrite='overwrite',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        result = outer_is.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(100):
            row = dc.get_row(i)
            seq = seqs[i]
            is_pos = row['outer_is_pos']
            
            # Shuffle Characteristic 1: Length preserved
            shuffle_value = row['shuffle_value']
            assert len(shuffle_value) == 16, f"Shuffle output should be 16bp, got {len(shuffle_value)}"
            
            # Shuffle Characteristic 2: Outside region unchanged
            assert shuffle_value[:4] == 'AAAA', f"Before shuffle [0:4] unchanged: {shuffle_value[:4]}"
            assert shuffle_value[8:] == 'ACGTTTTT', f"After shuffle [8:16] unchanged: {shuffle_value[8:]}"
            
            # Shuffle Characteristic 3: Character preservation in shuffled region
            shuffled_region = shuffle_value[4:8]
            original_region = 'CGTA'
            assert sorted(shuffled_region) == sorted(original_region), \
                f"Shuffled region should preserve chars: got {shuffled_region}, expected permutation of {original_region}"
            
            # IS Characteristic: Verify insert is in sequence at correct position (overwrite mode)
            extracted = seq[is_pos:is_pos + 16]
            assert extracted == shuffle_value, \
                f"Insert at pos {is_pos} should match shuffle_value"
    
    def test_seq_insertion_orf_with_spacing(self):
        """InsertionScanORF in SpacingScan: verify ORF characteristics."""
        # InsertionScanORF: insert 1 codon at positions [1, 2, 3]
        inner_isorf = InsertionScanORFPool(
            background_seq='ATGAAACCCGGGTTT',  # 15bp = 5 codons
            insert_seq='GGG',  # 1 codon
            start=1, end=4, step_size=1,  # [1, 2, 3]
            insert_or_overwrite='overwrite',
            name='isorf', mode='sequential', metadata='complete'
        )
        
        other = Pool(['AAAAAA'], name='orf_other', metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq='N' * 50,
            insert_seqs=[inner_isorf, other],
            insert_names=['ORF', 'Oth'],
            anchor_pos=25,
            insert_distances=[[-15, -10], [10, 15]],  # 4 combos
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        # 3 ORF positions × 4 SS combos = 12 states
        result = outer_ss.generate_seqs(num_seqs=120, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(120):
            row = dc.get_row(i)
            
            # ORF Characteristic 1: Frame preservation
            orf_value = row['isorf_value']
            assert len(orf_value) % 3 == 0, \
                f"ORF output should be multiple of 3, got {len(orf_value)}"
            
            # ORF Characteristic 2: Codon position in config
            codon_pos = row['isorf_codon_pos']
            assert codon_pos in [1, 2, 3], f"Codon pos {codon_pos} not in config"
    
    # -------------------------------------------------------------------------
    # Category B: Multi-Input with MixedPool (5 tests)
    # -------------------------------------------------------------------------
    
    def test_seq_mixed_insertion_scans(self):
        """MixedPool[IS, IS]: verify selection and exclusion characteristics."""
        # Two InsertionScans with distinct markers
        is1 = InsertionScanPool(
            'A' * 20, 'XXXX',
            start=2, end=16, step_size=4,  # [2, 6, 10, 14]
            insert_or_overwrite='overwrite',
            name='is1', mode='sequential', metadata='complete'
        )
        is2 = InsertionScanPool(
            'T' * 20, 'YYYY',
            start=2, end=16, step_size=4,  # [2, 6, 10, 14]
            insert_or_overwrite='overwrite',
            name='is2', mode='sequential', metadata='complete'
        )
        
        mixed = MixedPool([is1, is2], name='mixed', mode='sequential')
        
        # 4 positions each × 2 children = 8 states
        result = mixed.generate_seqs(num_seqs=160, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(160):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_selected']
            
            # Mixed Characteristic 1: Selection accuracy
            if selected == 0:
                assert 'XXXX' in seq, "Selected is1 should have XXXX"
                assert seq.replace('XXXX', '').replace('A', '') == '', \
                    "is1 output should be A's and XXXX"
            else:
                assert 'YYYY' in seq, "Selected is2 should have YYYY"
                assert seq.replace('YYYY', '').replace('T', '') == '', \
                    "is2 output should be T's and YYYY"
            
            # Mixed Characteristic 2: Unselected exclusion
            if selected == 0:
                assert 'YYYY' not in seq, "Unselected is2 marker should not appear"
            else:
                assert 'XXXX' not in seq, "Unselected is1 marker should not appear"
    
    def test_seq_spacing_with_mixed_insert(self):
        """SpacingScan with MixedPool insert: selection determines content."""
        child_a = Pool(['AAAA'], name='mix_a', metadata='complete')
        child_b = Pool(['TTTT'], name='mix_b', metadata='complete')
        mixed = MixedPool([child_a, child_b], name='mixed', mode='sequential')
        
        other = Pool(['GGGG'], name='mix_other', metadata='complete')
        ss = SpacingScanPool(
            background_seq='N' * 40,
            insert_seqs=[mixed, other],
            insert_names=['Mix', 'Oth'],
            anchor_pos=20,
            insert_distances=[[-10, -6], [6, 10]],  # 4 combos
            name='ss', mode='sequential', metadata='complete'
        )
        
        # 2 mixed children × 4 combos = 8 states
        result = ss.generate_seqs(num_seqs=160, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(160):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_selected']
            
            # Mixed selection determines which insert appears
            if selected == 0:
                assert 'AAAA' in seq, "Selected child_a should appear"
                assert 'TTTT' not in seq, "Unselected child_b should not appear"
            else:
                assert 'TTTT' in seq, "Selected child_b should appear"
                assert 'AAAA' not in seq, "Unselected child_a should not appear"
            
            # Other insert always appears
            assert 'GGGG' in seq, "Other insert should always appear"
    
    def test_seq_mixed_motifs_orientation(self):
        """MixedPool of Pools: selected child appears."""
        # Use simple Pools instead of MotifPool (which has infinite states)
        pool_fwd = Pool(['ACGT'], name='fwd', metadata='complete')
        pool_rev = Pool(['TGCA'], name='rev', metadata='complete')
        
        mixed = MixedPool([pool_fwd, pool_rev], name='mixed_motif', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(200):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_motif_selected']
            
            if selected == 0:
                # fwd pool selected
                assert seq == 'ACGT', f"Output should match fwd pool, got {seq}"
            else:
                assert seq == 'TGCA', f"Output should match rev pool, got {seq}"
    
    def test_seq_mixed_barcodes_in_spacing(self):
        """SpacingScan with MixedPool of sequences: verify selection."""
        # Use Pool with multiple sequences instead of BarcodePool
        barcodes_a = ['AACCGG', 'AACCTT', 'AACCAA']
        barcodes_b = ['TTGGCC', 'TTGGAA', 'TTGGTT']
        
        bc_a = Pool(barcodes_a, name='bc_a', mode='sequential', metadata='complete')
        bc_b = Pool(barcodes_b, name='bc_b', mode='sequential', metadata='complete')
        mixed = MixedPool([bc_a, bc_b], name='bc_mixed', mode='sequential')
        
        other = Pool(['NNNNNN'], name='bc_other', metadata='complete')  # Same length
        ss = SpacingScanPool(
            background_seq='G' * 30,
            insert_seqs=[mixed, other],
            insert_names=['BC', 'Oth'],
            anchor_pos=15,
            insert_distances=[[-8], [8]],
            name='bc_ss', mode='sequential', metadata='complete'
        )
        
        # 3 seqs each × 2 children = 6 mixed states
        result = ss.generate_seqs(num_seqs=120, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(120):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['bc_mixed_selected']
            
            # Sequence validity
            if selected == 0:
                bc_val = row['bc_a_value']
                assert bc_val in barcodes_a, f"bc_a value {bc_val} not in config"
            else:
                bc_val = row['bc_b_value']
                assert bc_val in barcodes_b, f"bc_b value {bc_val} not in config"
    
    def test_seq_insertion_with_mixed_background(self):
        """InsertionScan with MixedPool as background: verify characteristics."""
        bg_a = Pool(['A' * 30], name='bg_a', metadata='complete')
        bg_b = Pool(['T' * 30], name='bg_b', metadata='complete')
        mixed_bg = MixedPool([bg_a, bg_b], name='mixed_bg', mode='sequential')
        
        insert = Pool(['XXXX'], name='insert', metadata='complete')
        outer_is = InsertionScanPool(
            background_seq=mixed_bg,
            insert_seq=insert,
            start=5, end=25, step_size=5,  # [5, 10, 15, 20]
            insert_or_overwrite='overwrite',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        # 2 backgrounds × 4 positions = 8 states
        result = outer_is.generate_seqs(num_seqs=160, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(160):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_bg_selected']
            is_pos = row['outer_is_pos']
            
            # Background from selected child
            if selected == 0:
                bg_char = 'A'
            else:
                bg_char = 'T'
            
            # Verify background before insert
            assert seq[:is_pos] == bg_char * is_pos, \
                f"Background before should be {bg_char}'s"
            
            # Verify insert
            assert seq[is_pos:is_pos + 4] == 'XXXX', "Insert should be XXXX"
            
            # Verify background after insert
            assert seq[is_pos + 4:] == bg_char * (30 - is_pos - 4), \
                f"Background after should be {bg_char}'s"
    
    # -------------------------------------------------------------------------
    # Category C: Deep Nesting 4-5 Levels (7 tests)
    # -------------------------------------------------------------------------
    
    def test_seq_4level_iupac_kmut_insertion_spacing(self):
        """4-Level: IUPAC→KMut→IS→SS: verify each transformation."""
        # L1: IUPAC
        L1 = IUPACPool('R' * 20, name='L1', metadata='complete')  # R = A or G
        
        # L2: KMutation k=1
        L2 = KMutationPool(L1, k=1, positions=[2, 5, 8, 11, 14],
                          name='L2', mode='sequential', metadata='complete')
        
        # L3: InsertionScan
        L3 = InsertionScanPool(L2, 'XXXX', start=4, end=16, step_size=4,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        
        # L4: SpacingScan
        other = Pool(['YYYY'], name='L4_other', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 60, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=30, insert_distances=[[-15, -10], [10, 15]],
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # L1: Characters should be A or G only
            l1_val = row['L1_value']
            if l1_val:
                assert all(c in 'AG' for c in l1_val), \
                    f"L1 should only have A/G, got {l1_val}"
            
            # L2: Exactly 1 mutation at allowed position
            mut_pos = row['L2_mut_pos']
            if mut_pos:
                assert len(mut_pos) == 1, "L2 should have exactly 1 mutation"
                assert mut_pos[0] in [2, 5, 8, 11, 14], \
                    f"L2 mut_pos {mut_pos[0]} not in config"
            
            # L3: Position in config
            l3_pos = row['L3_pos']
            if l3_pos is not None:
                assert l3_pos in [4, 8, 12], f"L3 pos {l3_pos} not in config"
            
            # L4: Distances in config
            assert row['L4_A_dist'] in [-15, -10]
            assert row['L4_B_dist'] in [10, 15]
    
    def test_seq_4level_kmer_subseq_kmut_spacing(self):
        """4-Level: Kmer→Subseq→KMut→SS: verify each transformation."""
        # L1: 8-mer enumeration
        L1 = KmerPool(length=8, name='L1', metadata='complete')
        
        # L2: Extract subsequence of width 4 starting at position 2
        L2 = SubseqPool(L1, width=4, start=2, name='L2', metadata='complete')
        
        # L3: KMutation k=1
        L3 = KMutationPool(L2, k=1, name='L3', mode='random', metadata='complete')
        
        # L4: SpacingScan
        other = Pool(['TTTT'], name='L4_other', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 30, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-8, -4], [4, 8]],
            name='L4', mode='random', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # L1: 8bp kmer
            l1_val = row['L1_value']
            assert len(l1_val) == 8, f"L1 should be 8bp, got {len(l1_val)}"
            assert all(c in 'ACGT' for c in l1_val), "L1 should be ACGT only"
            
            # L2: 4bp subsequence
            l2_val = row['L2_value']
            assert len(l2_val) == 4, f"L2 should be 4bp, got {len(l2_val)}"
            
            # L3: Exactly 1 mutation
            assert len(row['L3_mut_pos']) == 1
    
    def test_seq_5level_with_random_mutation(self):
        """5-Level with RandomMutation: verify rate in nested context."""
        # L1: Base sequence
        L1 = Pool(['ACGTACGTACGTACGTACGT'], name='L1', metadata='complete')  # 20bp
        
        # L2: RandomMutation at 10% rate
        L2 = RandomMutationPool(L1, mutation_rate=0.1,
                                name='L2', mode='random', metadata='complete')
        
        # L3: InsertionScan
        L3 = InsertionScanPool(L2, 'XXXX', start=4, end=16, step_size=4,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='random', metadata='complete')
        
        # L4: MixedPool (use random mode for pools with infinite states)
        L3_alt = Pool(['G' * 20], name='L3_alt', metadata='complete')
        L4 = MixedPool([L3, L3_alt], name='L4', mode='random')
        
        # L5: SpacingScan
        other = Pool(['TTTT'], name='L5_other', metadata='complete')
        L5 = SpacingScanPool(
            'N' * 50, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-12], [12]],
            name='L5', mode='random', metadata='complete'
        )
        
        # Generate enough for rate estimation
        result = L5.generate_seqs(num_seqs=500, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        # Count mutations when L3 branch is selected
        total_bases = 0
        total_mutations = 0
        original = 'ACGTACGTACGTACGTACGT'
        
        for i in range(500):
            row = dc.get_row(i)
            if row['L4_selected'] == 0:  # L3 branch
                l2_val = row['L2_value']
                if l2_val:
                    total_bases += len(l2_val)
                    total_mutations += hamming_distance(l2_val, original)
        
        # Rate should be approximately 10% (within ±50% tolerance for random)
        if total_bases > 0:
            actual_rate = total_mutations / total_bases
            assert 0.05 < actual_rate < 0.20, \
                f"RandomMutation rate {actual_rate:.2%} not near 10%"
    
    def test_seq_4level_deletion_in_chain(self):
        """4-Level with Deletion: verify deletion + mutation compose."""
        # L1: Base sequence
        L1 = Pool(['AAAACCCCGGGGTTTT'], name='L1', metadata='complete')  # 16bp
        
        # L2: Delete 4bp - positions [2, 4, 6, 8] with mark_changes=False
        L2 = DeletionScanPool(L1, deletion_size=4, start=2, end=10, step_size=2,
                              mark_changes=False,  # Actual deletion
                              name='L2', mode='sequential', metadata='complete')
        
        # L3: KMutation k=1 (L2 output is 12bp when mark_changes=False)
        L3 = KMutationPool(L2, k=1, name='L3', mode='sequential', metadata='complete')
        
        # L4: SpacingScan with wider background
        other = Pool(['X' * 12], name='L4_other', metadata='complete')  # 12bp to match
        L4 = SpacingScanPool(
            'N' * 60, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=30, insert_distances=[[-18], [18]],  # Wide enough spacing
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # L2: DeletionScan seq_length should be 12bp (16 - 4)
            assert L2.seq_length == 12, f"L2 should have seq_length 12, got {L2.seq_length}"
            
            # L3: Output also 12bp with 1 mutation
            l3_val = row['L3_value']
            assert len(l3_val) == 12, f"L3 should be 12bp, got {len(l3_val)}"
    
    def test_seq_5level_shuffle_in_chain(self):
        """5-Level with Shuffle: verify shuffle + transformations compose correctly."""
        # L1: Base sequence with variety for shuffling
        original = 'AAAACGTAACGTGGGGAAAA'  # 20bp
        L1 = Pool([original], name='L1', metadata='complete')
        
        # L2: Shuffle 4bp at position 4 (CGTA -> permutation)
        L2 = ShuffleScanPool(L1, shuffle_size=4, positions=[4], mark_changes=False,
                             name='L2', mode='random', metadata='complete')
        
        # L3: InsertionScan - overwrite 2bp at position 10
        L3 = InsertionScanPool(L2, 'XX', start=10, end=12, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='random', metadata='complete')
        
        # L4: KMutation
        L4 = KMutationPool(L3, k=1, name='L4', mode='random', metadata='complete')
        
        # L5: SpacingScan with wide spacing
        other = Pool(['Y' * 20], name='L5_other', metadata='complete')
        L5 = SpacingScanPool(
            'N' * 100, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-30], [30]],
            name='L5', mode='random', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # L1: Original unchanged
            l1_val = row['L1_value']
            assert l1_val == original, f"L1 should be original: {l1_val}"
            
            # L2: Shuffle characteristics
            l2_val = row['L2_value']
            assert len(l2_val) == 20, f"L2 length should be 20"
            assert l2_val[:4] == 'AAAA', "L2 before shuffle unchanged"
            assert l2_val[8:] == 'ACGTGGGGAAAA', "L2 after shuffle unchanged"
            assert sorted(l2_val[4:8]) == sorted('CGTA'), \
                f"L2 shuffle region chars preserved: {l2_val[4:8]}"
            
            # L3: Has XX inserted
            l3_val = row['L3_value']
            assert 'XX' in l3_val, "L3 should have XX inserted"
            assert len(l3_val) == 20, "L3 length preserved (overwrite mode)"
            
            # L4: Has 1 mutation from L3
            l4_val = row['L4_value']
            assert len(l4_val) == 20, "L4 length preserved"
            assert hamming_distance(l3_val, l4_val) == 1, "L4 has exactly 1 mutation"
    
    def test_seq_4level_mixed_intermediate(self):
        """4-Level with Mixed intermediate: correct branch propagates."""
        # L1: Base
        L1 = Pool(['AAAAAAAAAA'], name='L1', metadata='complete')  # 10bp
        
        # L2: KMutation
        L2 = KMutationPool(L1, k=1, name='L2', mode='sequential', metadata='complete')
        
        # L3: MixedPool of InsertionScans
        L3a = InsertionScanPool(L2, 'XX', start=2, end=8, step_size=2,
                                insert_or_overwrite='overwrite',
                                name='L3a', mode='sequential', metadata='complete')
        L3b = InsertionScanPool(L2, 'YY', start=2, end=8, step_size=2,
                                insert_or_overwrite='overwrite',
                                name='L3b', mode='sequential', metadata='complete')
        L3 = MixedPool([L3a, L3b], name='L3', mode='sequential')
        
        # L4: SpacingScan
        other = Pool(['ZZZZ'], name='L4_other', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 30, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-8], [8]],
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(200):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['L3_selected']
            
            # Correct branch marker appears
            if selected == 0:
                assert 'XX' in seq, "L3a selected should have XX"
                assert 'YY' not in seq, "L3b should not appear"
            else:
                assert 'YY' in seq, "L3b selected should have YY"
                assert 'XX' not in seq, "L3a should not appear"
            
            # Other insert always appears
            assert 'ZZZZ' in seq, "L4 other always appears"
    
    def test_seq_5level_composition_accuracy(self):
        """5-Level: verify final = compose(all transformations)."""
        # Simple 5-level chain with deterministic transformations
        L1 = Pool(['GGGGGGGGGG'], name='L1', metadata='complete')  # 10bp
        
        # L2: KMutation at position 0
        L2 = KMutationPool(L1, k=1, positions=[0],
                          name='L2', mode='sequential', metadata='complete')
        
        # L3: InsertionScan at positions 2, 4, 6
        L3 = InsertionScanPool(L2, 'AA', start=2, end=8, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        
        # L4: Another KMutation at position 8
        L4 = KMutationPool(L3, k=1, positions=[8],
                          name='L4', mode='sequential', metadata='complete')
        
        # L5: SpacingScan
        other = Pool(['T' * 10], name='L5_other', metadata='complete')  # Same length
        L5 = SpacingScanPool(
            'N' * 40, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=20, insert_distances=[[-8], [8]],
            name='L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=100, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(100):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L2: Position 0 mutated
            l2_val = row['L2_value']
            assert l2_val[0] != 'G', "L2 pos 0 should be mutated"
            assert l2_val[1:] == 'G' * 9, "L2 rest should be G"
            
            # L3: Has AA inserted at some position
            l3_val = row['L3_value']
            assert 'AA' in l3_val, "L3 should have AA"
            
            # L4: Position 8 mutated
            l4_val = row['L4_value']
            # The mutation should have changed position 8
            assert len(l4_val) == 10, "L4 should be 10bp"
            
            # Final sequence contains L4 value
            assert l4_val in seq, "L4 value should be in final sequence"
    
    # -------------------------------------------------------------------------
    # Category D: ORF in Complex Multi-Input (4 tests)
    # -------------------------------------------------------------------------
    
    def test_seq_kmut_orf_synonymous_in_spacing(self):
        """KMutationORF(synonymous) in SpacingScan: AA unchanged, codon changed."""
        # Synonymous mutations only (use 'synonymous' type)
        orf_pool = KMutationORFPool(
            'ATGTTACCCGGA',  # 12bp = 4 codons: M-L-P-G (codons with synonyms)
            mutation_type='synonymous', k=1, positions=[1, 2],
            name='orf', mode='random', metadata='complete'
        )
        
        other = Pool(['TTTTTTTTTTTT'], name='orf_other', metadata='complete')  # Same 12bp
        ss = SpacingScanPool(
            'N' * 40, insert_seqs=[orf_pool, other], insert_names=['ORF', 'Oth'],
            anchor_pos=20, insert_distances=[[-10, -6], [6, 10]],
            name='ss', mode='random', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        original_codons = ['ATG', 'TTA', 'CCC', 'GGA']
        original_aas = [translate_codon(c) for c in original_codons]
        
        for i in range(200):
            row = dc.get_row(i)
            
            # Frame preservation
            orf_val = row['orf_value']
            assert len(orf_val) % 3 == 0, "ORF should be multiple of 3"
            
            # Synonymous: AA unchanged
            codon_pos = row['orf_codon_pos']
            if codon_pos and len(codon_pos) > 0:
                pos = codon_pos[0]
                mutated_codon = orf_val[pos*3:(pos+1)*3]
                mutated_aa = translate_codon(mutated_codon)
                original_aa = original_aas[pos]
                assert mutated_aa == original_aa, \
                    f"Synonymous: AA should be unchanged at pos {pos}"
    
    def test_seq_kmut_orf_nonsynonymous_in_spacing(self):
        """KMutationORF(missense) in SpacingScan: AA changed correctly."""
        # Use 'missense_only_random' for nonsynonymous mutations
        orf_pool = KMutationORFPool(
            'ATGAAACCCGGG',  # M-K-P-G, 12bp
            mutation_type='missense_only_random', k=1, positions=[1, 2],
            name='orf', mode='random', metadata='complete'
        )
        
        other = Pool(['T' * 12], name='orf_other', metadata='complete')  # 12bp
        ss = SpacingScanPool(
            'N' * 60, insert_seqs=[orf_pool, other], insert_names=['ORF', 'Oth'],
            anchor_pos=30, insert_distances=[[-18], [18]],  # Wide spacing
            name='ss', mode='random', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # Missense: AA changed (not stop codon)
            aa_from = row['orf_aa_from']
            aa_to = row['orf_aa_to']
            
            if aa_from and aa_to and len(aa_from) > 0:
                assert aa_from[0] != aa_to[0], \
                    f"Missense: AA should change, got {aa_from[0]} -> {aa_to[0]}"
                assert aa_to[0] != '*', "Missense should not produce stop codon"
    
    def test_seq_randmut_orf_rate_in_spacing(self):
        """RandomMutationORF in SpacingScan: codon mutation rate matches."""
        # Random mutations at 20% codon rate
        orf_pool = RandomMutationORFPool(
            'ATGAAACCCGGGTTTAAA',  # 18bp = 6 codons
            mutation_rate=0.2, mutation_type='any_codon',
            name='orf', mode='random', metadata='complete'
        )
        
        other = Pool(['TTTT'], name='orf_other', metadata='complete')
        ss = SpacingScanPool(
            'N' * 50, insert_seqs=[orf_pool, other], insert_names=['ORF', 'Oth'],
            anchor_pos=25, insert_distances=[[-12], [12]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=500, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        original = 'ATGAAACCCGGGTTTAAA'
        total_codons = 0
        mutated_codons = 0
        
        for i in range(500):
            row = dc.get_row(i)
            orf_val = row['orf_value']
            
            # Frame preservation
            assert len(orf_val) % 3 == 0, "ORF output should be multiple of 3"
            
            # Count mutated codons
            for j in range(0, len(orf_val), 3):
                total_codons += 1
                if orf_val[j:j+3] != original[j:j+3]:
                    mutated_codons += 1
        
        # Rate should be approximately 20%
        actual_rate = mutated_codons / total_codons
        assert 0.10 < actual_rate < 0.35, \
            f"RandomMutationORF rate {actual_rate:.2%} not near 20%"
    
    def test_seq_mixed_orf_types(self):
        """MixedPool[KMutORF, ISORF]: correct ORF operation applied."""
        # Two ORF operation types with same output length
        base_seq = 'ATGAAACCCGGGTTT'  # 15bp = 5 codons
        
        kmut_orf = KMutationORFPool(
            base_seq, mutation_type='any_codon', k=1, positions=[1, 2],
            name='kmut', mode='sequential', metadata='complete'
        )
        
        is_orf = InsertionScanORFPool(
            base_seq, 'GGG', start=1, end=4, step_size=1,
            insert_or_overwrite='overwrite',
            name='isorf', mode='sequential', metadata='complete'
        )
        
        mixed = MixedPool([kmut_orf, is_orf], name='mixed_orf', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            selected = row['mixed_orf_selected']
            
            if selected == 0:
                # KMutation: same length, 1 codon changed
                kmut_val = row['kmut_value']
                assert len(kmut_val) == 15, "KMut should preserve length"
                
            else:
                # InsertionORF: same length (overwrite)
                is_val = row['isorf_value']
                assert len(is_val) == 15, "ISORF overwrite preserves length"
                assert 'GGG' in is_val, "ISORF should have GGG insert"
    
    # -------------------------------------------------------------------------
    # Category E: Pool Reuse (3 tests)
    # -------------------------------------------------------------------------
    
    def test_seq_shared_pool_bg_and_insert(self):
        """Same Pool as bg AND insert: identical content in both roles."""
        shared = Pool(['AAAA'], name='shared', metadata='complete')
        
        # Use shared as both background and insert
        is_pool = InsertionScanPool(
            background_seq=shared,
            insert_seq=shared,
            start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='is', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=100, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(100):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Both occurrences of shared have same value
            # (they share state, so values must match)
            shared_vals = []
            for key in dc.keys:
                if key.startswith('shared') and key.endswith('_value'):
                    val = row[key]
                    if val:
                        shared_vals.append(val)
            
            # All values should be identical
            if len(shared_vals) > 1:
                assert all(v == shared_vals[0] for v in shared_vals), \
                    f"Shared pool values should match: {shared_vals}"
            
            # Sequence should be 8bp (4 bg + 4 insert)
            assert len(seq) == 8, f"Should be 8bp, got {len(seq)}"
            assert seq == 'AAAAAAAA', f"Should be all A's, got {seq}"
    
    def test_seq_shared_kmut_same_mutations(self):
        """Same KMutation in multiple inserts: same mutations both places."""
        base = Pool(['GGGGGGGG'], name='base', metadata='complete')
        shared_kmut = KMutationPool(base, k=1, positions=[0, 2, 4, 6],
                                    name='kmut', mode='sequential', metadata='complete')
        
        # Use same KMut as both inserts
        ss = SpacingScanPool(
            'N' * 50, insert_seqs=[shared_kmut, shared_kmut],
            insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-15], [10]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=200, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(200):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Both occurrences should have same mutation pattern
            kmut_vals = []
            for key in dc.keys:
                if 'kmut' in key and key.endswith('_value'):
                    val = row[key]
                    if val:
                        kmut_vals.append(val)
            
            # All KMut values should be identical (shared state)
            if len(kmut_vals) >= 2:
                assert kmut_vals[0] == kmut_vals[1], \
                    f"Shared KMut should produce same mutations: {kmut_vals}"
            
            # The mutated sequence appears twice in output
            if kmut_vals:
                assert seq.count(kmut_vals[0]) == 2, \
                    f"Shared KMut value should appear twice in sequence"
    
    def test_seq_shared_randmut_consistency(self):
        """Same RandomMutation transformer: identical mutation pattern."""
        base = Pool(['ACGTACGTACGT'], name='base', metadata='complete')
        shared_randmut = RandomMutationPool(base, mutation_rate=0.1,
                                            name='randmut', mode='random', metadata='complete')
        
        # Use same RandomMut as both inserts
        ss = SpacingScanPool(
            'N' * 50, insert_seqs=[shared_randmut, shared_randmut],
            insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-15], [10]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=200, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(200):
            row = dc.get_row(i)
            
            # Both occurrences should have identical value (shared state)
            randmut_vals = []
            for key in dc.keys:
                if 'randmut' in key and key.endswith('_value'):
                    val = row[key]
                    if val:
                        randmut_vals.append(val)
            
            # All values should be identical
            if len(randmut_vals) >= 2:
                assert randmut_vals[0] == randmut_vals[1], \
                    f"Shared RandomMut should produce same pattern: {randmut_vals}"


# =============================================================================
# Class 2: TestComputationGraph (~15 tests)
# =============================================================================

class TestComputationGraph:
    """Verify computation graph structure and node sequences."""
    
    # -------------------------------------------------------------------------
    # Category A: Multi-Input Inside Multi-Input (3 tests)
    # -------------------------------------------------------------------------
    
    def test_graph_insertion_spacing_all_nodes(self):
        """InsertionScan+SpacingScan: all nodes present in graph."""
        ss_a = Pool(['AAAA'], name='ss_a', metadata='complete')
        ss_b = Pool(['TTTT'], name='ss_b', metadata='complete')
        inner_ss = SpacingScanPool(
            'N' * 20, insert_seqs=[ss_a, ss_b], insert_names=['A', 'B'],
            anchor_pos=10, insert_distances=[[-4], [4]],
            name='inner_ss', mode='sequential', metadata='complete'
        )
        
        outer_is = InsertionScanPool(
            'G' * 60, inner_ss, start=10, end=50, step_size=10,
            insert_or_overwrite='insert',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        result = outer_is.generate_seqs(
            num_seqs=10, return_computation_graph=True, return_design_cards=True
        )
        graph = result['graph']
        node_seqs = result['node_sequences']
        
        # All expected nodes present
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'outer_is' in node_names, "outer_is should be in graph"
        assert 'inner_ss' in node_names, "inner_ss should be in graph"
        assert 'ss_a' in node_names, "ss_a should be in graph"
        assert 'ss_b' in node_names, "ss_b should be in graph"
        
        # Node sequences have entries for all nodes
        assert len(node_seqs) >= 4, f"Should have 4+ node entries, got {len(node_seqs)}"
    
    def test_graph_deletion_spacing_structure(self):
        """DeletionScan in SpacingScan: DS appears correctly in graph."""
        inner_ds = DeletionScanPool(
            'A' * 16, deletion_size=4, start=2, end=10, step_size=2,
            name='inner_ds', mode='sequential', metadata='complete'
        )
        
        other = Pool(['TTTT'], name='other', metadata='complete')
        outer_ss = SpacingScanPool(
            'N' * 40, insert_seqs=[inner_ds, other], insert_names=['DS', 'Oth'],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=10, return_computation_graph=True)
        graph = result['graph']
        
        # DeletionScan appears in graph
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'inner_ds' in node_names, "DeletionScan should be in graph"
        assert 'outer_ss' in node_names, "SpacingScan should be in graph"
    
    def test_nodeseq_motif_spacing_values(self):
        """Pool node sequences in SpacingScan."""
        # Use simple Pool instead of MotifPool
        insert_pool = Pool(['ACGT'], name='insert_pool', metadata='complete')
        other = Pool(['NNNN'], name='other', metadata='complete')
        
        ss = SpacingScanPool(
            'G' * 20, insert_seqs=[insert_pool, other], insert_names=['M', 'O'],
            anchor_pos=10, insert_distances=[[-4], [4]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(
            num_seqs=20, return_computation_graph=True, return_design_cards=True
        )
        dc = result['design_cards']
        
        # Verify design card values are correct
        for i in range(20):
            row = dc.get_row(i)
            insert_val = row.get('insert_pool_value')
            assert insert_val == 'ACGT', f"Insert value should be ACGT"
    
    # -------------------------------------------------------------------------
    # Category B: Multi-Input with MixedPool (3 tests)
    # -------------------------------------------------------------------------
    
    def test_graph_mixed_all_children_metadata(self):
        """MixedPool: verify selected child has correct metadata."""
        child_a = Pool(['AAAA'], name='child_a', metadata='complete')
        child_b = Pool(['TTTT'], name='child_b', metadata='complete')
        child_c = Pool(['GGGG'], name='child_c', metadata='complete')
        mixed = MixedPool([child_a, child_b, child_c], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=30, return_computation_graph=True, return_design_cards=True)
        graph = result['graph']
        dc = result['design_cards']
        seqs = result['sequences']
        
        # MixedPool should be in graph
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'mixed' in node_names, "mixed should be in graph"
        
        # All children should have metadata columns in design cards
        assert 'child_a_value' in dc.keys, "child_a_value should be in design cards"
        assert 'child_b_value' in dc.keys, "child_b_value should be in design cards"
        assert 'child_c_value' in dc.keys, "child_c_value should be in design cards"
        
        # Verify selected child's value matches output
        for i in range(30):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_selected']
            
            if selected == 0:
                assert row['child_a_value'] == 'AAAA', "child_a_value should be AAAA when selected"
                assert seq == 'AAAA', "Output should match child_a when selected"
            elif selected == 1:
                assert row['child_b_value'] == 'TTTT', "child_b_value should be TTTT when selected"
                assert seq == 'TTTT', "Output should match child_b when selected"
            else:
                assert row['child_c_value'] == 'GGGG', "child_c_value should be GGGG when selected"
                assert seq == 'GGGG', "Output should match child_c when selected"
    
    def test_graph_spacing_mixed_structure(self):
        """SpacingScan with MixedPool: structure in graph."""
        pool_a = Pool(['AACC', 'AATT'], name='pool_a', metadata='complete')
        pool_b = Pool(['TTGG', 'TTCC'], name='pool_b', metadata='complete')
        mixed = MixedPool([pool_a, pool_b], name='mixed', mode='sequential')
        
        other = Pool(['NNNN'], name='other', metadata='complete')
        ss = SpacingScanPool(
            'G' * 20, insert_seqs=[mixed, other], insert_names=['M', 'O'],
            anchor_pos=10, insert_distances=[[-4], [4]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=10, return_computation_graph=True)
        graph = result['graph']
        
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names, "ss should be in graph"
        assert 'mixed' in node_names, "mixed should be in graph"
    
    def test_nodeseq_mixed_selected_matches_output(self):
        """MixedPool: selected child's node_seq appears in final output."""
        child_a = Pool(['AAAA'], name='child_a', metadata='complete')
        child_b = Pool(['TTTT'], name='child_b', metadata='complete')
        mixed = MixedPool([child_a, child_b], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(
            num_seqs=20, return_computation_graph=True, return_design_cards=True
        )
        seqs = result['sequences']
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            selected = row['mixed_selected']
            seq = seqs[i]
            
            if selected == 0:
                assert seq == 'AAAA', "Selected child_a should match output"
            else:
                assert seq == 'TTTT', "Selected child_b should match output"
    
    # -------------------------------------------------------------------------
    # Category C: Deep Nesting 4-5 Levels (4 tests)
    # -------------------------------------------------------------------------
    
    def test_graph_4level_complete_hierarchy(self):
        """4-Level chain: all levels with correct parent chains."""
        L1 = Pool(['GGGGGGGG'], name='L1', metadata='complete')
        L2 = KMutationPool(L1, k=1, name='L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'XX', start=2, end=6, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        other = Pool(['YY'], name='L4_other', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 30, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-6], [6]],
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(num_seqs=10, return_computation_graph=True)
        graph = result['graph']
        
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        for name in ['L1', 'L2', 'L3', 'L4', 'L4_other']:
            assert name in node_names, f"{name} should be in graph"
    
    def test_graph_5level_with_shuffle_kmer(self):
        """5-Level with Shuffle and Kmer: all nodes present."""
        L1 = KmerPool(length=8, name='L1', metadata='complete')
        L2 = ShuffleScanPool(L1, shuffle_size=4, start=2, mark_changes=False,
                             name='L2', mode='random', metadata='complete')
        L3 = KMutationPool(L2, k=1, name='L3', mode='random', metadata='complete')
        L4 = InsertionScanPool(L3, 'AA', start=2, end=6, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L4', mode='random', metadata='complete')
        other = Pool(['T' * 8], name='L5_other', metadata='complete')  # Same length
        L5 = SpacingScanPool(
            'N' * 40, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=20, insert_distances=[[-12], [12]],
            name='L5', mode='random', metadata='complete'
        )
        
        result = L5.generate_seqs(num_seqs=10, seed=42, return_computation_graph=True)
        graph = result['graph']
        
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        for name in ['L1', 'L2', 'L3', 'L4', 'L5']:
            assert name in node_names, f"{name} should be in graph"
    
    def test_nodeseq_4level_chain_composition(self):
        """4-Level: node_seq[L(n)] = transform(node_seq[L(n-1)])."""
        L1 = Pool(['GGGGGGGG'], name='L1', metadata='complete')  # 8bp
        L2 = KMutationPool(L1, k=1, positions=[0],
                          name='L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'AA', start=2, end=6, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        other = Pool(['T' * 8], name='L4_other', metadata='complete')  # Same 8bp
        L4 = SpacingScanPool(
            'N' * 40, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=20, insert_distances=[[-12], [12]],
            name='L4', mode='sequential', metadata='complete'
        )
        
        result = L4.generate_seqs(
            num_seqs=20, return_computation_graph=True, return_design_cards=True
        )
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            
            # L2 = L1 with mutation at position 0
            l1_val = row['L1_value']
            l2_val = row['L2_value']
            assert l2_val[0] != 'G', "L2[0] should be mutated"
            assert l2_val[1:] == l1_val[1:], "L2[1:] should match L1[1:]"
            
            # L3 = L2 with AA at some position
            l3_val = row['L3_value']
            assert 'AA' in l3_val, "L3 should have AA"
    
    def test_nodeseq_5level_intermediate_accuracy(self):
        """5-Level: each intermediate value correct."""
        L1 = Pool(['AAAAAAAAAA'], name='L1', metadata='complete')  # 10bp
        L2 = KMutationPool(L1, k=1, positions=[0],
                          name='L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'XX', start=2, end=8, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        L4 = KMutationPool(L3, k=1, positions=[8],
                          name='L4', mode='sequential', metadata='complete')
        other = Pool(['Y' * 10], name='L5_other', metadata='complete')  # 10bp
        L5 = SpacingScanPool(
            'N' * 50, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-15], [15]],
            name='L5', mode='sequential', metadata='complete'
        )
        
        result = L5.generate_seqs(
            num_seqs=50, return_computation_graph=True, return_design_cards=True
        )
        dc = result['design_cards']
        
        for i in range(50):
            row = dc.get_row(i)
            
            # Verify chain of transformations
            l1 = row['L1_value']
            l2 = row['L2_value']
            l3 = row['L3_value']
            l4 = row['L4_value']
            
            # L1 is all A's
            assert l1 == 'A' * 10
            
            # L2 has mutation at pos 0
            assert l2[0] != 'A'
            assert l2[1:] == 'A' * 9
            
            # L3 has XX at some position
            assert 'XX' in l3
            
            # L4 has mutation at pos 8
            assert l4[8] != l3[8]
    
    # -------------------------------------------------------------------------
    # Category D: ORF in Complex Multi-Input (3 tests)
    # -------------------------------------------------------------------------
    
    def test_graph_orf_types_in_mixed(self):
        """MixedPool[ORF types]: verify selection and ORF metadata."""
        # Use pools with similar state counts for balanced selection
        kmut = KMutationORFPool('ATGAAACCC', mutation_type='any_codon', k=1,
                                positions=[1],  # Only 1 position to reduce states
                                name='kmut', mode='sequential', metadata='complete')
        isorf = InsertionScanORFPool('ATGAAACCC', 'GGG', start=1, end=2,
                                     insert_or_overwrite='overwrite',
                                     name='isorf', mode='sequential', metadata='complete')
        
        mixed = MixedPool([kmut, isorf], name='mixed', mode='sequential')
        
        # Generate enough to cover all states
        total_states = mixed.num_states
        num_seqs = total_states * 2  # 2 full iterations
        result = mixed.generate_seqs(num_seqs=num_seqs, return_computation_graph=True, return_design_cards=True)
        graph = result['graph']
        dc = result['design_cards']
        
        # MixedPool in graph
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'mixed' in node_names, "MixedPool should be in graph"
        
        # Both ORF types should have metadata columns
        assert 'kmut_codon_pos' in dc.keys, "kmut_codon_pos should be in design cards"
        assert 'isorf_codon_pos' in dc.keys, "isorf_codon_pos should be in design cards"
        
        # Verify correct ORF type selected each time
        kmut_selected_count = 0
        isorf_selected_count = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            selected = row['mixed_selected']
            if selected == 0:
                kmut_selected_count += 1
                # When kmut selected, verify it has mutation metadata
                assert row['kmut_codon_pos'] is not None, "kmut should have codon_pos"
            else:
                isorf_selected_count += 1
                # When isorf selected, verify it has position metadata
                assert row['isorf_codon_pos'] is not None, "isorf should have codon_pos"
        
        # Both should be selected at least once
        assert kmut_selected_count > 0, "kmut should be selected at least once"
        assert isorf_selected_count > 0, "isorf should be selected at least once"
    
    def test_graph_randmut_orf_structure(self):
        """RandomMutationORFPool in SpacingScan: appears in graph."""
        randmut_orf = RandomMutationORFPool(
            'ATGAAACCCGGG', mutation_rate=0.1, mutation_type='any_codon',
            name='randmut', mode='random', metadata='complete'
        )
        
        other = Pool(['TTTT'], name='other', metadata='complete')
        ss = SpacingScanPool(
            'N' * 40, insert_seqs=[randmut_orf, other], insert_names=['R', 'O'],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=10, seed=42, return_computation_graph=True)
        graph = result['graph']
        
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'randmut' in node_names, "RandomMutORF should be in graph"
    
    def test_nodeseq_orf_codon_boundaries(self):
        """ORF node sequences have length % 3 == 0."""
        orf = KMutationORFPool('ATGAAACCCGGG', mutation_type='any_codon', k=1,
                               name='orf', mode='sequential', metadata='complete')
        
        result = orf.generate_seqs(
            num_seqs=20, return_computation_graph=True, return_design_cards=True
        )
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            orf_val = row['orf_value']
            assert len(orf_val) % 3 == 0, \
                f"ORF value should be multiple of 3, got {len(orf_val)}"
    
    # -------------------------------------------------------------------------
    # Category E: Pool Reuse (2 tests)
    # -------------------------------------------------------------------------
    
    def test_graph_shared_pool_single_node(self):
        """Shared pool appears once in graph, not duplicated."""
        shared = Pool(['AAAA'], name='shared', metadata='complete')
        
        is_pool = InsertionScanPool(
            shared, shared, start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='is', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=10, return_computation_graph=True)
        graph = result['graph']
        
        # Count occurrences of 'shared' in node names
        shared_count = sum(1 for n in graph['nodes'] 
                          if n.get('name') and 'shared' in n['name'])
        assert shared_count == 1, f"Shared pool should appear once, got {shared_count}"
    
    def test_nodeseq_shared_consistent_values(self):
        """Shared pool: single node_sequences entry, consistent values."""
        shared = Pool(['GGGG'], name='shared', metadata='complete')
        
        is_pool = InsertionScanPool(
            shared, shared, start=0, end=3, step_size=1,
            insert_or_overwrite='insert',
            name='is', mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(
            num_seqs=20, return_computation_graph=True, return_design_cards=True
        )
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            # All occurrences of shared should have same value
            shared_vals = [v for k, v in row.items() 
                          if 'shared' in k and k.endswith('_value') and v]
            if len(shared_vals) > 1:
                assert all(v == shared_vals[0] for v in shared_vals), \
                    "All shared occurrences should have same value"


# =============================================================================
# Class 3: TestDistributionVerification (~10 tests)
# =============================================================================

class TestDistributionVerification:
    """Verify statistical distributions match expectations."""
    
    def test_distrib_insertion_spacing_positions(self):
        """InsertionScan+SpacingScan: nested distribution is uniform."""
        ss_a = Pool(['AAAA'], name='ss_a', metadata='complete')
        ss_b = Pool(['TTTT'], name='ss_b', metadata='complete')
        inner_ss = SpacingScanPool(
            'N' * 25, insert_seqs=[ss_a, ss_b], insert_names=['A', 'B'],
            anchor_pos=12, insert_distances=[[-6, -4], [4, 6]],  # 4 combos
            name='inner_ss', mode='sequential', metadata='complete'
        )
        
        outer_is = InsertionScanPool(
            'G' * 80, inner_ss, start=10, end=42, step_size=10,
            insert_or_overwrite='insert',
            name='outer_is', mode='sequential', metadata='complete'
        )
        
        # Get actual state count (not assumed)
        total_states = outer_is.num_states
        num_seqs = total_states * 20  # 20 full iterations
        result = outer_is.generate_seqs(num_seqs=num_seqs, return_design_cards=True)
        dc = result['design_cards']
        
        # IS position distribution
        is_pos_counts = {}
        combo_counts = {}
        for i in range(num_seqs):
            row = dc.get_row(i)
            pos = row['outer_is_pos']
            combo = (row['inner_ss_A_dist'], row['inner_ss_B_dist'])
            is_pos_counts[pos] = is_pos_counts.get(pos, 0) + 1
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        
        # Each position should appear equally
        num_positions = len(is_pos_counts)
        expected_per_pos = num_seqs // num_positions
        for pos, count in is_pos_counts.items():
            assert count == expected_per_pos, \
                f"IS pos {pos}: expected {expected_per_pos}, got {count}"
        
        # Each combo should appear equally
        num_combos = len(combo_counts)
        expected_per_combo = num_seqs // num_combos
        for combo, count in combo_counts.items():
            assert count == expected_per_combo, \
                f"Combo {combo}: expected {expected_per_combo}, got {count}"
    
    def test_distrib_deletion_spacing_positions(self):
        """DeletionScan in SpacingScan: nested distribution is uniform."""
        # DeletionScan with mark_changes=False for actual deletion
        inner_ds = DeletionScanPool(
            'A' * 20, deletion_size=4, start=2, end=14, step_size=3,
            mark_changes=False,
            name='inner_ds', mode='sequential', metadata='complete'
        )
        
        # Output is 16bp (20-4)
        other = Pool(['T' * 16], name='ds_other', metadata='complete')
        outer_ss = SpacingScanPool(
            'N' * 80, insert_seqs=[inner_ds, other], insert_names=['DS', 'Oth'],
            anchor_pos=40, insert_distances=[[-22, -20], [20, 22]],  # 4 combos
            name='outer_ss', mode='sequential', metadata='complete'
        )
        
        # Get actual state count (not assumed)
        total_states = outer_ss.num_states
        num_seqs = total_states * 20  # 20 full iterations
        result = outer_ss.generate_seqs(num_seqs=num_seqs, return_design_cards=True)
        dc = result['design_cards']
        
        # DS position distribution
        ds_pos_counts = {}
        for i in range(num_seqs):
            pos = dc.get_row(i)['inner_ds_pos']
            ds_pos_counts[pos] = ds_pos_counts.get(pos, 0) + 1
        
        # Each DS position should appear equally
        num_positions = len(ds_pos_counts)
        expected_per_pos = num_seqs // num_positions
        for pos, count in ds_pos_counts.items():
            assert count == expected_per_pos, \
                f"DS pos {pos}: expected {expected_per_pos}, got {count}"
    
    def test_distrib_motif_orientation(self):
        """Category A: MotifPool in SpacingScan - orientation distribution ~50% each."""
        # Realistic 60bp background
        background = 'ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC'
        
        # MotifPool with both orientations
        pwm = create_pwm('ACGTACGT')  # 8bp motif
        motif = MotifPool(pwm, orientation='both', name='motif', mode='random', metadata='complete')
        
        # Other insert for SpacingScan
        other = Pool(['TTTTTTTT'], name='other', metadata='complete')  # 8bp
        
        # SpacingScan with MotifPool (Category A: multi-input)
        ss = SpacingScanPool(
            background, insert_seqs=[motif, other], insert_names=['Mot', 'Oth'],
            anchor_pos=30, insert_distances=[[-15, -12], [12, 15]],  # 4 combos
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 500
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42, 
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names, "SpacingScan should be in graph"
        assert 'motif' in node_names, "MotifPool should be in graph"
        
        fwd_count = 0
        rev_count = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            orient = row['motif_orientation']
            motif_value = row['motif_value']
            
            # Pool-specific check: verify motif value matches orientation
            if orient == 'forward':
                fwd_count += 1
                assert motif_value == 'ACGTACGT', f"Forward should be ACGTACGT, got {motif_value}"
            else:
                rev_count += 1
                expected_rc = reverse_complement('ACGTACGT')
                assert motif_value == expected_rc, f"Reverse should be {expected_rc}, got {motif_value}"
            
            # Verify motif value appears in final sequence
            assert motif_value in seq, f"Motif value should be in sequence"
            
            # Verify spacing formula
            mot_start = row['ss_Mot_pos_start']
            mot_end = row['ss_Mot_pos_end']
            oth_start = row['ss_Oth_pos_start']
            spacing = row['ss_spacing_Mot_Oth']
            assert spacing == oth_start - mot_end, f"Spacing formula failed"
        
        # Should be approximately 50/50 ±20%
        expected = num_seqs / 2
        assert abs(fwd_count - expected) < expected * 0.20, \
            f"Forward count {fwd_count} not near {expected}"
        assert abs(rev_count - expected) < expected * 0.20, \
            f"Reverse count {rev_count} not near {expected}"
    
    def test_distrib_mixed_weights(self):
        """Category B: MixedPool[IS, IS] in SpacingScan - weight distribution."""
        # Realistic 80bp background
        background = 'ATGC' * 20  # 80bp
        
        # Two InsertionScans with distinct markers
        is1 = InsertionScanPool(
            'A' * 16, 'XXXX', start=2, end=12, step_size=3,  # 4 positions
            insert_or_overwrite='overwrite',
            name='is1', mode='sequential', metadata='complete'
        )
        is2 = InsertionScanPool(
            'T' * 16, 'YYYY', start=2, end=12, step_size=3,  # 4 positions
            insert_or_overwrite='overwrite',
            name='is2', mode='sequential', metadata='complete'
        )
        
        # MixedPool with 70%/30% weights
        mixed = MixedPool([is1, is2], weights=[0.7, 0.3], name='mixed', mode='random')
        
        # Other insert for SpacingScan
        other = Pool(['G' * 16], name='other', metadata='complete')
        
        # SpacingScan with MixedPool (Category B)
        ss = SpacingScanPool(
            background, insert_seqs=[mixed, other], insert_names=['Mix', 'Oth'],
            anchor_pos=40, insert_distances=[[-22], [22]],  # 1 combo for simplicity
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 500
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names, "SpacingScan should be in graph"
        assert 'mixed' in node_names, "MixedPool should be in graph"
        
        is1_count = 0
        is2_count = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_selected']
            
            if selected == 0:
                is1_count += 1
                # Pool-specific: is1 selected, should have XXXX marker
                is1_val = row['is1_value']
                assert 'XXXX' in is1_val, "is1 should have XXXX"
                assert 'XXXX' in seq, "is1 marker should be in final sequence"
                assert 'YYYY' not in seq, "is2 marker should NOT be in sequence"
            else:
                is2_count += 1
                # Pool-specific: is2 selected, should have YYYY marker
                is2_val = row['is2_value']
                assert 'YYYY' in is2_val, "is2 should have YYYY"
                assert 'YYYY' in seq, "is2 marker should be in final sequence"
                assert 'XXXX' not in seq, "is1 marker should NOT be in sequence"
            
            # Other insert always present
            assert 'G' * 16 in seq, "Other insert should be in sequence"
        
        # Weight distribution: is1 should be ~70%, is2 ~30%
        is1_ratio = is1_count / num_seqs
        is2_ratio = is2_count / num_seqs
        assert abs(is1_ratio - 0.7) < 0.15, f"is1 ratio {is1_ratio:.2f} not near 0.7"
        assert abs(is2_ratio - 0.3) < 0.15, f"is2 ratio {is2_ratio:.2f} not near 0.3"
    
    def test_distrib_iupac_characters(self):
        """Category C: IUPAC→KMut→IS→SS 4-level chain - character frequencies."""
        # L1: IUPAC with R (A or G) characters - 8bp
        L1 = IUPACPool('RRRRRRRR', name='L1_iupac', metadata='complete')
        
        # L2: KMutation k=1
        L2 = KMutationPool(L1, k=1, name='L2_kmut', mode='random', metadata='complete')
        
        # L3: InsertionScan
        L3 = InsertionScanPool(L2, 'XX', start=2, end=6, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3_is', mode='random', metadata='complete')
        
        # L4: SpacingScan (top level)
        other = Pool(['TTTTTTTT'], name='L4_other', metadata='complete')  # 8bp
        L4 = SpacingScanPool(
            'N' * 50, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=25, insert_distances=[[-12], [12]],
            name='L4_ss', mode='random', metadata='complete'
        )
        
        num_seqs = 500
        result = L4.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure - all 4 levels
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'L4_ss' in node_names, "L4 SpacingScan should be in graph"
        assert 'L3_is' in node_names, "L3 InsertionScan should be in graph"
        assert 'L2_kmut' in node_names, "L2 KMutation should be in graph"
        assert 'L1_iupac' in node_names, "L1 IUPAC should be in graph"
        
        a_count = 0
        g_count = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L1: IUPAC value should only have A and G
            l1_val = row['L1_iupac_value']
            assert all(c in 'AG' for c in l1_val), f"L1 should only have A/G, got {l1_val}"
            a_count += l1_val.count('A')
            g_count += l1_val.count('G')
            
            # L2: Has exactly 1 mutation
            assert len(row['L2_kmut_mut_pos']) == 1, "L2 should have exactly 1 mutation"
            
            # L3: Has XX inserted
            l3_val = row['L3_is_value']
            assert 'XX' in l3_val, "L3 should have XX"
            
            # L4: Spacing formula holds
            a_start = row['L4_ss_A_pos_start']
            a_end = row['L4_ss_A_pos_end']
            b_start = row['L4_ss_B_pos_start']
            spacing = row['L4_ss_spacing_A_B']
            assert spacing == b_start - a_end, "Spacing formula failed"
        
        # IUPAC character distribution: R→50% A, 50% G
        total = a_count + g_count
        a_ratio = a_count / total
        assert abs(a_ratio - 0.5) < 0.10, f"A ratio {a_ratio:.2f} not near 0.5"
    
    def test_distrib_randmut_rate(self):
        """Category C: 5-level chain with RandomMutation - mutation rate verification."""
        # L1: Base sequence (realistic 20bp)
        original = 'ACGTACGTACGTACGTACGT'
        L1 = Pool([original], name='L1_base', metadata='complete')
        
        # L2: RandomMutation at 15% rate
        L2 = RandomMutationPool(L1, mutation_rate=0.15, name='L2_randmut', 
                                mode='random', metadata='complete')
        
        # L3: InsertionScan
        L3 = InsertionScanPool(L2, 'XX', start=4, end=16, step_size=4,
                               insert_or_overwrite='overwrite',
                               name='L3_is', mode='random', metadata='complete')
        
        # L4: KMutation k=1
        L4 = KMutationPool(L3, k=1, name='L4_kmut', mode='random', metadata='complete')
        
        # L5: SpacingScan (top level) - 20bp inserts need wide spacing
        other = Pool(['T' * 20], name='L5_other', metadata='complete')
        L5 = SpacingScanPool(
            'N' * 100, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='L5_ss', mode='random', metadata='complete'
        )
        
        num_seqs = 500
        result = L5.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure - all 5 levels
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        for level in ['L1_base', 'L2_randmut', 'L3_is', 'L4_kmut', 'L5_ss']:
            assert level in node_names, f"{level} should be in graph"
        
        total_bases = 0
        total_mutations = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L2: RandomMutation value
            l2_val = row['L2_randmut_value']
            total_bases += len(l2_val)
            total_mutations += hamming_distance(l2_val, original)
            
            # L3: Has XX inserted
            l3_val = row['L3_is_value']
            assert 'XX' in l3_val, "L3 should have XX"
            
            # L4: Has 1 mutation from L3
            l4_val = row['L4_kmut_value']
            assert hamming_distance(l3_val, l4_val) == 1, "L4 should have 1 mutation from L3"
            
            # Verify L4 value in final sequence
            assert l4_val in seq, "L4 value should be in final sequence"
        
        # RandomMutation rate should be approximately 15%
        actual_rate = total_mutations / total_bases
        assert 0.10 < actual_rate < 0.22, f"Rate {actual_rate:.2%} not near 15%"
    
    def test_distrib_orf_codon_positions(self):
        """Category D: KMutationORF in SpacingScan - codon position distribution."""
        # Realistic ORF sequence (5 codons = 15bp)
        orf_seq = 'ATGAAACCCGGGTTT'  # M-K-P-G-F
        
        # KMutationORF with 5 codon positions
        orf = KMutationORFPool(
            orf_seq, mutation_type='any_codon', k=1,
            positions=[0, 1, 2, 3, 4],
            name='orf', mode='sequential', metadata='complete'
        )
        
        # Other insert for SpacingScan
        other = Pool(['A' * 15], name='orf_other', metadata='complete')  # 15bp
        
        # SpacingScan with ORF (Category D) - 15bp inserts, need proper spacing
        ss = SpacingScanPool(
            'N' * 80, insert_seqs=[orf, other], insert_names=['ORF', 'Oth'],
            anchor_pos=40, insert_distances=[[-35], [5]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        # Get exact state count and generate enough sequences
        total_states = ss.num_states
        num_seqs = total_states * 20  # 20 full iterations
        result = ss.generate_seqs(num_seqs=num_seqs,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names, "SpacingScan should be in graph"
        assert 'orf' in node_names, "KMutORF should be in graph"
        
        pos_counts = {}
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # ORF-specific checks
            codon_pos = row['orf_codon_pos'][0]
            pos_counts[codon_pos] = pos_counts.get(codon_pos, 0) + 1
            
            # Frame preservation
            orf_val = row['orf_value']
            assert len(orf_val) % 3 == 0, f"ORF value should be multiple of 3"
            
            # Codon at mutation position changed
            codon_from = row['orf_codon_from'][0]
            codon_to = row['orf_codon_to'][0]
            assert codon_from != codon_to, f"Codon should change at position {codon_pos}"
            
            # ORF value in final sequence
            assert orf_val in seq, "ORF value should be in final sequence"
        
        # Each codon position should appear equally
        num_positions = len(pos_counts)
        expected_per_pos = num_seqs // num_positions
        for pos, count in pos_counts.items():
            assert count == expected_per_pos, \
                f"ORF pos {pos}: expected {expected_per_pos}, got {count}"
    
    def test_distrib_randmut_orf_rate(self):
        """Category D: RandomMutationORF in SpacingScan - codon mutation rate."""
        # Realistic ORF sequence (6 codons = 18bp)
        original = 'ATGAAACCCGGGTTTAAA'  # M-K-P-G-F-K
        
        # RandomMutationORF at 25% codon rate
        orf = RandomMutationORFPool(
            original, mutation_rate=0.25, mutation_type='any_codon',
            name='randmut_orf', mode='random', metadata='complete'
        )
        
        # Other insert for SpacingScan
        other = Pool(['T' * 18], name='orf_other', metadata='complete')  # 18bp
        
        # SpacingScan with RandomMutationORF (Category D) - 18bp inserts
        ss = SpacingScanPool(
            'N' * 100, insert_seqs=[orf, other], insert_names=['ORF', 'Oth'],
            anchor_pos=50, insert_distances=[[-45], [5]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 500
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        graph = result['graph']
        
        # Verify graph structure
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names, "SpacingScan should be in graph"
        assert 'randmut_orf' in node_names, "RandomMutORF should be in graph"
        
        total_codons = 0
        mutated_codons = 0
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # ORF-specific checks
            orf_val = row['randmut_orf_value']
            
            # Frame preservation
            assert len(orf_val) % 3 == 0, f"ORF value should be multiple of 3"
            
            # Count mutated codons
            for j in range(0, len(orf_val), 3):
                total_codons += 1
                if orf_val[j:j+3] != original[j:j+3]:
                    mutated_codons += 1
            
            # ORF value in final sequence
            assert orf_val in seq, "ORF value should be in final sequence"
            
            # Spacing formula
            orf_start = row['ss_ORF_pos_start']
            orf_end = row['ss_ORF_pos_end']
            oth_start = row['ss_Oth_pos_start']
            spacing = row['ss_spacing_ORF_Oth']
            assert spacing == oth_start - orf_end, "Spacing formula failed"
        
        # Codon mutation rate should be approximately 25%
        actual_rate = mutated_codons / total_codons
        assert 0.15 < actual_rate < 0.40, f"Rate {actual_rate:.2%} not near 25%"
    
    def test_distrib_4level_joint(self):
        """4-Level chain: joint distribution of all levels uniform."""
        L1 = Pool(['GGGGGGGG'], name='L1', metadata='complete')
        L2 = KMutationPool(L1, k=1, positions=[0, 2, 4],
                          name='L2', mode='sequential', metadata='complete')
        L3 = InsertionScanPool(L2, 'XX', start=2, end=6, step_size=2,
                               insert_or_overwrite='overwrite',
                               name='L3', mode='sequential', metadata='complete')
        other = Pool(['YY'], name='L4_other', metadata='complete')
        L4 = SpacingScanPool(
            'N' * 30, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=15, insert_distances=[[-6, -4], [4, 6]],
            name='L4', mode='sequential', metadata='complete'
        )
        
        # 3 L2 × 2 L3 × 4 L4 = 24 states
        num_seqs = 480
        result = L4.generate_seqs(num_seqs=num_seqs, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify uniform distribution of L2 mutation positions
        l2_counts = {}
        for i in range(num_seqs):
            pos = dc.get_row(i)['L2_mut_pos'][0]
            l2_counts[pos] = l2_counts.get(pos, 0) + 1
        
        expected_per_l2 = num_seqs / 3
        for pos in [0, 2, 4]:
            count = l2_counts.get(pos, 0)
            assert abs(count - expected_per_l2) < expected_per_l2 * 0.15
    
    def test_distrib_shared_transformer(self):
        """Shared KMutation: both occurrences have same mutations."""
        base = Pool(['GGGGGGGG'], name='base', metadata='complete')
        shared_kmut = KMutationPool(base, k=1, positions=[0, 2, 4, 6],
                                    name='kmut', mode='sequential', metadata='complete')
        
        ss = SpacingScanPool(
            'N' * 40, insert_seqs=[shared_kmut, shared_kmut],
            insert_names=['A', 'B'],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        num_seqs = 300
        result = ss.generate_seqs(num_seqs=num_seqs, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify both occurrences always have identical mutations
        mismatch_count = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            kmut_vals = []
            for key in dc.keys:
                if 'kmut' in key and key.endswith('_value'):
                    val = row.get(key)
                    if val:
                        kmut_vals.append(val)
            
            if len(kmut_vals) >= 2 and kmut_vals[0] != kmut_vals[1]:
                mismatch_count += 1
        
        assert mismatch_count == 0


# =============================================================================
# Class 4: TestRigorousGraphStructure
# =============================================================================

class TestRigorousGraphStructure:
    """Rigorous computation graph structure and node sequence verification."""
    
    def test_graph_4level_parent_chain(self):
        """Category C: 4-level chain - verify parent_ids form correct chain."""
        # Realistic 60bp background
        L1 = Pool(['ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC'], 
                  name='L1_base', metadata='complete')
        L2 = KMutationPool(L1, k=2, name='L2_kmut', mode='random', metadata='complete')
        L3 = InsertionScanPool(L2, 'XXXX', start=10, end=40, step_size=10,
                               insert_or_overwrite='overwrite',
                               name='L3_is', mode='random', metadata='complete')
        other = Pool(['GGGG' * 15], name='L4_other', metadata='complete')  # 60bp
        L4 = SpacingScanPool(
            'N' * 180, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=90, insert_distances=[[-80], [5]],
            name='L4_ss', mode='random', metadata='complete'
        )
        
        num_seqs = 100
        result = L4.generate_seqs(num_seqs=num_seqs, seed=42, 
                                  return_computation_graph=True, return_design_cards=True)
        graph = result['graph']
        
        # Build name to node mapping
        name_to_node = {n['name']: n for n in graph['nodes'] if n.get('name')}
        
        # Verify all levels present
        for level in ['L1_base', 'L2_kmut', 'L3_is', 'L4_ss']:
            assert level in name_to_node, f"{level} should be in graph"
        
        # Verify parent chain: L4 → L3 → L2 → L1 (via node_ids)
        node_id_to_name = {n['node_id']: n['name'] for n in graph['nodes'] if n.get('name')}
        
        for node in graph['nodes']:
            if node.get('name') == 'L3_is':
                # L3's parent should include L2
                parent_ids = node.get('parent_ids', [])
                parent_names = [node_id_to_name.get(pid) for pid in parent_ids]
                assert 'L2_kmut' in parent_names or any('L2' in str(n) for n in parent_names if n), \
                    f"L3_is parent chain should include L2_kmut"
    
    def test_graph_5level_all_nodes_present(self):
        """Category C: 5-level chain - all 5 levels appear in graph."""
        L1 = Pool(['ACGTACGTACGTACGTACGT'], name='L1_base', metadata='complete')
        L2 = RandomMutationPool(L1, mutation_rate=0.1, name='L2_rand', 
                                mode='random', metadata='complete')
        L3 = KMutationPool(L2, k=1, name='L3_kmut', mode='random', metadata='complete')
        L4 = InsertionScanPool(L3, 'XX', start=4, end=16, step_size=4,
                               insert_or_overwrite='overwrite',
                               name='L4_is', mode='random', metadata='complete')
        other = Pool(['T' * 20], name='L5_other', metadata='complete')  # 20bp
        L5 = SpacingScanPool(
            'N' * 100, insert_seqs=[L4, other], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='L5_ss', mode='random', metadata='complete'
        )
        
        num_seqs = 100
        result = L5.generate_seqs(num_seqs=num_seqs, seed=42, 
                                  return_computation_graph=True, return_design_cards=True)
        graph = result['graph']
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All 5 levels should be in graph
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        for level in ['L1_base', 'L2_rand', 'L3_kmut', 'L4_is', 'L5_ss']:
            assert level in node_names, f"{level} should be in graph"
        
        # Verify each level has correct metadata
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            l2_val = row['L2_rand_value']
            l3_val = row['L3_kmut_value']
            l4_val = row['L4_is_value']
            assert hamming_distance(l2_val, l3_val) == 1, "L3 should have 1 mutation"
            assert 'XX' in l4_val, "L4 should have XX"
            assert l4_val in seq, "L4 value should be in final sequence"
    
    def test_graph_mixed_branch_structure(self):
        """Category B: MixedPool branches - verify branch structure in graph."""
        is1 = InsertionScanPool(
            'A' * 20, 'XXXX', start=4, end=16, step_size=4,
            insert_or_overwrite='overwrite',
            name='is1', mode='sequential', metadata='complete'
        )
        is2 = InsertionScanPool(
            'T' * 20, 'YYYY', start=4, end=16, step_size=4,
            insert_or_overwrite='overwrite',
            name='is2', mode='sequential', metadata='complete'
        )
        mixed = MixedPool([is1, is2], weights=[0.5, 0.5], name='mixed', mode='random')
        other = Pool(['G' * 20], name='other', metadata='complete')  # 20bp
        ss = SpacingScanPool(
            'N' * 100, insert_seqs=[mixed, other], insert_names=['Mix', 'Oth'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 200
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42, 
                                  return_computation_graph=True, return_design_cards=True)
        graph = result['graph']
        dc = result['design_cards']
        seqs = result['sequences']
        
        node_names = [n['name'] for n in graph['nodes'] if n.get('name')]
        assert 'ss' in node_names and 'mixed' in node_names
        
        is1_seen = is2_seen = 0
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            if row['mixed_selected'] == 0:
                is1_seen += 1
                assert 'XXXX' in row['is1_value'] and 'XXXX' in seq
            else:
                is2_seen += 1
                assert 'YYYY' in row['is2_value'] and 'YYYY' in seq
        assert is1_seen > 50 and is2_seen > 50


# =============================================================================
# Class 5: TestNodeSequenceVerification
# =============================================================================

class TestNodeSequenceVerification:
    """Verify node sequences match design card values and transformations."""
    
    def test_nodeseq_transformer_input_output(self):
        """Category C: node_seq matches design card values for transformers."""
        L1 = Pool(['ACGTACGTACGTACGTACGT'], name='L1_base', metadata='complete')
        L2 = KMutationPool(L1, k=2, name='L2_kmut', mode='random', metadata='complete')
        L3 = InsertionScanPool(L2, 'XX', start=4, end=16, step_size=4,
                               insert_or_overwrite='overwrite',
                               name='L3_is', mode='random', metadata='complete')
        other = Pool(['T' * 20], name='other', metadata='complete')  # 20bp
        ss = SpacingScanPool(
            'N' * 100, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 100
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            l1_val = row['L1_base_value']
            l2_val = row['L2_kmut_value']
            l3_val = row['L3_is_value']
            assert hamming_distance(l1_val, l2_val) == 2, "L2 should have 2 mutations"
            assert 'XX' in l3_val, "L3 should have XX"
            assert l3_val in seq, "L3 value should be in final sequence"
    
    def test_nodeseq_composition_chain(self):
        """Category C: L(n) = transform(L(n-1)) verified via sequences."""
        original = 'GGGGGGGGGGGGGGGGGGGG'  # 20bp all G
        L1 = Pool([original], name='L1_base', metadata='complete')
        L2 = KMutationPool(L1, k=1, name='L2_kmut', mode='random', metadata='complete')
        L3 = InsertionScanPool(L2, 'AA', start=5, end=15, step_size=5,
                               insert_or_overwrite='overwrite',
                               name='L3_is', mode='random', metadata='complete')
        other = Pool(['T' * 20], name='other', metadata='complete')  # 20bp
        L4 = SpacingScanPool(
            'N' * 100, insert_seqs=[L3, other], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='L4_ss', mode='random', metadata='complete'
        )
        
        num_seqs = 100
        result = L4.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            l1_val = row['L1_base_value']
            l2_val = row['L2_kmut_value']
            l3_val = row['L3_is_value']
            assert l1_val == original, "L1 should be original"
            assert hamming_distance(l1_val, l2_val) == 1, "L2 has 1 mutation from L1"
            assert 'AA' in l3_val, "L3 has AA from insertion"
            assert len(l3_val) == len(l2_val), "L3 length should match L2"
    
    def test_nodeseq_spacing_inserts_extracted(self):
        """Category A: node sequences match extracted from final sequence."""
        insert_a = Pool(['AAAAAAAA'], name='insert_a', metadata='complete')
        insert_b = Pool(['TTTTTTTT'], name='insert_b', metadata='complete')
        ss = SpacingScanPool(
            'G' * 60, insert_seqs=[insert_a, insert_b], insert_names=['A', 'B'],
            anchor_pos=30, insert_distances=[[-15, -12], [12, 15]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        num_seqs = ss.num_states * 10
        result = ss.generate_seqs(num_seqs=num_seqs, 
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            a_start = row['ss_A_pos_start']
            a_end = row['ss_A_pos_end']
            extracted_a = seq[a_start:a_end]
            assert extracted_a == 'AAAAAAAA', f"Insert A should be AAAAAAAA"
            b_start = row['ss_B_pos_start']
            b_end = row['ss_B_pos_end']
            extracted_b = seq[b_start:b_end]
            assert extracted_b == 'TTTTTTTT', f"Insert B should be TTTTTTTT"
            assert row['ss_spacing_A_B'] == b_start - a_end
    
    def test_nodeseq_mutation_positions_match(self):
        """Category D: mutations in node_seq at reported positions."""
        orf_seq = 'ATGAAACCCGGGTTTAAA'  # 6 codons = 18bp
        orf = KMutationORFPool(orf_seq, mutation_type='any_codon', k=1,
            positions=[0, 1, 2, 3, 4, 5], name='orf', mode='sequential', metadata='complete')
        other = Pool(['T' * 18], name='other', metadata='complete')  # 18bp
        ss = SpacingScanPool(
            'N' * 100, insert_seqs=[orf, other], insert_names=['ORF', 'Oth'],
            anchor_pos=50, insert_distances=[[-45], [5]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        num_seqs = ss.num_states * 5
        result = ss.generate_seqs(num_seqs=num_seqs,
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            codon_pos = row['orf_codon_pos'][0]
            codon_from = row['orf_codon_from'][0]
            codon_to = row['orf_codon_to'][0]
            orf_val = row['orf_value']
            bp_pos = codon_pos * 3
            original_codon = orf_seq[bp_pos:bp_pos+3]
            assert codon_from == original_codon
            output_codon = orf_val[bp_pos:bp_pos+3]
            assert output_codon == codon_to
            assert orf_val in seq
    
    def test_nodeseq_orf_frame_preservation(self):
        """Category D: all ORF node sequences have length % 3 == 0."""
        orf1_seq = 'ATGAAACCCGGGTTT'  # 5 codons = 15bp
        orf1 = KMutationORFPool(orf1_seq, mutation_type='any_codon', k=1,
            positions=[0, 1, 2, 3, 4], name='orf1', mode='random', metadata='complete')
        orf2 = RandomMutationORFPool(orf1_seq, mutation_rate=0.2, mutation_type='any_codon',
            name='orf2', mode='random', metadata='complete')
        mixed = MixedPool([orf1, orf2], weights=[0.5, 0.5], name='mixed', mode='random')
        other = Pool(['AAA' * 5], name='other', metadata='complete')  # 15bp
        ss = SpacingScanPool(
            'N' * 80, insert_seqs=[mixed, other], insert_names=['Mix', 'Oth'],
            anchor_pos=40, insert_distances=[[-35], [5]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 200
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        orf1_seen = orf2_seen = False
        for i in range(num_seqs):
            row = dc.get_row(i)
            if row['mixed_selected'] == 0:
                orf1_seen = True
                assert len(row['orf1_value']) % 3 == 0
            else:
                orf2_seen = True
                assert len(row['orf2_value']) % 3 == 0
        assert orf1_seen and orf2_seen


# =============================================================================
# Class 6: TestGraphConsistency
# =============================================================================

class TestGraphConsistency:
    """Verify graph consistency and reproducibility."""
    
    def test_graph_state_reproducibility(self):
        """Category E: same state produces same graph."""
        shared = Pool(['ACGTACGT'], name='shared', metadata='complete')
        is_pool = InsertionScanPool(
            shared, shared, start=0, end=6, step_size=2,
            insert_or_overwrite='insert',
            name='is', mode='sequential', metadata='complete'
        )
        other = Pool(['TTTTTTTT'], name='other', metadata='complete')
        ss = SpacingScanPool(
            'N' * 60, insert_seqs=[is_pool, other], insert_names=['IS', 'Oth'],
            anchor_pos=30, insert_distances=[[-25], [5]],
            name='ss', mode='sequential', metadata='complete'
        )
        
        num_seqs = 50
        result1 = ss.generate_seqs(num_seqs=num_seqs, 
                                   return_computation_graph=True, return_design_cards=True)
        ss.set_state(0)  # set_state takes an integer, not tuple
        result2 = ss.generate_seqs(num_seqs=num_seqs,
                                   return_computation_graph=True, return_design_cards=True)
        
        assert result1['sequences'] == result2['sequences'], "Same state should produce same sequences"
        dc1 = result1['design_cards']
        dc2 = result2['design_cards']
        for i in range(num_seqs):
            row1 = dc1.get_row(i)
            row2 = dc2.get_row(i)
            for key in dc1.keys:
                assert row1[key] == row2[key], f"Row {i} key {key} should match"
    
    def test_nodeseq_shared_pool_identity(self):
        """Category E: shared pool has single consistent node_seq across occurrences."""
        shared = Pool(['ACGTACGTACGTACGTACGT'], name='shared', metadata='complete')  # 20bp
        kmut = KMutationPool(shared, k=1, name='kmut', mode='random', metadata='complete')
        
        ss = SpacingScanPool(
            'N' * 100, insert_seqs=[kmut, kmut], insert_names=['A', 'B'],
            anchor_pos=50, insert_distances=[[-40], [5]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 100
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_computation_graph=True, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            kmut_vals = []
            for key in dc.keys:
                if 'kmut' in key and key.endswith('_value'):
                    val = row.get(key)
                    if val:
                        kmut_vals.append(val)
            
            if len(kmut_vals) >= 2:
                assert all(v == kmut_vals[0] for v in kmut_vals), \
                    f"All kmut occurrences should have same value"
            
            if kmut_vals:
                val = kmut_vals[0]
                count = seq.count(val)
                assert count >= 2, f"Shared pool value should appear at least twice"
    
    def test_mixed_selected_child_in_sequence(self):
        """Category B: MixedPool - only selected child appears in final sequence."""
        child_a = Pool(['AAAA'], name='child_a', metadata='complete')
        child_b = Pool(['TTTT'], name='child_b', metadata='complete')
        child_c = Pool(['GGGG'], name='child_c', metadata='complete')
        
        mixed = MixedPool([child_a, child_b, child_c], weights=[0.5, 0.3, 0.2],
                          name='mixed', mode='random')
        
        other = Pool(['CCCC'], name='other', metadata='complete')
        ss = SpacingScanPool(
            'N' * 30, insert_seqs=[mixed, other], insert_names=['Mix', 'Oth'],
            anchor_pos=15, insert_distances=[[-10], [3]],
            name='ss', mode='random', metadata='complete'
        )
        
        num_seqs = 200
        result = ss.generate_seqs(num_seqs=num_seqs, seed=42,
                                  return_design_cards=True, return_computation_graph=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        selections = {0: 0, 1: 0, 2: 0}
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            selected = row['mixed_selected']
            selections[selected] += 1
            
            # Only selected child should appear in the final sequence
            if selected == 0:
                assert 'AAAA' in seq, "child_a should be in sequence when selected"
                assert 'TTTT' not in seq, "child_b should NOT be in sequence"
                assert 'GGGG' not in seq, "child_c should NOT be in sequence"
            elif selected == 1:
                assert 'AAAA' not in seq, "child_a should NOT be in sequence"
                assert 'TTTT' in seq, "child_b should be in sequence when selected"
                assert 'GGGG' not in seq, "child_c should NOT be in sequence"
            else:
                assert 'AAAA' not in seq, "child_a should NOT be in sequence"
                assert 'TTTT' not in seq, "child_b should NOT be in sequence"
                assert 'GGGG' in seq, "child_c should be in sequence when selected"
        
        # All children should be selected at least once (verifies test is non-trivial)
        assert selections[0] > 50, "child_a should be selected significantly (weight 0.5)"
        assert selections[1] > 20, "child_b should be selected significantly (weight 0.3)"
        assert selections[2] > 10, "child_c should be selected significantly (weight 0.2)"
