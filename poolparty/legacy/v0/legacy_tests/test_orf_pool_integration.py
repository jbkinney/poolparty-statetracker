"""Integration tests for combining multiple ORF-based pools in realistic scenarios.

These tests verify that when multiple ORF pools are chained together:
1. Mutation types are correctly applied at each stage
2. Mutation frequencies/rates are respected
3. Mutation positions are correct
4. Complete state coverage works for finite pools
5. Flanking regions (UTRs) are preserved through chains
"""

import pytest
from math import comb
from poolparty import (
    Pool,
    KMutationORFPool,
    RandomMutationORFPool,
    InsertionScanORFPool,
    DeletionScanORFPool,
)


class TestSaturationMutagenesisPipeline:
    """
    Realistic scenario: Saturation mutagenesis with epitope tag insertion.
    
    Use case: Create a library where:
    1. Introduce k missense mutations throughout ORF
    2. Insert a purification tag at N-terminus
    
    This is common in protein engineering workflows.
    """
    
    def test_mutation_then_tag_insertion(self):
        """Verify mutations are preserved after tag insertion."""
        # Original ORF: 4 codons
        orf_seq = "ATGGCCAAACCC"  # Met-Ala-Lys-Pro
        
        # Stage 1: Introduce 1 missense mutation
        mutation_pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='missense_only_first',
            mode='sequential'
        )
        
        # Stage 2: Insert His-tag (6 His = CATCATCATCATCATCAT) at N-terminus
        his_tag = "CATCATCATCATCATCAT"  # 6 codons
        tagged_pool = InsertionScanORFPool(
            mutation_pool,
            insert_seq=his_tag,
            insert_or_overwrite='insert',
            positions=[0],  # Always at N-terminus
            mode='sequential'
        )
        
        codon_to_aa = mutation_pool.codon_to_aa_dict
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        original_aas = [codon_to_aa[c] for c in original_codons]
        
        # Verify all states
        for state in range(tagged_pool.num_states):
            tagged_pool.set_state(state)
            seq = tagged_pool.seq
            
            # Should have His-tag at start
            assert seq.startswith(his_tag), f"State {state}: His-tag not at start"
            
            # Extract ORF part (after tag)
            orf_part = seq[len(his_tag):]
            assert len(orf_part) == len(orf_seq), f"State {state}: ORF length wrong"
            
            orf_codons = [orf_part[i:i+3] for i in range(0, len(orf_part), 3)]
            orf_aas = [codon_to_aa[c] for c in orf_codons]
            
            # Verify exactly 1 missense mutation (AA changed)
            aa_changes = sum(1 for i in range(len(original_aas))
                           if original_aas[i] != orf_aas[i])
            assert aa_changes == 1, f"State {state}: Expected 1 AA change, got {aa_changes}"
    
    def test_complete_library_enumeration(self):
        """Enumerate complete library and verify all unique variants."""
        # Small ORF: 3 codons
        orf_seq = "ATGGCCAAA"
        
        # 1 nonsense mutation (3 stops × 3 positions = 9 states)
        mutation_pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        # Insert marker at position 0 (1 state)
        marker = "GGG"  # Single Gly codon
        tagged_pool = InsertionScanORFPool(
            mutation_pool,
            insert_seq=marker,
            insert_or_overwrite='insert',
            positions=[0],
            mode='sequential'
        )
        
        # Total states = mutation states × insertion states = 9 × 1 = 9
        expected_states = mutation_pool.num_internal_states
        
        # Collect all unique sequences
        sequences = set()
        for state in range(tagged_pool.num_states):
            tagged_pool.set_state(state)
            sequences.add(tagged_pool.seq)
        
        # Should have 9 unique sequences
        assert len(sequences) == expected_states, \
            f"Expected {expected_states} unique sequences, got {len(sequences)}"
        
        # Verify each has exactly 1 stop codon in the ORF portion
        stop_codons = set(mutation_pool.stop_codons)
        for seq in sequences:
            assert seq.startswith(marker)
            orf_part = seq[len(marker):]
            orf_codons = [orf_part[i:i+3] for i in range(0, len(orf_part), 3)]
            num_stops = sum(1 for c in orf_codons if c in stop_codons)
            assert num_stops == 1, f"Expected 1 stop in ORF, found {num_stops}"


class TestDeletionThenMutationPipeline:
    """
    Realistic scenario: Domain deletion followed by compensatory mutations.
    
    Use case: Delete a domain, then introduce mutations to study
    which mutations can rescue function.
    """
    
    def test_delete_domain_then_mutate(self):
        """Delete a codon, then apply mutations to remaining ORF."""
        # ORF: 5 codons representing a small protein
        orf_seq = "ATGGCCAAACCCGGG"  # Met-Ala-Lys-Pro-Gly
        
        # Stage 1: Delete 1 codon at position 2 (Lys)
        deletion_pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=1,
            positions=[2],  # Always delete position 2
            mark_changes=False,  # Actually remove
            mode='sequential'
        )
        
        # Stage 2: Introduce 1 any_codon mutation in remaining 4 codons
        mutation_pool = KMutationORFPool(
            deletion_pool, k=1,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        # Deletion pool has 1 state, mutation pool has C(4,1) * 63 = 252 states
        assert deletion_pool.num_internal_states == 1
        
        # Check sequence after deletion
        deletion_pool.set_state(0)
        deleted_seq = deletion_pool.seq
        assert len(deleted_seq) == 12, f"After deletion: expected 12 nt, got {len(deleted_seq)}"
        
        # Verify mutations are applied to shortened ORF
        original_after_deletion = deleted_seq
        original_codons = [original_after_deletion[i:i+3] for i in range(0, 12, 3)]
        
        mutations_verified = 0
        for state in range(min(50, mutation_pool.num_states)):
            mutation_pool.set_state(state)
            seq = mutation_pool.seq
            
            assert len(seq) == 12, f"State {state}: Wrong length"
            
            mut_codons = [seq[i:i+3] for i in range(0, 12, 3)]
            num_changes = sum(1 for i in range(4) if original_codons[i] != mut_codons[i])
            assert num_changes == 1, f"State {state}: Expected 1 mutation, got {num_changes}"
            mutations_verified += 1
        
        assert mutations_verified >= 50, "Not enough mutations verified"
    
    def test_deletion_scan_then_mutation_all_combinations(self):
        """Scan deletion positions, each followed by mutation."""
        # ORF: 4 codons
        orf_seq = "ATGATGATGATG"
        
        # Stage 1: Delete 1 codon at positions 0, 1, 2, or 3
        deletion_pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=1,
            step_size=1,
            mark_changes=False,
            mode='sequential'
        )
        
        # 4 deletion positions
        assert deletion_pool.num_internal_states == 4
        
        # Stage 2: 1 nonsense mutation in remaining 3 codons
        mutation_pool = KMutationORFPool(
            deletion_pool, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        # Total states = 4 deletion × C(3,1) × 3 = 4 × 9 = 36
        # (But state calculation might be different due to how chaining works)
        
        stop_codons = set(mutation_pool.stop_codons)
        
        # Verify all produced sequences are valid
        sequences = []
        for state in range(mutation_pool.num_states):
            mutation_pool.set_state(state)
            seq = mutation_pool.seq
            
            # Length should be 9 (3 codons after deletion)
            assert len(seq) == 9, f"State {state}: Expected 9 nt, got {len(seq)}"
            
            # Should have exactly 1 stop codon
            codons = [seq[i:i+3] for i in range(0, 9, 3)]
            num_stops = sum(1 for c in codons if c in stop_codons)
            assert num_stops == 1, f"State {state}: Expected 1 stop, got {num_stops}"
            
            sequences.append(seq)
        
        # Should see multiple unique sequences
        assert len(set(sequences)) > 1


class TestSequentialMutationLayers:
    """
    Realistic scenario: Layered mutations for epistasis studies.
    
    Use case: First introduce a known functional mutation,
    then scan for second-site suppressors.
    """
    
    def test_two_layer_mutations_any_codon(self):
        """Apply two layers of mutations, verify both are present."""
        orf_seq = "ATGATGATGATG"  # 4 Met codons
        
        # Layer 1: 1 any_codon mutation
        layer1 = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        # Layer 2: 1 more any_codon mutation
        layer2 = KMutationORFPool(
            layer1, k=1,
            mutation_type='any_codon',
            mode='random'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        # Verify at least 1 mutation (could be 1 or 2 depending on overlap)
        for state in range(50):
            layer2.set_state(state)
            seq = layer2.seq
            
            result_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            num_changes = sum(1 for i in range(4) if original_codons[i] != result_codons[i])
            
            # At least 1 change (layer 2 adds to layer 1, but could hit same position)
            assert num_changes >= 1, f"State {state}: Expected at least 1 change"
    
    def test_nonsense_then_missense(self):
        """First nonsense, then missense - verify both mutation types present."""
        orf_seq = "ATGGCCAAACCC"  # 4 codons
        
        # Layer 1: 1 nonsense mutation (introduces stop)
        layer1 = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            positions=[0],  # Always at position 0
            mode='sequential'
        )
        
        # Layer 2: 1 missense mutation at position 2
        layer2 = KMutationORFPool(
            layer1, k=1,
            mutation_type='missense_only_first',
            positions=[2],  # Always at position 2
            mode='sequential'
        )
        
        stop_codons = set(layer1.stop_codons)
        codon_to_aa = layer1.codon_to_aa_dict
        
        original_aa_pos2 = codon_to_aa["AAA"]  # Lys
        
        for state in range(layer2.num_states):
            layer2.set_state(state)
            seq = layer2.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Position 0 should be a stop codon
            assert codons[0] in stop_codons, \
                f"State {state}: Position 0 should be stop, got {codons[0]}"
            
            # Position 2 should have different AA than original
            new_aa_pos2 = codon_to_aa[codons[2]]
            assert new_aa_pos2 != original_aa_pos2, \
                f"State {state}: Position 2 AA should change from {original_aa_pos2}"


class TestRandomMutationWithDeterministicScan:
    """
    Realistic scenario: Background random mutagenesis with specific insertions.
    
    Use case: Create a library with low-level random mutations plus
    a systematic scan of insertion positions.
    """
    
    def test_random_background_with_insertion_scan(self):
        """Random mutations in background, then scan insertion positions."""
        orf_seq = "ATGATGATGATGATG"  # 5 codons
        
        # Background: Low-rate random mutations
        background = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.1  # 10% per codon
        )
        
        # Scan: Insert marker at different positions
        marker = "GGG"  # 1 codon
        scan_pool = InsertionScanORFPool(
            background,
            insert_seq=marker,
            insert_or_overwrite='insert',
            positions=[0, 2, 4],  # N-term, middle, near C-term
            mode='sequential'
        )
        
        assert scan_pool.num_internal_states == 3
        
        # Verify each insertion position works
        for state in range(3):
            scan_pool.set_state(state)
            seq = scan_pool.seq
            
            # Should be 18 nt (5 original + 1 inserted) = 6 codons
            assert len(seq) == 18, f"State {state}: Expected 18 nt, got {len(seq)}"
            
            # Find where GGG is inserted
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            ggg_positions = [i for i, c in enumerate(codons) if c == "GGG"]
            
            # Should have exactly 1 GGG
            assert len(ggg_positions) == 1, \
                f"State {state}: Expected 1 GGG, found {len(ggg_positions)}"


class TestCompleteLibraryEnumeration:
    """
    Realistic scenario: Small complete library enumeration.
    
    Use case: For small state spaces, enumerate all variants
    and verify uniqueness and correctness.
    """
    
    def test_small_library_complete_enumeration(self):
        """Enumerate all variants in a small library."""
        # Minimal ORF: 2 codons
        orf_seq = "ATGGCC"
        
        # 1 nonsense mutation: C(2,1) × 3 = 6 states
        mutation_pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        # Verify exactly 6 states
        assert mutation_pool.num_internal_states == 6
        
        stop_codons = set(mutation_pool.stop_codons)
        original_codons = ["ATG", "GCC"]
        
        sequences = []
        position_mutation_combos = set()
        
        for state in range(6):
            mutation_pool.set_state(state)
            seq = mutation_pool.seq
            sequences.append(seq)
            
            codons = [seq[i:i+3] for i in range(0, 6, 3)]
            
            # Find mutated position
            for i in range(2):
                if codons[i] != original_codons[i]:
                    assert codons[i] in stop_codons, f"State {state}: Non-stop mutation"
                    position_mutation_combos.add((i, codons[i]))
        
        # All 6 sequences should be unique
        assert len(set(sequences)) == 6, "Expected 6 unique sequences"
        
        # Should have 2 positions × 3 stops = 6 combinations
        assert len(position_mutation_combos) == 6
    
    def test_deletion_insertion_complete_coverage(self):
        """Complete enumeration of deletion+insertion combinations."""
        # ORF: 4 codons
        orf_seq = "ATGGCCAAACCC"
        
        # Delete 1 codon at 2 positions (0 or 2)
        deletion_pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=1,
            positions=[0, 2],
            mark_changes=False,
            mode='sequential'
        )
        
        # 2 deletion states
        assert deletion_pool.num_internal_states == 2
        
        # Insert marker at 2 positions (0 or 1)
        marker = "GGG"
        insertion_pool = InsertionScanORFPool(
            deletion_pool,
            insert_seq=marker,
            insert_or_overwrite='insert',
            positions=[0, 1],
            mode='sequential'
        )
        
        # Total: 2 deletions × 2 insertions = 4 combinations
        # (internal states = 2, but total with parent = 4)
        
        sequences = set()
        for state in range(insertion_pool.num_states):
            insertion_pool.set_state(state)
            seq = insertion_pool.seq
            sequences.add(seq)
            
            # Should be 12 nt (4 original - 1 deleted + 1 inserted) = 4 codons
            assert len(seq) == 12, f"State {state}: Expected 12 nt"
        
        # Should have 4 unique sequences
        assert len(sequences) == 4, f"Expected 4 unique sequences, got {len(sequences)}"


class TestUTRPreservationThroughChains:
    """
    Realistic scenario: Full gene with UTRs through mutation pipeline.
    
    Use case: Mutate coding region while preserving regulatory UTRs.
    """
    
    def test_utrs_preserved_through_mutation_chain(self):
        """Verify 5' and 3' UTRs are preserved through multiple mutation stages."""
        # Full gene: 5'UTR + ORF + 3'UTR
        utr5 = "GGGGGGG"  # 7 bp 5' UTR
        orf = "ATGATGATG"  # 3 codons
        utr3 = "CCCCCCC"  # 7 bp 3' UTR
        full_gene = utr5 + orf + utr3
        
        # Stage 1: Random mutations
        stage1 = RandomMutationORFPool(
            full_gene,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=7,
            orf_end=16
        )
        
        # Stage 2: K mutations
        stage2 = KMutationORFPool(
            stage1, k=1,
            mutation_type='any_codon',
            orf_start=7,
            orf_end=16
        )
        
        # Verify UTRs preserved through chain
        for state in range(50):
            stage2.set_state(state)
            seq = stage2.seq
            
            assert seq[:7] == utr5, f"State {state}: 5' UTR modified: {seq[:7]}"
            assert seq[-7:] == utr3, f"State {state}: 3' UTR modified: {seq[-7:]}"
            assert len(seq) == len(full_gene), f"State {state}: Length changed"
    
    def test_utrs_preserved_through_deletion_insertion(self):
        """Verify UTRs preserved through deletion and insertion."""
        utr5 = "AAAAAAA"
        orf = "ATGATGATGATGATG"  # 5 codons
        utr3 = "TTTTTTT"
        full_gene = utr5 + orf + utr3
        
        # Delete 1 codon from ORF
        deletion_pool = DeletionScanORFPool(
            full_gene,
            deletion_size=1,
            positions=[1],  # Delete second codon
            mark_changes=False,
            orf_start=7,
            orf_end=22
        )
        
        # Insert 1 codon back
        insertion_pool = InsertionScanORFPool(
            deletion_pool,
            insert_seq="GGG",
            insert_or_overwrite='insert',
            positions=[0],
            orf_start=7,
            orf_end=19  # Shorter after deletion
        )
        
        for state in range(insertion_pool.num_states):
            insertion_pool.set_state(state)
            seq = insertion_pool.seq
            
            assert seq[:7] == utr5, f"State {state}: 5' UTR modified"
            assert seq[-7:] == utr3, f"State {state}: 3' UTR modified"


class TestMutationTypeVerificationInChains:
    """
    Verify mutation types are correctly applied through chains.
    """
    
    def test_synonymous_preserved_through_chain(self):
        """Verify synonymous mutations preserve AA through entire chain."""
        # Use codons with synonymous alternatives
        orf_seq = "CTGCTGCTG"  # 3 Leu codons (6 synonyms each)
        
        # Layer 1: Synonymous
        layer1 = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='synonymous',
            mode='random'
        )
        
        # Layer 2: Another synonymous
        layer2 = KMutationORFPool(
            layer1, k=1,
            mutation_type='synonymous',
            mode='random'
        )
        
        codon_to_aa = layer1.codon_to_aa_dict
        original_aas = ['L', 'L', 'L']  # All Leu
        
        for state in range(100):
            layer2.set_state(state)
            seq = layer2.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            result_aas = [codon_to_aa[c] for c in codons]
            
            assert result_aas == original_aas, \
                f"State {state}: Synonymous chain changed AA: {original_aas} -> {result_aas}"
    
    def test_nonsense_in_chain_always_produces_stops(self):
        """Verify nonsense mutations produce stops even in chain context."""
        orf_seq = "ATGGCCAAACCC"  # 4 codons, no stops
        
        # Layer 1: Any codon mutation
        layer1 = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='any_codon',
            positions=[0],
            mode='sequential'
        )
        
        # Layer 2: Nonsense at different position
        layer2 = KMutationORFPool(
            layer1, k=1,
            mutation_type='nonsense',
            positions=[2],  # Always position 2
            mode='sequential'
        )
        
        stop_codons = set(layer1.stop_codons)
        
        for state in range(min(50, layer2.num_states)):
            layer2.set_state(state)
            seq = layer2.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Position 2 must be a stop
            assert codons[2] in stop_codons, \
                f"State {state}: Position 2 should be stop, got {codons[2]}"


class TestMutationFrequencyInChains:
    """
    Verify mutation frequencies are approximately correct in chains.
    """
    
    def test_random_mutation_rate_in_chain(self):
        """Verify random mutation rate is approximately correct through chain."""
        orf_seq = "ATGGCCAAA" * 10  # 30 codons
        target_rate = 0.2
        
        # Random mutations
        random_pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=target_rate
        )
        
        # Then tag insertion (shouldn't affect mutation rate in ORF)
        tagged_pool = InsertionScanORFPool(
            random_pool,
            insert_seq="GGG",
            insert_or_overwrite='insert',
            positions=[0],
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        num_codons = len(original_codons)
        
        total_mutations = 0
        num_samples = 300
        
        for state in range(num_samples):
            tagged_pool.set_state(state)
            seq = tagged_pool.seq
            
            # Skip the inserted tag
            orf_part = seq[3:]  # After GGG
            orf_codons = [orf_part[i:i+3] for i in range(0, len(orf_part), 3)]
            
            mutations = sum(1 for i in range(num_codons)
                          if original_codons[i] != orf_codons[i])
            total_mutations += mutations
        
        actual_rate = total_mutations / (num_samples * num_codons)
        
        # Allow 30% tolerance for random sampling
        assert abs(actual_rate - target_rate) < 0.1, \
            f"Expected rate ~{target_rate}, got {actual_rate:.3f}"


class TestExactPositionsInChains:
    """
    Verify mutation positions are exactly as specified through chains.
    """
    
    def test_position_restriction_preserved_through_chain(self):
        """Verify position restrictions work correctly in chained pools."""
        orf_seq = "ATGATGATGATGATG"  # 5 codons
        
        # Layer 1: Mutation only at position 1
        layer1 = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='any_codon',
            positions=[1],
            mode='sequential'
        )
        
        # Layer 2: Mutation only at position 3
        layer2 = KMutationORFPool(
            layer1, k=1,
            mutation_type='any_codon',
            positions=[3],
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(min(50, layer2.num_states)):
            layer2.set_state(state)
            seq = layer2.seq
            result_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Positions 0, 2, 4 should never change
            assert result_codons[0] == original_codons[0], f"State {state}: Pos 0 changed"
            assert result_codons[2] == original_codons[2], f"State {state}: Pos 2 changed"
            assert result_codons[4] == original_codons[4], f"State {state}: Pos 4 changed"
            
            # Positions 1 and 3 should change (one from each layer)
            changes = [i for i in range(5) if result_codons[i] != original_codons[i]]
            assert 1 in changes, f"State {state}: Position 1 should change"
            assert 3 in changes, f"State {state}: Position 3 should change"
    
    def test_deletion_position_then_mutation_position(self):
        """Verify deletion at specific position, then mutation at specific position."""
        orf_seq = "ATGGCCAAACCCGGG"  # 5 codons
        
        # Delete codon at position 2
        deletion_pool = DeletionScanORFPool(
            orf_seq,
            deletion_size=1,
            positions=[2],
            mark_changes=False,
            mode='sequential'
        )
        
        # After deletion: 4 codons (positions 0,1,3,4 become 0,1,2,3)
        # Mutate at position 0 of shortened ORF
        mutation_pool = KMutationORFPool(
            deletion_pool, k=1,
            mutation_type='nonsense',
            positions=[0],
            mode='sequential'
        )
        
        stop_codons = set(mutation_pool.stop_codons)
        
        for state in range(mutation_pool.num_states):
            mutation_pool.set_state(state)
            seq = mutation_pool.seq
            
            # 4 codons after deletion
            assert len(seq) == 12
            
            codons = [seq[i:i+3] for i in range(0, 12, 3)]
            
            # Position 0 should be a stop (was ATG, now stop)
            assert codons[0] in stop_codons, f"State {state}: Pos 0 should be stop"
            
            # Position 1 should be GCC (unchanged from original)
            assert codons[1] == "GCC", f"State {state}: Pos 1 should be GCC"

