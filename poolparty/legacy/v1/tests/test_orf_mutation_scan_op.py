"""Tests for orf_mutation_scan operation."""

import pytest
from math import comb
from poolparty import Pool, from_seqs_op
from poolparty.orf_operation import ORFOp
from poolparty.orf_operations.orf_mutation_scan_op import orf_mutation_scan_op, ORFMutationScanOp


class TestORFOperationBase:
    """Tests for ORFOperation base class."""
    
    def test_standard_genetic_code_has_all_codons(self):
        """Test that standard genetic code has all 64 codons."""
        total_codons = sum(len(codons) for codons in ORFOp.STANDARD_GENETIC_CODE.values())
        assert total_codons == 64
    
    def test_standard_genetic_code_has_21_amino_acids(self):
        """Test that standard genetic code has 21 amino acids (including stop)."""
        assert len(ORFOp.STANDARD_GENETIC_CODE) == 21
    
    def test_standard_genetic_code_has_stop_codons(self):
        """Test that stop codons are present."""
        assert '*' in ORFOp.STANDARD_GENETIC_CODE
        assert set(ORFOp.STANDARD_GENETIC_CODE['*']) == {'TGA', 'TAA', 'TAG'}


class TestORFMutationScanCreation:
    """Tests for orf_mutation_scan factory function."""
    
    def test_basic_creation_from_string(self):
        """Test basic creation from string ORF."""
        pool = orf_mutation_scan_op('ATGAAATTT', k=1)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 9
    
    def test_basic_creation_from_pool(self):
        """Test creation from another Pool."""
        parent = from_seqs_op(['ATGAAATTT'])
        pool = orf_mutation_scan_op(parent, k=1)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 9
    
    def test_invalid_dna_raises(self):
        """Test that non-DNA sequence raises error."""
        with pytest.raises(ValueError, match="ACGT"):
            orf_mutation_scan_op('ATGXYZ', k=1)
    
    def test_non_divisible_by_3_raises(self):
        """Test that sequence not divisible by 3 raises error."""
        with pytest.raises(ValueError, match="divisible by 3"):
            orf_mutation_scan_op('ATGA', k=1)
    
    def test_k_zero_raises(self):
        """Test that k=0 raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            orf_mutation_scan_op('ATGAAA', k=0)
    
    def test_k_too_large_raises(self):
        """Test that k > num_codons raises error."""
        with pytest.raises(ValueError, match="must be <="):
            orf_mutation_scan_op('ATGAAA', k=5)  # Only 2 codons
    
    def test_invalid_mutation_type_raises(self):
        """Test that invalid mutation_type raises error."""
        with pytest.raises(ValueError, match="mutation_type must be one of"):
            orf_mutation_scan_op('ATGAAA', mutation_type='invalid_type')


class TestORFMutationScanMutationTypes:
    """Tests for different mutation types."""
    
    def test_any_codon_is_uniform(self):
        """Test that any_codon has uniform alternatives (63)."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='any_codon')
        op = pool.operation
        assert op.is_uniform
        assert op.uniform_num_possible_mutations == 63
    
    def test_missense_only_first_is_uniform(self):
        """Test that missense_only_first has uniform alternatives (19)."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='missense_only_first')
        op = pool.operation
        assert op.is_uniform
        assert op.uniform_num_possible_mutations == 19
    
    def test_nonsynonymous_first_is_uniform(self):
        """Test that nonsynonymous_first has uniform alternatives (20)."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='nonsynonymous_first')
        op = pool.operation
        assert op.is_uniform
        assert op.uniform_num_possible_mutations == 20
    
    def test_nonsense_is_uniform(self):
        """Test that nonsense has uniform alternatives (3)."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='nonsense')
        op = pool.operation
        assert op.is_uniform
        assert op.uniform_num_possible_mutations == 3
    
    def test_synonymous_is_nonuniform(self):
        """Test that synonymous is non-uniform (variable by codon)."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='synonymous')
        op = pool.operation
        assert not op.is_uniform
        assert pool.operation.num_states == -1
    
    def test_missense_only_random_is_nonuniform(self):
        """Test that missense_only_random is non-uniform."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='missense_only_random')
        op = pool.operation
        assert not op.is_uniform
    
    def test_nonsynonymous_random_is_nonuniform(self):
        """Test that nonsynonymous_random is non-uniform."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='nonsynonymous_random')
        op = pool.operation
        assert not op.is_uniform


class TestORFMutationScanNumStates:
    """Tests for state space calculation."""
    
    def test_num_states_k1_missense(self):
        """Test num_states for k=1 with missense_only_first."""
        # 3 codons × 19 alternatives = 57
        pool = orf_mutation_scan_op('ATGAAATTT', k=1, mutation_type='missense_only_first')
        assert pool.operation.num_states == 3 * 19
    
    def test_num_states_k2_missense(self):
        """Test num_states for k=2 with missense_only_first."""
        # C(3, 2) × 19^2 = 3 × 361 = 1083
        pool = orf_mutation_scan_op('ATGAAATTT', k=2, mutation_type='missense_only_first')
        assert pool.operation.num_states == comb(3, 2) * (19 ** 2)
    
    def test_num_states_formula(self):
        """Test that num_states follows C(L, k) × alternatives^k."""
        seq = 'ATGAAATTTGGG'  # 4 codons
        
        for k in [1, 2, 3]:
            pool = orf_mutation_scan_op(seq, k=k, mutation_type='missense_only_first')
            expected = comb(4, k) * (19 ** k)
            assert pool.operation.num_states == expected, f"Failed for k={k}"
    
    def test_num_states_any_codon(self):
        """Test num_states with any_codon type."""
        # 3 codons × 63 alternatives = 189
        pool = orf_mutation_scan_op('ATGAAATTT', k=1, mutation_type='any_codon')
        assert pool.operation.num_states == 3 * 63


class TestORFMutationScanSequences:
    """Tests for sequence generation."""
    
    def test_mutation_preserves_length(self):
        """Test that mutations don't change sequence length."""
        original = 'ATGAAATTT'
        pool = orf_mutation_scan_op(original, k=1)
        
        for _ in range(10):
            seq = pool.seq
            assert len(seq) == len(original)
    
    def test_mutation_changes_codon(self):
        """Test that mutation changes at least one codon."""
        original = 'AAAAAA'  # Two AAA codons (Lysine)
        pool = orf_mutation_scan_op(original, k=1, mutation_type='missense_only_first')
        
        # Generate several sequences
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        # All should differ from original
        for seq in seqs:
            assert seq != original
    
    def test_mutation_changes_exactly_k_codons(self):
        """Test that exactly k codons are mutated."""
        original = 'AAAAAAAAA'  # 3 codons
        pool = orf_mutation_scan_op(original, k=2, mutation_type='any_codon', mode='sequential')
        
        for seq in pool.generate_library(num_complete_iterations=1):
            # Split into codons
            orig_codons = [original[i:i+3] for i in range(0, 9, 3)]
            seq_codons = [seq[i:i+3] for i in range(0, 9, 3)]
            
            # Count changed codons
            changed = sum(1 for o, s in zip(orig_codons, seq_codons) if o != s)
            assert changed == 2, f"Expected 2 codon changes, got {changed}"
    
    def test_sequential_mode_enumerates_all(self):
        """Test that sequential mode covers all states."""
        pool = orf_mutation_scan_op('AAAAAA', k=1, mutation_type='nonsense', mode='sequential')
        # 2 codons × 3 stop codons = 6 states
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 6
        assert len(set(seqs)) == 6  # All unique
    
    def test_missense_produces_different_amino_acid(self):
        """Test that missense mutations produce different amino acids."""
        pool = orf_mutation_scan_op('ATGAAATTT', k=1, mutation_type='missense_only_first')
        op = pool.operation
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        original_codons = ['ATG', 'AAA', 'TTT']
        original_aas = [op.codon_to_aa_dict[c] for c in original_codons]
        
        for seq in seqs:
            seq_codons = [seq[i:i+3] for i in range(0, 9, 3)]
            seq_aas = [op.codon_to_aa_dict[c] for c in seq_codons]
            
            # At least one position should have different AA
            different = [i for i in range(3) if seq_aas[i] != original_aas[i]]
            assert len(different) >= 1
            
            # No stop codons should be introduced
            assert '*' not in seq_aas
    
    def test_nonsense_produces_stop_codon(self):
        """Test that nonsense mutations produce stop codons."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='nonsense')
        op = pool.operation
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            seq_codons = [seq[i:i+3] for i in range(0, 6, 3)]
            seq_aas = [op.codon_to_aa_dict[c] for c in seq_codons]
            
            # At least one stop codon should be present
            assert '*' in seq_aas


class TestORFMutationScanFlankingRegions:
    """Tests for flanking region support."""
    
    def test_flanking_regions_preserved(self):
        """Test that flanking regions are preserved."""
        # 5' UTR + ORF + 3' UTR
        full_seq = 'GGGGATGAAATTTCCCC'  # GGGG + ATGAAATTT + CCCC
        pool = orf_mutation_scan_op(full_seq, k=1, orf_start=4, orf_end=13)
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert seq.startswith('GGGG'), f"5' UTR not preserved: {seq}"
            assert seq.endswith('CCCC'), f"3' UTR not preserved: {seq}"
            assert len(seq) == len(full_seq)
    
    def test_orf_boundaries_work(self):
        """Test that orf_start and orf_end define the mutatable region."""
        full_seq = 'AAATTTGGGCCC'  # AAA TTT GGG CCC = 4 codons
        # Only mutate middle two codons (TTT GGG)
        pool = orf_mutation_scan_op(full_seq, k=1, orf_start=3, orf_end=9)
        
        seqs = pool.generate_library(num_seqs=20, seed=42)
        
        for seq in seqs:
            # First and last codons should be unchanged
            assert seq[:3] == 'AAA', f"First codon changed: {seq}"
            assert seq[9:] == 'CCC', f"Last codon changed: {seq}"


class TestORFMutationScanPositions:
    """Tests for position restriction."""
    
    def test_positions_restrict_mutations(self):
        """Test that positions parameter restricts mutation sites."""
        # Only allow mutations at position 0 (first codon)
        pool = orf_mutation_scan_op('ATGAAATTT', k=1, positions=[0], mutation_type='any_codon')
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            # Only first codon should change
            assert seq[3:6] == 'AAA', f"Second codon changed: {seq}"
            assert seq[6:9] == 'TTT', f"Third codon changed: {seq}"
    
    def test_positions_with_k2(self):
        """Test positions with k=2 mutations."""
        # Allow mutations at positions 0 and 2 only
        pool = orf_mutation_scan_op('ATGAAATTT', k=2, positions=[0, 2], mutation_type='any_codon')
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            # Middle codon should not change
            assert seq[3:6] == 'AAA', f"Middle codon changed: {seq}"
    
    def test_invalid_position_raises(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            orf_mutation_scan_op('ATGAAA', k=1, positions=[5])  # Only 2 codons (0, 1)
    
    def test_duplicate_positions_raises(self):
        """Test that duplicate positions raises error."""
        with pytest.raises(ValueError, match="duplicates"):
            orf_mutation_scan_op('ATGAAATTT', k=1, positions=[0, 0, 1])


class TestORFMutationScanMarkChanges:
    """Tests for mark_changes option."""
    
    def test_mark_changes_swapcase(self):
        """Test that mark_changes applies swapcase to mutated codons."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mark_changes=True)
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            # Should have some lowercase letters (mutated codons)
            assert any(c.islower() for c in seq), f"No lowercase in {seq}"
    
    def test_no_mark_changes_uppercase(self):
        """Test that without mark_changes, output is uppercase."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mark_changes=False)
        
        seqs = pool.generate_library(num_seqs=10, seed=42)
        
        for seq in seqs:
            assert seq.isupper(), f"Found lowercase in {seq}"


class TestORFMutationScanSequentialMode:
    """Tests for sequential mode."""
    
    def test_sequential_with_uniform_type(self):
        """Test that sequential mode works with uniform mutation types."""
        pool = orf_mutation_scan_op('AAAAAA', k=1, mutation_type='nonsense', mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        # Should enumerate all states
        assert len(seqs) == pool.operation.num_states
    
    def test_sequential_with_nonuniform_raises(self):
        """Test that sequential mode with non-uniform type raises error."""
        with pytest.raises(ValueError, match="sequential"):
            orf_mutation_scan_op('ATGAAA', k=1, mutation_type='synonymous', mode='sequential')
    
    def test_sequential_deterministic(self):
        """Test that sequential mode is deterministic."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='missense_only_first', mode='sequential')
        
        seqs1 = pool.generate_library(num_complete_iterations=1)
        seqs2 = pool.generate_library(num_complete_iterations=1)
        
        assert seqs1 == seqs2


class TestORFMutationScanNonsenseValidation:
    """Tests for nonsense mutation validation."""
    
    def test_nonsense_with_stop_codon_raises(self):
        """Test that nonsense mutation with stop codon in input raises error."""
        with pytest.raises(ValueError, match="stop codon"):
            orf_mutation_scan_op('ATGTAA', k=1, mutation_type='nonsense')  # TAA is stop
    
    def test_nonsense_without_stop_works(self):
        """Test that nonsense mutation without stop codon works."""
        pool = orf_mutation_scan_op('ATGAAA', k=1, mutation_type='nonsense')
        assert pool.operation.num_states > 0


class TestORFMutationScanDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_seed_same_sequences(self):
        """Test that same seed produces same sequences."""
        pool = orf_mutation_scan_op('ATGAAATTTGGG', k=2)
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        pool = orf_mutation_scan_op('ATGAAATTTGGG', k=2)
        
        seqs1 = pool.generate_library(num_seqs=10, seed=42)
        seqs2 = pool.generate_library(num_seqs=10, seed=123)
        
        assert seqs1 != seqs2


class TestORFMutationScanAncestors:
    """Tests for ancestor tracking."""
    
    def test_has_parent_pool_from_string(self):
        """Test that pool has parent when created from string."""
        pool = orf_mutation_scan_op('ATGAAA', k=1)
        parents = pool.operation.parent_pools
        assert len(parents) == 1
    
    def test_has_parent_pool_from_pool(self):
        """Test that pool has correct parent when created from Pool."""
        parent = from_seqs_op(['ATGAAA', 'TTTGGG'])
        pool = orf_mutation_scan_op(parent, k=1)
        
        parents = pool.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent
    
    def test_ancestors_include_all(self):
        """Test that ancestors include both self and parent."""
        parent = from_seqs_op(['ATGAAA'])
        pool = orf_mutation_scan_op(parent, k=1)
        
        assert pool in pool.ancestors
        assert parent in pool.ancestors
        assert len(pool.ancestors) == 2
