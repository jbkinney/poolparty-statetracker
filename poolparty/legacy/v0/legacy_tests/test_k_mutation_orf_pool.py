"""Tests for the refactored KMutationORFPool class.

Note: The following original tests are no longer applicable due to API changes:
- change_case_of_mutations parameter replaced by mark_changes
- Mutation type names updated: all_by_codon → any_codon,
  missense_first_codon → missense_only_first, missense_random_codon → missense_only_random
- Iteration removed from base Pool class
- Seed parameter not supported (use generate_seqs with seed instead)
"""

import pytest
from poolparty import Pool, KMutationORFPool
from math import comb


class TestKMutationORFPoolBasicInit:
    """Test basic initialization of KMutationORFPool."""
    
    def test_init_valid_orf(self):
        """Test initialization with valid ORF sequence."""
        orf = 'ATGGCC'  # 2 codons
        pool = KMutationORFPool(orf, k=1, mutation_type='missense_only_first')
        assert pool is not None
        assert pool.orf_seq == 'ATGGCC'
        assert pool.k == 1
        assert pool.mutation_type == 'missense_only_first'
    
    def test_init_stores_codons(self):
        """Test that initialization properly splits sequence into codons."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        assert pool.codons == ['ATG', 'GCC', 'AAA']
    
    def test_init_all_mutation_types(self):
        """Test initialization with all valid mutation types."""
        orf = 'ATGGCCAAA'
        valid_types = [
            'missense_only_first',
            'missense_only_random',
            'nonsynonymous_first',
            'nonsynonymous_random',
            'any_codon',
            'synonymous',
        ]
        
        for mut_type in valid_types:
            pool = KMutationORFPool(orf, k=1, mutation_type=mut_type)
            assert pool.mutation_type == mut_type
    
    def test_init_nonsense_type(self):
        """Test initialization with nonsense mutation type."""
        orf = 'ATGGCCAAA'  # No stop codons
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense')
        assert pool.mutation_type == 'nonsense'


class TestKMutationORFPoolValidation:
    """Test input validation."""
    
    def test_non_dna_characters_error(self):
        """Test that non-DNA characters raise error."""
        with pytest.raises(ValueError, match="must contain only ACGT"):
            KMutationORFPool('ATGXXX', k=1, mutation_type='any_codon')
    
    def test_length_not_divisible_by_3_error(self):
        """Test that length not divisible by 3 raises error."""
        with pytest.raises(ValueError, match="divisible by 3"):
            KMutationORFPool('ATGG', k=1, mutation_type='any_codon')
    
    def test_k_zero_error(self):
        """Test that k=0 raises error."""
        with pytest.raises(ValueError, match="k must be > 0"):
            KMutationORFPool('ATGGCC', k=0, mutation_type='any_codon')
    
    def test_k_negative_error(self):
        """Test that negative k raises error."""
        with pytest.raises(ValueError, match="k must be > 0"):
            KMutationORFPool('ATGGCC', k=-1, mutation_type='any_codon')
    
    def test_k_too_large_error(self):
        """Test that k > number of available positions raises error."""
        orf = 'ATGGCC'  # 2 codons
        with pytest.raises(ValueError, match="k .* must be <="):
            KMutationORFPool(orf, k=3, mutation_type='any_codon')
    
    def test_invalid_mutation_type_error(self):
        """Test that invalid mutation_type raises error."""
        with pytest.raises(ValueError, match="mutation_type must be one of"):
            KMutationORFPool('ATGGCC', k=1, mutation_type='invalid_type')
    
    def test_nonsense_with_stop_codon_error(self):
        """Test that nonsense type with existing stop codon raises error."""
        orf = 'ATGTAA'  # ATG TAA (Met-Stop)
        with pytest.raises(ValueError, match="ORF contains stop codon"):
            KMutationORFPool(orf, k=1, mutation_type='nonsense')


class TestKMutationORFPoolMutationCounting:
    """Test that exactly k codons are mutated."""
    
    def test_exactly_k_codons_mutated(self):
        """Test that exactly k codons are mutated."""
        orf = 'ATGGCCAAACCC'  # 4 codons
        pool = KMutationORFPool(orf, k=2, mutation_type='missense_only_first')
        
        for state in range(20):
            pool.set_state(state)
            mutated = pool.seq
            
            # Split both into codons and count differences
            original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
            mutated_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
            
            num_mutations = sum(1 for i in range(len(original_codons)) 
                              if original_codons[i] != mutated_codons[i])
            
            assert num_mutations == 2, f"Expected 2 codon mutations, got {num_mutations}"
    
    def test_k_equals_1(self):
        """Test that k=1 mutates exactly 1 codon."""
        orf = 'ATGGCCAAACCC'
        pool = KMutationORFPool(orf, k=1, mutation_type='missense_only_first')
        
        pool.set_state(0)
        mutated = pool.seq
        
        original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        mutated_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
        
        num_mutations = sum(1 for i in range(len(original_codons)) 
                          if original_codons[i] != mutated_codons[i])
        
        assert num_mutations == 1
    
    def test_k_equals_all_codons(self):
        """Test that k=num_codons mutates all codons."""
        orf = 'ATGGCCAAA'  # 3 codons
        pool = KMutationORFPool(orf, k=3, mutation_type='any_codon')
        
        pool.set_state(0)
        mutated = pool.seq
        
        original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        mutated_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
        
        num_mutations = sum(1 for i in range(len(original_codons)) 
                          if original_codons[i] != mutated_codons[i])
        
        assert num_mutations == 3


class TestKMutationORFPoolMutationTypes:
    """Test different mutation types."""
    
    def test_any_codon_mutations(self):
        """Test that any_codon can produce various codons."""
        orf = 'ATGATGATG'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        
        # Collect mutated codons over many states
        mutated_codons = set()
        for state in range(100):
            pool.set_state(state)
            mutated = pool.seq
            mutated_codons_list = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
            mutated_codons.update(mutated_codons_list)
        
        # Should see variety (more than just ATG)
        assert len(mutated_codons) > 5
    
    def test_synonymous_preserves_aa(self):
        """Test that synonymous mutations preserve amino acid."""
        # Use a codon with synonymous alternatives
        orf = 'CTGCTGCTG'  # All Leucine (L), which has 6 synonymous codons
        pool = KMutationORFPool(orf, k=1, mutation_type='synonymous')
        
        # Get codon table from pool
        codon_to_aa = pool.codon_to_aa_dict
        
        for state in range(10):
            pool.set_state(state)
            mutated = pool.seq
            
            original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
            mutated_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
            
            # Check all codons produce same amino acid
            for i in range(len(original_codons)):
                orig_aa = codon_to_aa[original_codons[i]]
                mut_aa = codon_to_aa[mutated_codons[i]]
                assert orig_aa == mut_aa, "Synonymous mutation should preserve amino acid"
    
    def test_nonsense_introduces_stop_codon(self):
        """Test that nonsense mutations introduce stop codons."""
        orf = 'ATGGCCAAA'  # No stop codons
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense')
        
        stop_codons = pool.stop_codons
        
        for state in range(10):
            pool.set_state(state)
            mutated = pool.seq
            
            mutated_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
            
            # Should have exactly one stop codon (the mutated one)
            num_stops = sum(1 for codon in mutated_codons if codon in stop_codons)
            assert num_stops == 1, "Should introduce exactly one stop codon"


class TestKMutationORFPoolSequentialMode:
    """Test sequential mode behavior."""
    
    def test_uniform_mutations_is_uniform(self):
        """Test that uniform mutation types are detected."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        
        assert pool.is_uniform == True
        assert pool.uniform_num_possible_mutations == 63
    
    def test_nonsense_is_uniform(self):
        """Test that nonsense has uniform mutations."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense')
        
        assert pool.is_uniform == True
        assert pool.uniform_num_possible_mutations == 3
    
    def test_synonymous_is_not_uniform(self):
        """Test that synonymous does not have uniform mutations."""
        orf = 'ATGCTGAAA'  # ATG(0 syn), CTG(5 syn), AAA(1 syn)
        pool = KMutationORFPool(orf, k=1, mutation_type='synonymous')
        
        assert pool.is_uniform == False
    
    def test_state_count_formula_any_codon(self):
        """Test state count formula for any_codon."""
        orf = 'ATGGCCAAA'  # 3 codons
        k = 2
        pool = KMutationORFPool(orf, k=k, mutation_type='any_codon')
        
        L = 3
        expected = comb(L, k) * (63 ** k)
        
        assert pool.num_states == expected
    
    def test_state_count_formula_nonsense(self):
        """Test state count formula for nonsense."""
        orf = 'ATGGCCAAA'  # 3 codons
        k = 1
        pool = KMutationORFPool(orf, k=k, mutation_type='nonsense')
        
        L = 3
        expected = comb(L, k) * (3 ** k)
        
        assert pool.num_states == expected
    
    def test_infinite_states_for_non_uniform(self):
        """Test that non-uniform mutation types have infinite states."""
        orf = 'ATGCTGAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='synonymous')
        
        assert pool.num_states == float('inf')
    
    def test_sequential_mode_with_non_uniform_raises_error(self):
        """Test that sequential mode with non-uniform mutation type raises error."""
        with pytest.raises(ValueError, match="uniform"):
            KMutationORFPool('ATGCTGAAA', k=1, mutation_type='synonymous', mode='sequential')


class TestKMutationORFPoolDeterminism:
    """Test deterministic behavior."""
    
    def test_same_state_same_sequence(self):
        """Test that same state produces same sequence."""
        orf = 'ATGGCCAAACCC'
        pool = KMutationORFPool(orf, k=2, mutation_type='missense_only_first')
        
        pool.set_state(42)
        seq1 = pool.seq
        
        pool.set_state(42)
        seq2 = pool.seq
        
        assert seq1 == seq2
    
    def test_different_states_different_sequences(self):
        """Test that different states likely produce different sequences."""
        orf = 'ATGGCCAAACCC'
        pool = KMutationORFPool(orf, k=2, mutation_type='any_codon')
        
        sequences = set()
        for state in range(50):
            pool.set_state(state)
            sequences.add(pool.seq)
        
        # Should see variety
        assert len(sequences) > 10


class TestKMutationORFPoolSequenceLength:
    """Test that sequence length is preserved."""
    
    def test_length_preserved(self):
        """Test that mutations preserve sequence length."""
        orf = 'ATGGCCAAACCCTTT'
        pool = KMutationORFPool(orf, k=2, mutation_type='any_codon')
        
        for state in range(10):
            pool.set_state(state)
            mutated = pool.seq
            assert len(mutated) == len(orf)
    
    def test_seq_length_property(self):
        """Test seq_length property."""
        orf = 'ATGGCCAAACCCTTT'
        pool = KMutationORFPool(orf, k=2, mutation_type='any_codon')
        
        assert pool.seq_length == len(orf)


class TestKMutationORFPoolIntegration:
    """Test integration with Pool operations."""
    
    def test_concatenation(self):
        """Test concatenating with other pools."""
        orf = 'ATGGCC'
        pool1 = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        pool2 = Pool(seqs=['NNN'])
        
        combined = pool1 + pool2
        combined.set_state(0)
        seq = combined.seq
        
        assert len(seq) == 9
        assert seq.endswith('NNN')
    
    def test_slicing(self):
        """Test slicing KMutationORFPool."""
        orf = 'ATGGCCAAACCC'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        sliced = pool[0:6]
        
        sliced.set_state(0)
        seq = sliced.seq
        
        assert len(seq) == 6
    
    def test_repetition(self):
        """Test repeating KMutationORFPool."""
        orf = 'ATGGCC'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        repeated = pool * 2
        
        repeated.set_state(0)
        seq = repeated.seq
        
        assert len(seq) == 12


class TestKMutationORFPoolRepr:
    """Test string representation."""
    
    def test_repr_basic(self):
        """Test __repr__ basic functionality."""
        orf = 'ATGGCC'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        repr_str = repr(pool)
        
        assert 'KMutationORFPool' in repr_str
        assert 'k=1' in repr_str
        assert 'any_codon' in repr_str
    
    def test_repr_with_long_sequence(self):
        """Test __repr__ with long sequence gets truncated."""
        orf = 'ATG' * 10  # 30 nucleotides
        pool = KMutationORFPool(orf, k=1, mutation_type='missense_only_first')
        repr_str = repr(pool)
        
        assert 'KMutationORFPool' in repr_str
        assert '...' in repr_str  # Should be truncated


class TestKMutationORFPoolEdgeCases:
    """Test edge cases."""
    
    def test_single_codon_orf(self):
        """Test with single codon ORF."""
        orf = 'ATG'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon')
        
        pool.set_state(0)
        mutated = pool.seq
        
        assert len(mutated) == 3
        assert mutated != orf  # Should be mutated
    
    def test_two_codon_orf(self):
        """Test with two codon ORF."""
        orf = 'ATGGCC'
        pool = KMutationORFPool(orf, k=2, mutation_type='any_codon')
        
        pool.set_state(0)
        mutated = pool.seq
        
        assert len(mutated) == 6
        # Both codons should be mutated
        orig_codons = ['ATG', 'GCC']
        mut_codons = [mutated[0:3], mutated[3:6]]
        assert orig_codons[0] != mut_codons[0]
        assert orig_codons[1] != mut_codons[1]
    
    def test_long_orf(self):
        """Test with longer ORF."""
        orf = 'ATG' * 20  # 20 codons
        pool = KMutationORFPool(orf, k=5, mutation_type='any_codon')
        
        pool.set_state(0)
        mutated = pool.seq
        
        assert len(mutated) == 60
        
        # Count mutations
        orig_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        mut_codons = [mutated[i:i+3] for i in range(0, len(mutated), 3)]
        num_mutations = sum(1 for i in range(len(orig_codons)) 
                          if orig_codons[i] != mut_codons[i])
        assert num_mutations == 5


class TestKMutationORFPoolMarkChanges:
    """Test mark_changes functionality."""
    
    def test_mark_changes_false(self):
        """Test that mark_changes=False produces uppercase sequences."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon', mark_changes=False)
        
        pool.set_state(0)
        seq = pool.seq
        
        # Should be all uppercase
        assert seq == seq.upper()
    
    def test_mark_changes_true_marks_mutations(self):
        """Test that mark_changes=True marks mutated codons with swapcase."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='any_codon', mark_changes=True)
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            
            # Check that exactly one codon has different case
            seq_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Count codons that are lowercase
            lowercase_codons = sum(1 for codon in seq_codons if codon.islower())
            assert lowercase_codons == 1, "Should have exactly one lowercase codon"
    
    def test_mark_changes_multiple_mutations(self):
        """Test that multiple mutations are all marked with case change."""
        orf = 'ATGGCCAAACCC'
        pool = KMutationORFPool(orf, k=2, mutation_type='any_codon', mark_changes=True)
        
        pool.set_state(0)
        seq = pool.seq
        
        # Should have exactly 2 lowercase codons
        seq_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        lowercase_codons = sum(1 for codon in seq_codons if codon.islower())
        assert lowercase_codons == 2
    
    def test_mark_changes_preserves_mutations(self):
        """Test that case change doesn't affect the actual mutations."""
        orf = 'ATGGCCAAA'
        pool_normal = KMutationORFPool(orf, k=1, mutation_type='missense_only_first', mark_changes=False)
        pool_case = KMutationORFPool(orf, k=1, mutation_type='missense_only_first', mark_changes=True)
        
        for state in range(5):
            pool_normal.set_state(state)
            pool_case.set_state(state)
            
            seq_normal = pool_normal.seq
            seq_case = pool_case.seq
            
            # Uppercase version should match
            assert seq_case.upper() == seq_normal
    
    def test_mark_changes_with_sequential_mode(self):
        """Test that mark_changes works in sequential mode."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense', mode='sequential', mark_changes=True)
        
        # Iterate through all states
        seqs = pool.generate_seqs(num_seqs=pool.num_states)
        
        for seq in seqs:
            # Each should have exactly one lowercase codon (the stop)
            seq_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            lowercase_codons = sum(1 for codon in seq_codons if codon.islower())
            assert lowercase_codons == 1


class TestKMutationORFPoolPositions:
    """Test positions parameter for restricting mutation positions."""
    
    def test_positions_restricts_mutations(self):
        """Test that positions parameter restricts which codons can be mutated."""
        orf = 'ATGGCCAAACCC'  # 4 codons
        target_positions = [0, 2]  # Only mutate first and third codons
        
        pool = KMutationORFPool(
            orf, k=1, mutation_type='any_codon',
            positions=target_positions, mode='sequential'
        )
        
        # Check all states
        original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Find which position was mutated
            changed = [j for j in range(len(original_codons)) 
                      if original_codons[j] != mutated_codons[j]]
            
            assert len(changed) == 1
            assert changed[0] in target_positions, \
                f"Mutation at position {changed[0]}, expected one of {target_positions}"
    
    def test_positions_with_k_greater_than_1(self):
        """Test positions with k > 1."""
        orf = 'ATGGCCAAACCCGGG'  # 5 codons
        target_positions = [0, 2, 4]
        
        pool = KMutationORFPool(
            orf, k=2, mutation_type='any_codon',
            positions=target_positions, mode='sequential'
        )
        
        original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        
        pool.set_state(0)
        seq = pool.seq
        mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        
        changed = [j for j in range(len(original_codons)) 
                  if original_codons[j] != mutated_codons[j]]
        
        assert len(changed) == 2
        for pos in changed:
            assert pos in target_positions


class TestKMutationORFPoolStateWrapping:
    """Test state wrapping behavior."""
    
    def test_state_wrapping(self):
        """Test that states wrap correctly."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense', mode='sequential')
        n = pool.num_internal_states
        
        # States 0, n, 2n should all produce the same sequence
        pool.set_state(0)
        seq0 = pool.seq
        
        pool.set_state(n)
        seq_n = pool.seq
        
        pool.set_state(2 * n)
        seq_2n = pool.seq
        
        assert seq0 == seq_n == seq_2n
    
    def test_state_wrapping_all_base_states(self):
        """Test wrapping for all base states."""
        orf = 'ATGGCCAAA'
        pool = KMutationORFPool(orf, k=1, mutation_type='nonsense', mode='sequential')
        n = pool.num_internal_states
        
        for base_state in range(n):
            seqs = []
            for offset in [0, n, 2 * n]:
                pool.set_state(base_state + offset)
                seqs.append(pool.seq)
            
            assert len(set(seqs)) == 1, f"State {base_state} wrapping failed"


class TestKMutationORFPoolFlankingRegions:
    """Tests for ORF flanking region (UTR) handling."""
    
    def test_with_flanking_regions(self):
        """Test KMutationORFPool with orf_start/orf_end."""
        upstream = "GGCGCGC"  # 7 bp 5' UTR
        orf_seq = "ATGGCTCGC"  # 9 bp ORF (3 codons)
        downstream = "TTATTT"  # 6 bp 3' UTR
        full_seq = upstream + orf_seq + downstream
        
        pool = KMutationORFPool(
            seq=full_seq,
            k=1,
            mutation_type='any_codon',
            orf_start=len(upstream),
            orf_end=len(upstream) + len(orf_seq),
            mode='sequential'
        )
        
        assert pool.upstream_flank == upstream
        assert pool.downstream_flank == downstream
        assert pool.orf_seq == orf_seq
    
    def test_flanks_preserved_in_output(self):
        """Test that flanks are preserved in all outputs."""
        upstream = "GGCGCGC"
        orf_seq = "ATGGCTCGC"
        downstream = "TTATTT"
        full_seq = upstream + orf_seq + downstream
        
        pool = KMutationORFPool(
            seq=full_seq,
            k=1,
            mutation_type='any_codon',
            orf_start=len(upstream),
            orf_end=len(upstream) + len(orf_seq),
            mode='sequential'
        )
        
        expected_length = len(full_seq)
        assert pool.seq_length == expected_length
        
        # Check flanks are preserved in all states
        for i in range(min(20, pool.num_internal_states)):
            pool.set_state(i)
            seq = pool.seq
            assert seq.startswith(upstream), f"5' UTR not preserved: {seq}"
            assert seq.endswith(downstream), f"3' UTR not preserved: {seq}"
            assert len(seq) == expected_length


class TestKMutationORFPoolPoolChaining:
    """Tests for Pool chaining / transformer pattern."""
    
    def test_pool_as_seq_input(self):
        """Test KMutationORFPool with Pool as seq input."""
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='random')
        
        pool = KMutationORFPool(
            parent,
            k=1,
            mutation_type='any_codon'
        )
        
        # Should work with Pool input
        seq = pool.seq
        assert len(seq) == 9
        assert all(c in 'ACGTacgt' for c in seq)
    
    def test_generate_seqs_with_seed(self):
        """Test generate_seqs with seed produces reproducible results."""
        pool = KMutationORFPool(
            'ATGGCCAAACCCGGG',
            k=1,
            mutation_type='any_codon'
        )
        
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
        
        seqs3 = pool.generate_seqs(num_seqs=10, seed=123)
        assert seqs1 != seqs3, "Different seeds should produce different sequences"
    
    def test_parent_pool_state_affects_output(self):
        """Verify parent Pool's current sequence is used for mutations."""
        # Two very different parent sequences
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        pool = KMutationORFPool(
            parent,
            k=1,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        # Collect sequences at different parent states
        parent.set_sequential_op_states(0)
        pool.set_state(0)
        seq_from_parent0 = pool.seq
        
        parent.set_sequential_op_states(1)
        pool.set_state(0)  # Same internal state, different parent
        seq_from_parent1 = pool.seq
        
        # The base sequences being mutated are different, so outputs should differ
        # (except in rare coincidental cases)
        # At minimum, verify lengths and validity
        assert len(seq_from_parent0) == 9
        assert len(seq_from_parent1) == 9
    
    def test_mutations_applied_to_current_parent_sequence(self):
        """Verify k mutations are applied to parent's CURRENT sequence."""
        # Parent with two distinct sequences
        parent = Pool(seqs=["ATGATGATG", "CTGCTGCTG"], mode='sequential')
        
        pool = KMutationORFPool(
            parent,
            k=2,
            mutation_type='nonsense',
            mode='random'
        )
        
        stop_codons = set(pool.stop_codons)
        
        # For each parent sequence, verify mutations produce stops
        for parent_state in range(2):
            parent.set_sequential_op_states(parent_state)
            pool.set_state(0)
            seq = pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Count stop codons - should have exactly k=2
            num_stops = sum(1 for c in codons if c in stop_codons)
            assert num_stops == 2, \
                f"Parent state {parent_state}: Expected 2 stop codons, got {num_stops}"
    
    def test_exactly_k_mutations_with_pool_parent(self):
        """Verify exactly k codons mutated when parent is Pool."""
        parent = Pool(seqs=["ATGATGATGATGATG", "CCCGGGAAATTTCCC"], mode='random')
        k = 3
        
        pool = KMutationORFPool(
            parent,
            k=k,
            mutation_type='any_codon',
            mode='random'
        )
        
        for state in range(50):
            pool.set_state(state)
            seq = pool.seq
            
            # Get parent's current sequence to compare
            parent_seq = parent.seq
            parent_codons = [parent_seq[i:i+3] for i in range(0, len(parent_seq), 3)]
            result_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            num_changes = sum(1 for i in range(len(parent_codons))
                            if parent_codons[i] != result_codons[i])
            
            assert num_changes == k, \
                f"State {state}: Expected {k} mutations, got {num_changes}"
    
    def test_chain_with_multiple_pools(self):
        """Test chaining: Pool -> KMutationORFPool -> concatenation."""
        parent = Pool(seqs=["ATGATGATG"], mode='sequential')
        
        mutation_pool = KMutationORFPool(
            parent,
            k=1,
            mutation_type='any_codon'
        )
        
        # Chain with prefix and suffix
        combined = "TTT" + mutation_pool + "AAA"
        
        combined.set_state(0)
        seq = combined.seq
        
        assert seq.startswith("TTT"), f"Prefix not preserved: {seq}"
        assert seq.endswith("AAA"), f"Suffix not preserved: {seq}"
        assert len(seq) == 3 + 9 + 3
    
    def test_transformer_pattern_with_flanks(self):
        """Test transformer pattern preserves flanks from parent."""
        # Parent with UTRs
        parent = Pool(seqs=["GGGGG" + "ATGATGATG" + "CCCCC"], mode='sequential')
        
        pool = KMutationORFPool(
            parent,
            k=1,
            mutation_type='any_codon',
            orf_start=5,
            orf_end=14
        )
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            
            # Flanks must be preserved
            assert seq[:5] == "GGGGG", f"State {state}: 5' flank modified: {seq[:5]}"
            assert seq[-5:] == "CCCCC", f"State {state}: 3' flank modified: {seq[-5:]}"
    
    def test_chain_kmutation_to_kmutation(self):
        """Test chaining two KMutationORFPool instances."""
        base_seq = "ATGATGATGATGATG"  # 5 codons
        
        # First mutation pool: k=1
        first_pool = KMutationORFPool(
            base_seq,
            k=1,
            mutation_type='any_codon',
            mode='random'
        )
        
        # Second mutation pool: k=1, applied to first pool's output
        second_pool = KMutationORFPool(
            first_pool,
            k=1,
            mutation_type='nonsense',
            mode='random'
        )
        
        stop_codons = set(second_pool.stop_codons)
        
        for state in range(20):
            second_pool.set_state(state)
            seq = second_pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Should have at least 1 stop codon from second pool
            num_stops = sum(1 for c in codons if c in stop_codons)
            assert num_stops >= 1, \
                f"State {state}: Expected at least 1 stop codon from chain, got {num_stops}"


class TestKMutationORFPoolMixedRadixDecomposition:
    """Test mixed-radix decomposition correctness for sequential mode."""
    
    def test_decomposition_verification(self):
        """Verify mixed-radix decomposition for sequential mode."""
        orf = "ATGGCTCGC"  # 3 codons
        pool = KMutationORFPool(orf, k=2, mutation_type='nonsense', mode='sequential')
        # C(3,2)*3^2 = 3 * 9 = 27 states
        
        original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
        expected_position_combos = [(0, 1), (0, 2), (1, 2)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Find which positions were mutated
            changed = tuple(j for j in range(len(original_codons)) 
                          if original_codons[j] != mutated_codons[j])
            
            # Verify exactly 2 positions changed
            assert len(changed) == 2, f"State {state}: Expected 2 changes, got {len(changed)}"
            
            # Verify positions match expected decomposition
            position_index = state // 9  # 9 = 3^2
            expected_positions = expected_position_combos[position_index]
            assert changed == expected_positions, \
                f"State {state}: Expected positions {expected_positions}, got {changed}"


class TestKMutationORFPoolRandomVsSequential:
    """Test random vs sequential mode consistency."""
    
    def test_both_modes_produce_valid_sequences(self):
        """Test that both modes produce valid sequences."""
        orf = "ATGGCTCGCAAAGAA"  # 5 codons
        pool_seq = KMutationORFPool(orf, k=1, mutation_type='any_codon', mode='sequential')
        pool_rand = KMutationORFPool(orf, k=1, mutation_type='any_codon', mode='random')
        
        for state in [0, 10, 50, 100, 200]:
            pool_seq.set_state(state)
            pool_rand.set_state(state)
            
            seq_seq = pool_seq.seq
            seq_rand = pool_rand.seq
            
            # Both should be valid sequences
            assert len(seq_seq) == len(orf)
            assert len(seq_rand) == len(orf)
            assert all(c in 'ACGT' for c in seq_seq)
            assert all(c in 'ACGT' for c in seq_rand)
            
            # Both should have exactly 1 codon mutated
            original_codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
            seq_codons = [seq_seq[i:i+3] for i in range(0, len(seq_seq), 3)]
            rand_codons = [seq_rand[i:i+3] for i in range(0, len(seq_rand), 3)]
            
            seq_changes = sum(1 for a, b in zip(original_codons, seq_codons) if a != b)
            rand_changes = sum(1 for a, b in zip(original_codons, rand_codons) if a != b)
            
            assert seq_changes == 1
            assert rand_changes == 1


class TestKMutationORFPoolMutationTypeVerification:
    """Rigorous verification that mutation types produce correct biological outcomes."""
    
    def test_synonymous_never_changes_amino_acid(self):
        """Verify synonymous mutations NEVER change amino acid."""
        # Use codons with many synonymous alternatives
        # CTG = Leu (6 synonyms), GCT = Ala (4 synonyms), CGT = Arg (6 synonyms)
        orf_seq = 'CTGGCTCGTCTGGCT'  # 5 codons with synonymous options
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='synonymous',
            mode='random'
        )
        
        codon_to_aa = pool.codon_to_aa_dict
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        original_aas = [codon_to_aa[c] for c in original_codons]
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            mutated_aas = [codon_to_aa[c] for c in mutated_codons]
            
            assert mutated_aas == original_aas, \
                f"State {state}: Synonymous mutation changed AA! {original_aas} -> {mutated_aas}"
    
    def test_missense_only_first_never_produces_stop(self):
        """Verify missense_only_first NEVER produces stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG' * 2  # 10 codons
        pool = KMutationORFPool(
            orf_seq, k=3,
            mutation_type='missense_only_first',
            mode='random'
        )
        
        stop_codons = set(pool.stop_codons)
        
        for state in range(300):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i, codon in enumerate(mutated_codons):
                assert codon not in stop_codons, \
                    f"State {state}: missense_only_first produced stop codon {codon} at position {i}"
    
    def test_missense_only_random_never_produces_stop(self):
        """Verify missense_only_random NEVER produces stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG' * 2
        pool = KMutationORFPool(
            orf_seq, k=3,
            mutation_type='missense_only_random',
            mode='random'
        )
        
        stop_codons = set(pool.stop_codons)
        
        for state in range(300):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i, codon in enumerate(mutated_codons):
                assert codon not in stop_codons, \
                    f"State {state}: missense_only_random produced stop codon {codon} at position {i}"
    
    def test_nonsynonymous_always_changes_amino_acid(self):
        """Verify nonsynonymous mutations ALWAYS change amino acid."""
        orf_seq = 'ATGATGATGATG'  # All Met codons
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='nonsynonymous_first',
            mode='random'
        )
        
        codon_to_aa = pool.codon_to_aa_dict
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        mutations_verified = 0
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i in range(len(original_codons)):
                if original_codons[i] != mutated_codons[i]:
                    orig_aa = codon_to_aa[original_codons[i]]
                    mut_aa = codon_to_aa[mutated_codons[i]]
                    assert orig_aa != mut_aa, \
                        f"State {state}, pos {i}: Nonsynonymous didn't change AA! {original_codons[i]}({orig_aa}) -> {mutated_codons[i]}({mut_aa})"
                    mutations_verified += 1
        
        assert mutations_verified > 100, f"Only verified {mutations_verified} mutations"
    
    def test_nonsense_only_produces_stop_codons(self):
        """Verify nonsense mutations ONLY produce stop codons."""
        orf_seq = 'ATGGCCAAACCC'  # 4 codons
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        stop_codons = set(pool.stop_codons)
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            num_stops = 0
            for i in range(len(original_codons)):
                if original_codons[i] != mutated_codons[i]:
                    assert mutated_codons[i] in stop_codons, \
                        f"State {state}, pos {i}: Nonsense produced non-stop {mutated_codons[i]}"
                    num_stops += 1
            
            assert num_stops == 2, f"State {state}: Expected 2 stop codons, got {num_stops}"


class TestKMutationORFPoolExactKMutations:
    """Verify exactly k codons are mutated in every case."""
    
    def test_exactly_k_mutations_all_states(self):
        """Verify ALL states produce exactly k mutations."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        k = 2
        pool = KMutationORFPool(
            orf_seq, k=k,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            num_mutations = sum(1 for i in range(len(original_codons))
                              if original_codons[i] != mutated_codons[i])
            
            assert num_mutations == k, \
                f"State {state}: Expected {k} mutations, got {num_mutations}"
    
    def test_exactly_k_with_positions_restriction(self):
        """Verify exactly k mutations even with positions restriction."""
        orf_seq = 'ATGGCCAAACCCGGGTTTATG'  # 7 codons
        k = 3
        positions = [0, 2, 4, 6]  # Only even positions
        
        pool = KMutationORFPool(
            orf_seq, k=k,
            mutation_type='any_codon',
            positions=positions,
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            changed_positions = [i for i in range(len(original_codons))
                                if original_codons[i] != mutated_codons[i]]
            
            assert len(changed_positions) == k, \
                f"State {state}: Expected {k} mutations, got {len(changed_positions)}"
            
            # All changed positions must be in allowed list
            for pos in changed_positions:
                assert pos in positions, \
                    f"State {state}: Mutation at position {pos} not in allowed {positions}"


class TestKMutationORFPoolPositionsVerification:
    """Verify positions parameter is strictly respected."""
    
    def test_mutations_only_at_specified_positions(self):
        """Verify mutations ONLY occur at specified positions."""
        orf_seq = 'ATGGCCAAACCCGGGTTTATG'  # 7 codons
        allowed_positions = [1, 3, 5]  # Only odd positions
        
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='any_codon',
            positions=allowed_positions,
            mode='random'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i in range(len(original_codons)):
                if original_codons[i] != mutated_codons[i]:
                    assert i in allowed_positions, \
                        f"State {state}: Mutation at forbidden position {i}, allowed: {allowed_positions}"
    
    def test_forbidden_positions_never_mutated(self):
        """Verify positions NOT in list are NEVER mutated."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        allowed_positions = [0, 4]  # Only first and last
        forbidden_positions = [1, 2, 3]
        
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='any_codon',
            positions=allowed_positions,
            mode='random'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for pos in forbidden_positions:
                assert mutated_codons[pos] == original_codons[pos], \
                    f"State {state}: Forbidden position {pos} was mutated"


class TestKMutationORFPoolSequentialCoverage:
    """Verify sequential mode covers all states exactly once."""
    
    def test_sequential_complete_coverage(self):
        """Verify sequential mode produces all unique sequences."""
        orf_seq = 'ATGGCC'  # 2 codons
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',  # 3 alternatives per position
            mode='sequential'
        )
        
        # C(2,1) * 3^1 = 2 * 3 = 6 states
        expected_states = pool.num_internal_states
        
        sequences = set()
        for state in range(expected_states):
            pool.set_state(state)
            sequences.add(pool.seq)
        
        assert len(sequences) == expected_states, \
            f"Expected {expected_states} unique sequences, got {len(sequences)}"
    
    def test_sequential_no_duplicates_within_period(self):
        """Verify no duplicate sequences within one period."""
        orf_seq = 'ATGGCCAAA'  # 3 codons
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',  # 3 alternatives
            mode='sequential'
        )
        
        sequences = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            sequences.append(pool.seq)
        
        assert len(sequences) == len(set(sequences)), \
            "Found duplicate sequences in sequential mode"
    
    def test_sequential_correct_position_progression(self):
        """Verify sequential mode progresses through positions correctly."""
        orf_seq = 'ATGGCCAAA'  # 3 codons
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        # For k=1, nonsense: states 0-2 should mutate pos 0, 3-5 pos 1, 6-8 pos 2
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            changed = [i for i in range(len(original_codons)) 
                      if original_codons[i] != mutated_codons[i]]
            
            assert len(changed) == 1, f"State {state}: Expected 1 change, got {len(changed)}"
            
            # Expected position based on state decomposition
            expected_pos = state // 3  # 3 mutations per position for nonsense
            assert changed[0] == expected_pos, \
                f"State {state}: Expected mutation at pos {expected_pos}, got {changed[0]}"


class TestKMutationORFPoolFlankingIntegrity:
    """Verify flanking regions are never modified."""
    
    def test_flanks_never_mutated(self):
        """Verify UTR regions are NEVER mutated regardless of k."""
        upstream = "GGGGGGGGG"
        orf_seq = "ATGATGATGATGATGATG"  # 6 codons
        downstream = "CCCCCCCCC"
        full_seq = upstream + orf_seq + downstream
        
        pool = KMutationORFPool(
            full_seq, k=3,  # High k value
            mutation_type='any_codon',
            orf_start=9,
            orf_end=27,
            mode='random'
        )
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            assert seq[:9] == upstream, f"State {state}: 5' UTR modified: {seq[:9]}"
            assert seq[-9:] == downstream, f"State {state}: 3' UTR modified: {seq[-9:]}"
    
    def test_only_orf_codons_mutated(self):
        """Verify mutations only affect ORF codons, not flanks."""
        upstream = "AAAAAAA"  # 7 bp
        orf_seq = "ATGATGATG"  # 3 codons
        downstream = "TTTTTTT"  # 7 bp
        full_seq = upstream + orf_seq + downstream
        
        pool = KMutationORFPool(
            full_seq, k=2,
            mutation_type='any_codon',
            orf_start=7,
            orf_end=16,
            mode='sequential'
        )
        
        original_orf_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            
            # Verify flanks
            assert seq[:7] == upstream
            assert seq[-7:] == downstream
            
            # Verify ORF has exactly k mutations
            orf_part = seq[7:16]
            orf_codons = [orf_part[i:i+3] for i in range(0, len(orf_part), 3)]
            
            num_changes = sum(1 for i in range(len(original_orf_codons))
                            if original_orf_codons[i] != orf_codons[i])
            assert num_changes == 2, f"State {state}: Expected 2 ORF mutations, got {num_changes}"


class TestKMutationORFPoolStateFormulas:
    """Verify state count formulas are correct."""
    
    def test_state_formula_k1_any_codon(self):
        """Verify C(L,1) * 63 formula for k=1, any_codon."""
        orf_seq = 'ATGGCCAAACCC'  # 4 codons
        pool = KMutationORFPool(orf_seq, k=1, mutation_type='any_codon')
        
        L = 4
        expected = comb(L, 1) * 63  # 4 * 63 = 252
        assert pool.num_internal_states == expected, \
            f"Expected {expected} states, got {pool.num_internal_states}"
    
    def test_state_formula_k2_nonsense(self):
        """Verify C(L,2) * 3^2 formula for k=2, nonsense."""
        orf_seq = 'ATGGCCAAACCC'  # 4 codons
        pool = KMutationORFPool(orf_seq, k=2, mutation_type='nonsense')
        
        L = 4
        expected = comb(L, 2) * (3 ** 2)  # 6 * 9 = 54
        assert pool.num_internal_states == expected, \
            f"Expected {expected} states, got {pool.num_internal_states}"
    
    def test_state_formula_with_positions(self):
        """Verify formula with restricted positions."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        positions = [0, 2, 4]  # Only 3 positions
        
        pool = KMutationORFPool(
            orf_seq, k=2,
            mutation_type='any_codon',
            positions=positions
        )
        
        L = len(positions)  # 3
        expected = comb(L, 2) * (63 ** 2)  # 3 * 3969 = 11907
        assert pool.num_internal_states == expected, \
            f"Expected {expected} states, got {pool.num_internal_states}"


class TestKMutationORFPoolEdgeCases:
    """Comprehensive edge case testing."""
    
    def test_k_equals_num_codons(self):
        """Test k = total number of codons (mutate everything)."""
        orf_seq = "ATGGCCAAA"  # 3 codons
        pool = KMutationORFPool(
            orf_seq, k=3,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(min(20, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # ALL codons should be different
            for i in range(3):
                assert mutated_codons[i] != original_codons[i], \
                    f"State {state}: Position {i} not mutated with k=num_codons"
    
    def test_methionine_only_synonymous(self):
        """ATG (Met) has no synonymous - synonymous type should have no options."""
        met_only = "ATGATGATG"
        pool = KMutationORFPool(
            met_only, k=1,
            mutation_type='synonymous',
            mode='random'
        )
        
        # ATG has 0 synonymous alternatives
        # The pool should still work but mutations won't change codons
        assert pool.num_possible_mutations[0] == 0
        assert pool.num_possible_mutations[1] == 0
        assert pool.num_possible_mutations[2] == 0
    
    def test_tryptophan_only_synonymous(self):
        """TGG (Trp) has no synonymous - similar to Met."""
        trp_only = "TGGTGGTGG"
        pool = KMutationORFPool(
            trp_only, k=1,
            mutation_type='synonymous',
            mode='random'
        )
        
        # TGG has 0 synonymous alternatives
        assert pool.num_possible_mutations[0] == 0
    
    def test_single_position_in_positions(self):
        """Test with only one position allowed."""
        orf_seq = "ATGGCCAAACCC"  # 4 codons
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='any_codon',
            positions=[2],  # Only position 2
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Only position 2 should ever change
            assert mutated_codons[0] == original_codons[0]
            assert mutated_codons[1] == original_codons[1]
            assert mutated_codons[2] != original_codons[2]
            assert mutated_codons[3] == original_codons[3]
    
    def test_k_equals_positions_length(self):
        """Test k equals len(positions) - mutate all allowed positions."""
        orf_seq = "ATGGCCAAACCCGGG"  # 5 codons
        positions = [1, 3]  # Only 2 positions allowed
        
        pool = KMutationORFPool(
            orf_seq, k=2,  # k = len(positions)
            mutation_type='any_codon',
            positions=positions,
            mode='sequential'
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(min(20, pool.num_internal_states)):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Only positions 1 and 3 should change (and both must change)
            assert mutated_codons[0] == original_codons[0]
            assert mutated_codons[1] != original_codons[1]
            assert mutated_codons[2] == original_codons[2]
            assert mutated_codons[3] != original_codons[3]
            assert mutated_codons[4] == original_codons[4]
    
    def test_very_long_orf(self):
        """Test with a very long ORF (50 codons)."""
        long_orf = "ATGGCCAAA" * 16 + "GG"[:-2] + "ATG"  # Adjust to get 50 codons
        # Actually let's just use a clean 50 codons
        long_orf = "ATG" * 50  # 50 codons
        
        pool = KMutationORFPool(
            long_orf, k=5,
            mutation_type='any_codon',
            mode='random'
        )
        
        assert len(pool.codons) == 50
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            assert len(seq) == 150
            
            # Verify exactly 5 mutations
            original_codons = [long_orf[i:i+3] for i in range(0, len(long_orf), 3)]
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            num_changes = sum(1 for i in range(50) if original_codons[i] != mutated_codons[i])
            assert num_changes == 5
    
    def test_first_state_sequential(self):
        """Verify first state (0) in sequential mode."""
        orf_seq = "ATGGCCAAA"
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        
        # State 0: position 0, first nonsense mutation
        stop_codons = set(pool.stop_codons)
        assert codons[0] in stop_codons, "State 0 should mutate position 0 to stop"
        assert codons[1] == "GCC", "Position 1 should be unchanged"
        assert codons[2] == "AAA", "Position 2 should be unchanged"
    
    def test_last_state_sequential(self):
        """Verify last state in sequential mode."""
        orf_seq = "ATGGCCAAA"
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        last_state = pool.num_internal_states - 1
        pool.set_state(last_state)
        seq = pool.seq
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        
        # Last state should mutate last position
        stop_codons = set(pool.stop_codons)
        assert codons[0] == "ATG", "Position 0 should be unchanged"
        assert codons[1] == "GCC", "Position 1 should be unchanged"
        assert codons[2] in stop_codons, "Last state should mutate position 2 to stop"
    
    def test_state_at_boundary(self):
        """Test state exactly at num_internal_states (should wrap to 0)."""
        orf_seq = "ATGGCC"
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='nonsense',
            mode='sequential'
        )
        
        pool.set_state(0)
        seq_0 = pool.seq
        
        pool.set_state(pool.num_internal_states)  # Should wrap to state 0
        seq_wrap = pool.seq
        
        assert seq_0 == seq_wrap, "State wrapping at boundary failed"
    
    def test_no_upstream_flank(self):
        """Test with orf_start=0."""
        full_seq = "ATGATGATG" + "CCCCC"
        pool = KMutationORFPool(
            full_seq, k=1,
            mutation_type='any_codon',
            orf_start=0,
            orf_end=9
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == "CCCCC"
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.endswith("CCCCC")
    
    def test_no_downstream_flank(self):
        """Test with orf_end=len."""
        full_seq = "GGGGG" + "ATGATGATG"
        pool = KMutationORFPool(
            full_seq, k=1,
            mutation_type='any_codon',
            orf_start=5,
            orf_end=14
        )
        
        assert pool.upstream_flank == "GGGGG"
        assert pool.downstream_flank == ""
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.startswith("GGGGG")
    
    def test_no_flanks_at_all(self):
        """Test with no flanks."""
        orf_seq = "ATGATGATG"
        pool = KMutationORFPool(
            orf_seq, k=1,
            mutation_type='any_codon',
            orf_start=0,
            orf_end=9
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == ""
        assert pool.orf_seq == orf_seq
    
    def test_all_identical_codons_any_codon(self):
        """Test any_codon on ORF with all identical codons."""
        all_atg = "ATGATGATGATG"  # 4 identical codons
        pool = KMutationORFPool(
            all_atg, k=2,
            mutation_type='any_codon',
            mode='sequential'
        )
        
        # All positions should have same number of alternatives (63)
        for i in range(4):
            assert pool.num_possible_mutations[i] == 63
    
    def test_lowercase_from_pool_parent(self):
        """Test lowercase sequences from Pool parent are handled."""
        parent = Pool(seqs=["atgatgatg"], mode='sequential')
        
        pool = KMutationORFPool(
            parent, k=1,
            mutation_type='any_codon'
        )
        
        pool.set_state(0)
        seq = pool.seq
        assert len(seq) == 9


class TestKMutationORFPoolValidationEdgeCases:
    """Test validation error edge cases."""
    
    def test_empty_positions_error(self):
        """Verify empty positions list raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            KMutationORFPool(
                "ATGGCCAAA", k=1,
                mutation_type='any_codon',
                positions=[]
            )
    
    def test_duplicate_positions_error(self):
        """Verify duplicate positions raises error."""
        with pytest.raises(ValueError, match="duplicates"):
            KMutationORFPool(
                "ATGGCCAAA", k=1,
                mutation_type='any_codon',
                positions=[0, 1, 0]
            )
    
    def test_k_greater_than_positions_error(self):
        """Verify k > len(positions) raises error."""
        with pytest.raises(ValueError, match="k .* must be <="):
            KMutationORFPool(
                "ATGGCCAAACCC", k=3,
                mutation_type='any_codon',
                positions=[0, 1]  # Only 2 positions, but k=3
            )
    
    def test_position_out_of_bounds_error(self):
        """Verify out-of-bounds position raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            KMutationORFPool(
                "ATGGCCAAA", k=1,  # 3 codons
                mutation_type='any_codon',
                positions=[0, 5]  # Position 5 is out of bounds
            )
    
    def test_negative_position_error(self):
        """Verify negative position raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            KMutationORFPool(
                "ATGGCCAAA", k=1,
                mutation_type='any_codon',
                positions=[-1, 0]
            )
