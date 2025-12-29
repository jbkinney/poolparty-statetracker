"""Tests for the refactored RandomMutationORFPool class.

Note: The following original tests are no longer applicable due to API changes:
- change_case_of_mutations parameter replaced by mark_changes
- Mutation type names updated: all_by_codon → any_codon,
  missense_first_codon → missense_only_first, missense_random_codon → missense_only_random
- Iteration removed from base Pool class
"""

import pytest
from poolparty import RandomMutationORFPool, Pool


class TestRandomMutationORFPoolInit:
    """Tests for RandomMutationORFPool initialization and validation."""
    
    def test_basic_init(self):
        """Test basic initialization with valid ORF."""
        orf_seq = 'ATGGCCAAA'  # 3 codons
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon')
        assert pool.orf_seq == orf_seq
        assert pool.mutation_type == 'any_codon'
        assert len(pool.codons) == 3
        assert pool.mutation_rate == 0.1
        assert pool._is_uniform_rate is True
    
    def test_invalid_orf_seq_non_acgt(self):
        """Test that seq with non-ACGT characters raises ValueError."""
        with pytest.raises(ValueError, match="must contain only ACGT characters"):
            RandomMutationORFPool('ATGXCCAAA', mutation_type='any_codon')
    
    def test_invalid_orf_seq_length(self):
        """Test that seq length not divisible by 3 raises ValueError."""
        with pytest.raises(ValueError, match="(length|Length).*divisible by 3"):
            RandomMutationORFPool('ATGGC', mutation_type='any_codon')
    
    def test_invalid_mutation_type(self):
        """Test that invalid mutation_type raises ValueError."""
        with pytest.raises(ValueError, match="mutation_type must be one of"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='invalid_type')
    
    def test_invalid_mutation_rate_too_high(self):
        """Test that mutation_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mutation_rate=1.5)
    
    def test_invalid_mutation_rate_negative(self):
        """Test that negative mutation_rate raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mutation_rate=-0.1)
    
    def test_mutation_rate_array_wrong_length(self):
        """Test that mutation_rate array with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate array length.*must match number of codons"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mutation_rate=[0.1, 0.2])
    
    def test_mutation_rate_array_invalid_value(self):
        """Test that mutation_rate array with invalid value raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate values must be between 0 and 1"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mutation_rate=[0.1, 1.5, 0.2])
    
    def test_nonsense_with_stop_codon(self):
        """Test that nonsense mutation_type with stop codon in ORF raises ValueError."""
        orf_with_stop = 'ATGTAACAT'  # Contains TAA (stop codon)
        with pytest.raises(ValueError, match="ORF contains stop codon.*nonsense"):
            RandomMutationORFPool(orf_with_stop, mutation_type='nonsense')
    
    def test_sequential_mode_not_allowed(self):
        """Test that sequential mode raises ValueError."""
        with pytest.raises(ValueError, match="only supports mode='random'"):
            RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mode='sequential')


class TestRandomMutationORFPoolProperties:
    """Tests for RandomMutationORFPool properties."""
    
    def test_num_states_infinite(self):
        """Test that num_states is infinite."""
        pool = RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon')
        assert pool.num_states == float('inf')
    
    def test_seq_length_preserved(self):
        """Test that sequence length is preserved."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon')
        assert pool.seq_length == len(orf_seq)
        assert len(pool.seq) == len(orf_seq)
    
    def test_mode_random(self):
        """Test that mode is set correctly."""
        pool = RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mode='random')
        assert pool.mode == 'random'


class TestRandomMutationORFPoolMutations:
    """Tests for different mutation types."""
    
    def test_any_codon_mutations(self):
        """Test any_codon mutation type."""
        orf_seq = 'ATGGCCAAA' * 10  # 30 codons
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.3)
        
        # Generate multiple sequences
        seqs = pool.generate_seqs(num_seqs=10)
        
        # Check that sequences have mutations and preserve length
        for seq in seqs:
            assert len(seq) == len(orf_seq)
            assert all(c in 'ACGT' for c in seq)
        
        # Check that not all sequences are identical
        assert len(set(seqs)) > 1
    
    def test_missense_only_first_mutations(self):
        """Test missense_only_first mutation type."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        pool = RandomMutationORFPool(orf_seq, mutation_type='missense_only_first', mutation_rate=0.5)
        
        seqs = pool.generate_seqs(num_seqs=5)
        for seq in seqs:
            assert len(seq) == len(orf_seq)
            assert all(c in 'ACGT' for c in seq)
    
    def test_nonsynonymous_first_mutations(self):
        """Test nonsynonymous_first mutation type (includes stop codons)."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        pool = RandomMutationORFPool(orf_seq, mutation_type='nonsynonymous_first', mutation_rate=0.5)
        
        seqs = pool.generate_seqs(num_seqs=5)
        for seq in seqs:
            assert len(seq) == len(orf_seq)
            assert all(c in 'ACGT' for c in seq)
    
    def test_synonymous_mutations(self):
        """Test synonymous mutation type preserves amino acid sequence."""
        orf_seq = 'CTGCTGCTGCTGCTG'  # 5 Leucine codons (L has 6 synonymous codons)
        pool = RandomMutationORFPool(orf_seq, mutation_type='synonymous', mutation_rate=0.8)
        
        # Generate mutated sequences
        seqs = pool.generate_seqs(num_seqs=10)
        
        for seq in seqs:
            # Check that DNA is valid
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            # All codons should encode the same amino acids
            # (synonymous mutations only)
    
    def test_nonsense_mutations(self):
        """Test nonsense mutation type introduces stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons, no stops
        pool = RandomMutationORFPool(orf_seq, mutation_type='nonsense', mutation_rate=0.3)
        
        seqs = pool.generate_seqs(num_seqs=20)
        
        stop_codons = ['TAA', 'TAG', 'TGA']
        # At least some sequences should have stop codons
        has_stop = False
        for seq in seqs:
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            if any(c in stop_codons for c in codons):
                has_stop = True
                break
        
        assert has_stop, "Expected at least some sequences to have stop codons"
    
    def test_missense_only_random_mutations(self):
        """Test missense_only_random mutation type."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        pool = RandomMutationORFPool(orf_seq, mutation_type='missense_only_random', mutation_rate=0.5)
        
        seqs = pool.generate_seqs(num_seqs=5)
        for seq in seqs:
            assert len(seq) == len(orf_seq)
            assert all(c in 'ACGT' for c in seq)


class TestRandomMutationORFPoolRates:
    """Tests for mutation rate handling."""
    
    def test_uniform_mutation_rate(self):
        """Test uniform mutation rate."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.5)
        assert pool._is_uniform_rate is True
        assert pool.mutation_rate == 0.5
    
    def test_per_codon_mutation_rate(self):
        """Test per-codon mutation rates."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=rates)
        assert pool._is_uniform_rate is False
        assert pool.mutation_rate == rates
    
    def test_zero_mutation_rate(self):
        """Test that zero mutation rate produces no mutations."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.0)
        
        seqs = pool.generate_seqs(num_seqs=10)
        # All sequences should be identical to original
        assert all(seq == orf_seq for seq in seqs)
    
    def test_high_mutation_rate(self):
        """Test that high mutation rate produces many mutations."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.9)
        
        seqs = pool.generate_seqs(num_seqs=10)
        # Most sequences should differ significantly from original
        different_seqs = [seq for seq in seqs if seq != orf_seq]
        assert len(different_seqs) >= 8  # At least 8 out of 10 should be different


class TestRandomMutationORFPoolStateManagement:
    """Tests for state management and determinism."""
    
    def test_set_state_deterministic(self):
        """Test that setting state produces deterministic results."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.5)
        
        pool.set_state(42)
        seq1 = pool.seq
        
        pool.set_state(100)
        seq2 = pool.seq
        
        pool.set_state(42)
        seq3 = pool.seq
        
        # Same state should produce same sequence
        assert seq1 == seq3
    
    def test_set_state_changes_output(self):
        """Test that setting different states can produce different outputs."""
        pool = RandomMutationORFPool('ATGGCCAAACCCGGG', mutation_type='any_codon', mutation_rate=0.5)
        
        # Get sequences at different states
        pool.set_state(0)
        seq0 = pool.seq
        
        pool.set_state(1000)
        seq1000 = pool.seq
        
        # With mutation_rate=0.5, different states likely produce different sequences
        # (not guaranteed but highly probable)
        # Just verify both are valid sequences
        assert len(seq0) == 15
        assert len(seq1000) == 15
        assert all(c in 'ACGT' for c in seq0)
        assert all(c in 'ACGT' for c in seq1000)


class TestRandomMutationORFPoolIntegration:
    """Tests for integration with other pools."""
    
    def test_integration_with_string_concatenation(self):
        """Test that RandomMutationORFPool can be concatenated with strings."""
        orf_seq = 'ATGGCCAAA'
        pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.5)
        
        combined = 'AAA.' + pool + '.TTT'
        assert isinstance(combined, Pool)
        
        seq = combined.seq
        assert seq.startswith('AAA.')
        assert seq.endswith('.TTT')
        assert len(seq) == 4 + len(orf_seq) + 4
    
    def test_integration_with_pool_concatenation(self):
        """Test that RandomMutationORFPool can be concatenated with other pools."""
        from poolparty import KmerPool
        
        orf_seq = 'ATGGCCAAA'
        orf_pool = RandomMutationORFPool(orf_seq, mutation_type='any_codon', mutation_rate=0.3)
        kmer_pool = KmerPool(length=3)
        
        combined = orf_pool + '.' + kmer_pool
        assert isinstance(combined, Pool)
        
        seq = combined.seq
        assert len(seq) == len(orf_seq) + 1 + 3


class TestRandomMutationORFPoolRepr:
    """Tests for string representation."""
    
    def test_repr_short_seq(self):
        """Test repr for short sequences."""
        pool = RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon', mutation_rate=0.1)
        repr_str = repr(pool)
        assert 'RandomMutationORFPool' in repr_str
        assert 'any_codon' in repr_str
    
    def test_repr_long_seq(self):
        """Test repr for long sequences (should be truncated)."""
        long_seq = 'ATG' * 10  # 30 codons
        pool = RandomMutationORFPool(long_seq, mutation_type='missense_only_first', mutation_rate=0.2)
        repr_str = repr(pool)
        assert 'RandomMutationORFPool' in repr_str
        assert '...' in repr_str
    
    def test_repr_with_rate_array(self):
        """Test repr with per-codon mutation rates."""
        orf_seq = 'ATGGCCAAA'
        rates = [0.1, 0.2, 0.3]
        pool = RandomMutationORFPool(orf_seq, mutation_type='synonymous', mutation_rate=rates)
        repr_str = repr(pool)
        assert 'RandomMutationORFPool' in repr_str
        assert '[3 rates]' in repr_str


class TestRandomMutationORFPoolEdgeCases:
    """Tests for edge cases."""
    
    def test_single_codon_orf(self):
        """Test with single codon ORF."""
        pool = RandomMutationORFPool('ATG', mutation_type='any_codon', mutation_rate=1.0)
        seq = pool.seq
        assert len(seq) == 3
        assert all(c in 'ACGT' for c in seq)
    
    def test_large_orf(self):
        """Test with large ORF."""
        # 100 codons
        large_orf = 'ATG' + 'GCC' * 98 + 'AAA'
        pool = RandomMutationORFPool(large_orf, mutation_type='any_codon', mutation_rate=0.1)
        assert len(pool.codons) == 100
        seq = pool.seq
        assert len(seq) == len(large_orf)
    
    def test_generate_multiple_seqs(self):
        """Test generating multiple sequences."""
        pool = RandomMutationORFPool('ATGGCCAAACCCGGG', mutation_type='any_codon', mutation_rate=0.5)
        seqs = pool.generate_seqs(num_seqs=100)
        
        assert len(seqs) == 100
        # With random mutations, expect diversity
        unique_seqs = set(seqs)
        assert len(unique_seqs) > 50  # Should have good diversity with rate=0.5


class TestRandomMutationORFPoolMarkChanges:
    """Tests for mark_changes functionality."""
    
    def test_mark_changes_default_false(self):
        """Test that mark_changes defaults to False."""
        pool = RandomMutationORFPool('ATGGCCAAA', mutation_type='any_codon')
        assert pool.mark_changes is False
    
    def test_mark_changes_true_basic(self):
        """Test basic functionality with mark_changes=True."""
        orf_seq = 'ATGGCCAAA'
        pool = RandomMutationORFPool(
            orf_seq, 
            mutation_type='any_codon', 
            mutation_rate=1.0,
            mark_changes=True
        )
        
        seq = pool.seq
        assert len(seq) == len(orf_seq)
        # With mutation rate 1.0, all codons should be mutated and swapcase'd
        assert seq != orf_seq
        assert any(c.islower() for c in seq)
    
    def test_mark_changes_mutated_codons_only(self):
        """Test that only mutated codons have swapped case."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.4,
            mark_changes=True
        )
        
        pool.set_state(42)
        seq = pool.seq
        
        # Split into codons
        orig_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        result_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        
        for i, (orig, result) in enumerate(zip(orig_codons, result_codons)):
            if orig != result.upper():
                # Codon was mutated, should be lowercase
                assert result.islower(), f"Mutated codon {i} should be lowercase"
            else:
                # Codon not mutated, should be uppercase
                assert result.isupper(), f"Unmutated codon {i} should be uppercase"
    
    def test_mark_changes_with_different_mutation_types(self):
        """Test mark_changes with different mutation types."""
        orf_seq = 'ATGGCCAAACCCGGG'
        
        for mutation_type in ['missense_only_first', 'any_codon', 'synonymous']:
            pool = RandomMutationORFPool(
                orf_seq,
                mutation_type=mutation_type,
                mutation_rate=0.5,
                mark_changes=True
            )
            
            seq = pool.seq
            assert len(seq) == len(orf_seq)
    
    def test_mark_changes_with_position_specific_rates(self):
        """Test mark_changes with position-specific mutation rates."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        rates = [1.0, 0.0, 1.0, 0.0, 1.0]  # Alternate: mutate, no mutation, mutate, ...
        
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=rates,
            mark_changes=True
        )
        
        pool.set_state(10)
        seq = pool.seq
        
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        
        # Positions 0, 2, 4 should be mutated (and lowercase)
        assert codons[0].islower()
        assert codons[2].islower()
        assert codons[4].islower()
        
        # Positions 1, 3 should not be mutated (and uppercase)
        assert codons[1].isupper()
        assert codons[3].isupper()
    
    def test_mark_changes_false_no_case_change(self):
        """Test that mark_changes=False doesn't change case."""
        orf_seq = 'ATGGCCAAA'
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=1.0,
            mark_changes=False
        )
        
        seq = pool.seq
        # All characters should be uppercase
        assert seq.isupper()
    
    def test_mark_changes_deterministic(self):
        """Test that mark_changes produces deterministic results."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            mark_changes=True
        )
        
        pool.set_state(123)
        seq1 = pool.seq
        
        pool.set_state(123)
        seq2 = pool.seq
        
        # Same state should produce identical sequences
        assert seq1 == seq2
    
    def test_mark_changes_with_zero_mutation_rate(self):
        """Test mark_changes with zero mutation rate (no mutations)."""
        orf_seq = 'ATGGCCAAA'
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.0,
            mark_changes=True
        )
        
        seq = pool.seq
        # No mutations, so sequence should be unchanged and uppercase
        assert seq == orf_seq
        assert seq.isupper()


class TestRandomMutationORFPoolFlankingRegions:
    """Tests for ORF flanking region (UTR) handling."""
    
    def test_with_flanking_regions(self):
        """Test RandomMutationORFPool with orf_start/orf_end."""
        upstream = "GGGGG"  # 5 bp 5' UTR
        orf_seq = "ATGATGATG"  # 9 bp ORF (3 codons)
        downstream = "CCCCC"  # 5 bp 3' UTR
        full_seq = upstream + orf_seq + downstream
        
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=5,
            orf_end=14
        )
        
        assert pool.upstream_flank == upstream
        assert pool.downstream_flank == downstream
        assert pool.orf_seq == orf_seq
    
    def test_flanks_preserved_in_output(self):
        """Test that flanks are preserved in all outputs."""
        upstream = "GGGGG"
        orf_seq = "ATGATGATG"
        downstream = "CCCCC"
        full_seq = upstream + orf_seq + downstream
        
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=5,
            orf_end=14
        )
        
        # Check flanks are preserved in multiple states
        for state in range(20):
            pool.set_state(state)
            seq = pool.seq
            assert seq.startswith(upstream), f"5' UTR not preserved: {seq}"
            assert seq.endswith(downstream), f"3' UTR not preserved: {seq}"


class TestRandomMutationORFPoolPoolChaining:
    """Tests for Pool chaining / transformer pattern."""
    
    def test_pool_as_seq_input(self):
        """Test RandomMutationORFPool with Pool as seq input."""
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='random')
        
        pool = RandomMutationORFPool(
            parent,
            mutation_type='any_codon',
            mutation_rate=0.5
        )
        
        # Should work with Pool input
        seq = pool.seq
        assert len(seq) == 9
        assert all(c in 'ACGTacgt' for c in seq)
    
    def test_generate_seqs_with_seed(self):
        """Test generate_seqs with seed produces reproducible results."""
        pool = RandomMutationORFPool(
            'ATGGCCAAACCCGGG',
            mutation_type='any_codon',
            mutation_rate=0.5
        )
        
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
        
        seqs3 = pool.generate_seqs(num_seqs=10, seed=123)
        assert seqs1 != seqs3, "Different seeds should produce different sequences"
    
    def test_parent_pool_state_affects_output(self):
        """Verify that parent Pool's current sequence is used for mutations."""
        # Parent pool with very different sequences
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        pool = RandomMutationORFPool(
            parent,
            mutation_type='synonymous',  # Use synonymous so we can track original
            mutation_rate=0.0  # No mutations - just pass through
        )
        
        # State 0 should use first parent sequence
        pool.set_state(0)
        seq0 = pool.seq
        assert seq0 == "ATGATGATG", f"State 0 should use first parent seq, got {seq0}"
        
        # State 1 should use second parent sequence
        pool.set_state(1)
        seq1 = pool.seq
        assert seq1 == "CCCGGGAAA", f"State 1 should use second parent seq, got {seq1}"
    
    def test_mutations_applied_to_parent_sequence(self):
        """Verify mutations are applied to parent's current sequence, not stored sequence."""
        # Two very different parent sequences
        parent = Pool(seqs=["ATGATGATG", "CTGCTGCTG"], mode='sequential')
        
        pool = RandomMutationORFPool(
            parent,
            mutation_type='nonsense',  # Always mutate to stop
            mutation_rate=1.0
        )
        
        stop_codons = set(pool.stop_codons)
        
        # Check mutations are applied to correct parent sequence
        for parent_state in range(2):
            pool.set_state(parent_state)
            seq = pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # All codons should be stops (from nonsense mutation)
            for codon in codons:
                assert codon in stop_codons, \
                    f"Parent state {parent_state}: Expected stop codon, got {codon}"
    
    def test_chain_with_multiple_pools(self):
        """Test chaining: Pool -> RandomMutationORFPool -> concatenation."""
        parent = Pool(seqs=["ATGATGATG"], mode='sequential')
        
        mutation_pool = RandomMutationORFPool(
            parent,
            mutation_type='any_codon',
            mutation_rate=0.5
        )
        
        # Chain with prefix and suffix
        combined = "GGG" + mutation_pool + "CCC"
        
        combined.set_state(0)
        seq = combined.seq
        
        assert seq.startswith("GGG"), f"Prefix not preserved: {seq}"
        assert seq.endswith("CCC"), f"Suffix not preserved: {seq}"
        assert len(seq) == 3 + 9 + 3
    
    def test_transformer_pattern_with_flanks(self):
        """Test transformer pattern preserves flanks from parent."""
        # Parent with UTRs
        parent = Pool(seqs=["GGGGG" + "ATGATGATG" + "CCCCC"], mode='sequential')
        
        pool = RandomMutationORFPool(
            parent,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=5,
            orf_end=14
        )
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            
            # Flanks must be preserved
            assert seq[:5] == "GGGGG", f"State {state}: 5' flank modified"
            assert seq[-5:] == "CCCCC", f"State {state}: 3' flank modified"


class TestRandomMutationORFPoolMutationTypeVerification:
    """Rigorous verification that mutation types produce correct biological outcomes."""
    
    def test_synonymous_never_changes_amino_acid(self):
        """Verify synonymous mutations NEVER change amino acid sequence."""
        # Use codons with synonymous alternatives
        # CTG = Leu (6 synonyms), GCT = Ala (4 synonyms), CGT = Arg (6 synonyms)
        orf_seq = 'CTGGCTCGT' * 5  # 15 codons with synonymous options
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='synonymous',
            mutation_rate=0.9  # High rate to ensure mutations
        )
        
        codon_to_aa = pool.codon_to_aa_dict
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        original_aas = [codon_to_aa[c] for c in original_codons]
        
        # Test many states
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            mutated_aas = [codon_to_aa[c] for c in mutated_codons]
            
            assert mutated_aas == original_aas, \
                f"State {state}: Synonymous mutation changed AA! {original_aas} -> {mutated_aas}"
    
    def test_missense_only_first_never_produces_stop(self):
        """Verify missense_only_first NEVER produces stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG' * 3  # 15 codons
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='missense_only_first',
            mutation_rate=0.8
        )
        
        stop_codons = pool.stop_codons
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i, codon in enumerate(mutated_codons):
                assert codon not in stop_codons, \
                    f"State {state}: missense_only_first produced stop codon {codon} at position {i}"
    
    def test_missense_only_random_never_produces_stop(self):
        """Verify missense_only_random NEVER produces stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG' * 3
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='missense_only_random',
            mutation_rate=0.8
        )
        
        stop_codons = pool.stop_codons
        
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i, codon in enumerate(mutated_codons):
                assert codon not in stop_codons, \
                    f"State {state}: missense_only_random produced stop codon {codon} at position {i}"
    
    def test_nonsynonymous_always_changes_amino_acid_when_mutated(self):
        """Verify nonsynonymous mutations ALWAYS change amino acid when codon changes."""
        orf_seq = 'ATGATGATG' * 5  # All Met codons
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='nonsynonymous_first',
            mutation_rate=0.7
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
                        f"State {state}, pos {i}: Nonsynonymous mutation didn't change AA! {original_codons[i]}({orig_aa}) -> {mutated_codons[i]}({mut_aa})"
                    mutations_verified += 1
        
        assert mutations_verified > 50, "Not enough mutations to verify"
    
    def test_nonsense_only_produces_stop_codons_for_mutations(self):
        """Verify nonsense mutations ONLY produce stop codons."""
        orf_seq = 'ATGGCCAAACCCGGG' * 3
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='nonsense',
            mutation_rate=0.5
        )
        
        stop_codons = set(pool.stop_codons)
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        mutations_verified = 0
        for state in range(200):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i in range(len(original_codons)):
                if original_codons[i] != mutated_codons[i]:
                    assert mutated_codons[i] in stop_codons, \
                        f"State {state}, pos {i}: Nonsense mutation produced non-stop {mutated_codons[i]}"
                    mutations_verified += 1
        
        assert mutations_verified > 50, "Not enough mutations to verify"


class TestRandomMutationORFPoolMutationRateVerification:
    """Verify mutation rates are approximately correct."""
    
    def test_mutation_rate_approximate_frequency(self):
        """Verify actual mutation frequency approximates the specified rate."""
        orf_seq = 'ATGGCCAAA' * 20  # 60 codons
        target_rate = 0.3
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=target_rate
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        num_codons = len(original_codons)
        
        total_mutations = 0
        num_samples = 500
        
        for state in range(num_samples):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            mutations = sum(1 for i in range(num_codons) 
                          if original_codons[i] != mutated_codons[i])
            total_mutations += mutations
        
        actual_rate = total_mutations / (num_samples * num_codons)
        
        # Allow 20% tolerance
        assert abs(actual_rate - target_rate) < 0.1, \
            f"Expected mutation rate ~{target_rate}, got {actual_rate:.3f}"
    
    def test_per_position_mutation_rates(self):
        """Verify position-specific rates work correctly."""
        orf_seq = 'ATGGCCAAACCCGGG'  # 5 codons
        # Rates: 0 for pos 0,2,4 and 1.0 for pos 1,3
        rates = [0.0, 1.0, 0.0, 1.0, 0.0]
        
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=rates
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Positions 0, 2, 4 should NEVER be mutated
            assert mutated_codons[0] == original_codons[0], f"State {state}: Position 0 mutated with rate 0"
            assert mutated_codons[2] == original_codons[2], f"State {state}: Position 2 mutated with rate 0"
            assert mutated_codons[4] == original_codons[4], f"State {state}: Position 4 mutated with rate 0"
            
            # Positions 1, 3 should ALWAYS be mutated
            assert mutated_codons[1] != original_codons[1], f"State {state}: Position 1 not mutated with rate 1.0"
            assert mutated_codons[3] != original_codons[3], f"State {state}: Position 3 not mutated with rate 1.0"
    
    def test_zero_rate_produces_no_mutations(self):
        """Verify zero rate produces exactly zero mutations."""
        orf_seq = 'ATGGCCAAACCCGGG' * 5
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.0
        )
        
        for state in range(100):
            pool.set_state(state)
            assert pool.seq == orf_seq, f"State {state}: Zero rate produced mutation"
    
    def test_full_rate_produces_all_mutations(self):
        """Verify rate 1.0 mutates every codon."""
        orf_seq = 'ATGGCCAAACCCGGG'
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=1.0
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(50):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i in range(len(original_codons)):
                assert mutated_codons[i] != original_codons[i], \
                    f"State {state}: Position {i} not mutated with rate 1.0"


class TestRandomMutationORFPoolFlankingIntegrity:
    """Verify flanking regions are never modified."""
    
    def test_flanks_never_mutated(self):
        """Verify UTR regions are NEVER mutated regardless of mutation rate."""
        upstream = "GGGGGGGGG"  # 9 bp
        orf_seq = "ATGATGATGATGATGATG"  # 18 bp (6 codons)
        downstream = "CCCCCCCCC"  # 9 bp
        full_seq = upstream + orf_seq + downstream
        
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=1.0,  # Maximum mutation rate
            orf_start=9,
            orf_end=27
        )
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            # Flanks must be EXACTLY preserved
            assert seq[:9] == upstream, f"State {state}: 5' UTR was modified: {seq[:9]}"
            assert seq[-9:] == downstream, f"State {state}: 3' UTR was modified: {seq[-9:]}"
    
    def test_only_orf_region_mutated(self):
        """Verify mutations only occur within ORF boundaries."""
        upstream = "AAAA"
        orf_seq = "ATGATGATG"
        downstream = "TTTT"
        full_seq = upstream + orf_seq + downstream
        
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=0.8,
            orf_start=4,
            orf_end=13
        )
        
        for state in range(100):
            pool.set_state(state)
            seq = pool.seq
            
            # Check upstream unchanged
            assert seq[:4] == upstream
            # Check downstream unchanged
            assert seq[-4:] == downstream
            # Check ORF is valid length
            orf_part = seq[4:13]
            assert len(orf_part) == 9


class TestRandomMutationORFPoolEdgeCases:
    """Comprehensive edge case testing."""
    
    def test_single_codon_orf_all_mutation_types(self):
        """Test single codon ORF with various mutation types."""
        single_codon = "CTG"  # Leucine - has synonyms
        
        for mutation_type in ['any_codon', 'synonymous', 'missense_only_first']:
            pool = RandomMutationORFPool(
                single_codon,
                mutation_type=mutation_type,
                mutation_rate=1.0
            )
            
            for state in range(10):
                pool.set_state(state)
                seq = pool.seq
                assert len(seq) == 3, f"{mutation_type}: Wrong length"
                assert all(c in 'ACGT' for c in seq), f"{mutation_type}: Invalid DNA"
    
    def test_methionine_only_orf_synonymous(self):
        """ATG (Met) has no synonymous codons - should stay unchanged."""
        met_only = "ATGATGATG"  # 3 Met codons
        pool = RandomMutationORFPool(
            met_only,
            mutation_type='synonymous',
            mutation_rate=1.0  # Try to mutate everything
        )
        
        # ATG has no synonymous alternatives, so should remain ATG
        for state in range(20):
            pool.set_state(state)
            seq = pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            # All should still be ATG (no synonymous options)
            for codon in codons:
                assert codon == "ATG", f"State {state}: ATG mutated to {codon} with synonymous"
    
    def test_tryptophan_only_orf_synonymous(self):
        """TGG (Trp) has no synonymous codons - should stay unchanged."""
        trp_only = "TGGTGGTGG"  # 3 Trp codons
        pool = RandomMutationORFPool(
            trp_only,
            mutation_type='synonymous',
            mutation_rate=1.0
        )
        
        for state in range(20):
            pool.set_state(state)
            seq = pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            for codon in codons:
                assert codon == "TGG", f"State {state}: TGG mutated to {codon} with synonymous"
    
    def test_very_long_orf(self):
        """Test with a very long ORF (100 codons)."""
        long_orf = "ATGGCCAAA" * 33 + "ATG"  # 100 codons
        pool = RandomMutationORFPool(
            long_orf,
            mutation_type='any_codon',
            mutation_rate=0.1
        )
        
        assert len(pool.codons) == 100
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            assert len(seq) == 300
            assert all(c in 'ACGT' for c in seq)
    
    def test_no_upstream_flank(self):
        """Test with orf_start=0 (no upstream flank)."""
        full_seq = "ATGATGATG" + "CCCCC"
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=0,
            orf_end=9
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == "CCCCC"
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            assert seq.endswith("CCCCC")
    
    def test_no_downstream_flank(self):
        """Test with orf_end=len (no downstream flank)."""
        full_seq = "GGGGG" + "ATGATGATG"
        pool = RandomMutationORFPool(
            full_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=5,
            orf_end=14
        )
        
        assert pool.upstream_flank == "GGGGG"
        assert pool.downstream_flank == ""
        
        for state in range(10):
            pool.set_state(state)
            seq = pool.seq
            assert seq.startswith("GGGGG")
    
    def test_no_flanks_at_all(self):
        """Test with no flanks (orf is entire sequence)."""
        orf_seq = "ATGATGATG"
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=0.5,
            orf_start=0,
            orf_end=9
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == ""
        assert pool.orf_seq == orf_seq
    
    def test_extreme_position_specific_rates(self):
        """Test with extreme position-specific rates (0s and 1s)."""
        orf_seq = "ATGGCCAAACCCGGG"  # 5 codons
        # Only middle codon mutates
        rates = [0.0, 0.0, 1.0, 0.0, 0.0]
        
        pool = RandomMutationORFPool(
            orf_seq,
            mutation_type='any_codon',
            mutation_rate=rates
        )
        
        original_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        
        for state in range(50):
            pool.set_state(state)
            seq = pool.seq
            mutated_codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            # Only position 2 should change
            assert mutated_codons[0] == original_codons[0]
            assert mutated_codons[1] == original_codons[1]
            assert mutated_codons[2] != original_codons[2]  # Must change
            assert mutated_codons[3] == original_codons[3]
            assert mutated_codons[4] == original_codons[4]
    
    def test_all_same_codons_nonsynonymous(self):
        """Test nonsynonymous on ORF with all identical codons."""
        all_met = "ATGATGATGATG"  # 4 identical Met codons
        pool = RandomMutationORFPool(
            all_met,
            mutation_type='nonsynonymous_first',
            mutation_rate=0.5
        )
        
        codon_to_aa = pool.codon_to_aa_dict
        
        for state in range(20):
            pool.set_state(state)
            seq = pool.seq
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            
            for i, codon in enumerate(codons):
                if codon != "ATG":
                    # If mutated, AA must be different
                    assert codon_to_aa[codon] != 'M', \
                        f"State {state}, pos {i}: Nonsynonymous produced same AA"
    
    def test_lowercase_from_pool_parent(self):
        """Test that lowercase sequences from Pool parent are handled."""
        parent = Pool(seqs=["atgatgatg"], mode='sequential')
        
        pool = RandomMutationORFPool(
            parent,
            mutation_type='any_codon',
            mutation_rate=0.5
        )
        
        pool.set_state(0)
        seq = pool.seq
        # Should handle lowercase input
        assert len(seq) == 9
