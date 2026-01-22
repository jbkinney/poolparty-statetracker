"""Tests for the MutagenizeOrf operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.orf_ops.mutagenize_orf import MutagenizeOrfOp, mutagenize_orf
from poolparty.codon_table import CodonTable
from poolparty.codon_table import STANDARD_GENETIC_CODE


class TestCodonTable:
    """Test CodonTable class."""
    
    def test_standard_codon_table(self):
        """Test standard codon table initialization."""
        ct = CodonTable('standard')
        assert len(ct.all_codons) == 64
        assert len(ct.stop_codons) == 3
        assert ct.codon_to_aa['ATG'] == 'M'
        assert 'ATG' in ct.aa_to_codons['M']
    
    def test_mutation_lookup_exists(self):
        """Test that mutation lookup contains all types."""
        ct = CodonTable('standard')
        expected_types = [
            'any_codon', 'nonsynonymous_first', 'nonsynonymous_random',
            'missense_only_first', 'missense_only_random', 'synonymous', 'nonsense'
        ]
        for mt in expected_types:
            assert mt in ct.mutation_lookup
    
    def test_any_codon_mutations(self):
        """Test any_codon returns 63 alternatives."""
        ct = CodonTable('standard')
        for codon in ct.all_codons:
            alts = ct.get_mutations(codon, 'any_codon')
            assert len(alts) == 63
            assert codon not in alts
    
    def test_synonymous_mutations(self):
        """Test synonymous mutations are correct."""
        ct = CodonTable('standard')
        # Methionine has only one codon - no synonymous mutations
        assert len(ct.get_mutations('ATG', 'synonymous')) == 0
        # Leucine has 6 codons - 5 synonymous mutations
        assert len(ct.get_mutations('CTG', 'synonymous')) == 5
    
    def test_nonsense_mutations(self):
        """Test nonsense mutations return stop codons."""
        ct = CodonTable('standard')
        # Non-stop codon should get 3 stop options
        assert len(ct.get_mutations('ATG', 'nonsense')) == 3
        # Stop codon should get no options
        assert len(ct.get_mutations('TAA', 'nonsense')) == 0
    
    def test_is_uniform(self):
        """Test uniformity detection."""
        ct = CodonTable('standard')
        # Truly uniform across all 64 codons
        assert ct.is_uniform('any_codon') is True
        assert ct.is_uniform('nonsynonymous_first') is True
        # Non-uniform
        assert ct.is_uniform('synonymous') is False
        assert ct.is_uniform('nonsynonymous_random') is False
        assert ct.is_uniform('missense_only_random') is False
        # missense_only_first and nonsense are uniform for non-stop codons only
        # (tested via UNIFORM_MUTATION_TYPES dict in mutagenize_orf.py)
    
    def test_custom_codon_table(self):
        """Test custom codon table."""
        custom = {"A": ["GCT", "GCC"], "M": ["ATG"]}
        ct = CodonTable(custom)
        assert len(ct.all_codons) == 3
        assert ct.codon_to_aa['GCT'] == 'A'


class TestMutagenizeOrfFactory:
    """Test mutagenize_orf factory function."""
    
    def test_returns_pool(self):
        """mutagenize_orf returns a Pool object."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTT', num_mutations=1)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_mutagenize_orf_op(self):
        """Pool's operation is MutagenizeOrfOp."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTT', num_mutations=1)
            assert isinstance(pool.operation, MutagenizeOrfOp)
    
    def test_accepts_string_input(self):
        """Factory accepts a string with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTT', num_mutations=1).named('mutant')
        
        df = pool.generate_library(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 9
    
    def test_accepts_pool_input(self):
        """Factory accepts an existing Pool as input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ATGAAATTT'])
            pool = mutagenize_orf(seq, num_mutations=1).named('mutant')
        
        df = pool.generate_library(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 9


class TestMutagenizeOrfParameterValidation:
    """Test parameter validation."""
    
    def test_requires_num_or_rate(self):
        """Must provide either num_mutations or mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Either num_mutations or mutation_rate must be provided"):
                mutagenize_orf('ATGAAATTT')
    
    def test_exclusive_num_and_rate(self):
        """Cannot provide both num_mutations and mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Only one of num_mutations or mutation_rate"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, mutation_rate=0.1)
    
    def test_num_mutations_minimum(self):
        """num_mutations must be >= 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize_orf('ATGAAATTT', num_mutations=0)
    
    def test_mutation_rate_range(self):
        """mutation_rate must be between 0 and 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_orf('ATGAAATTT', mutation_rate=-0.1)
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_orf('ATGAAATTT', mutation_rate=1.5)
    
    def test_invalid_mutation_type(self):
        """Invalid mutation_type raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_type must be one of"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, mutation_type='invalid')
    
    def test_sequential_mode_requires_num_mutations(self):
        """mode='sequential' not allowed with mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mode='sequential' is not supported with mutation_rate"):
                mutagenize_orf('ATGAAATTT', mutation_rate=0.1, mode='sequential')
    
    def test_sequential_mode_requires_uniform_type(self):
        """mode='sequential' requires uniform mutation type."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mode='sequential' requires a uniform mutation type"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, mutation_type='synonymous', mode='sequential')
    
    def test_orf_length_must_be_divisible_by_3(self):
        """ORF length must be divisible by 3."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="ORF region length must be divisible by 3"):
                mutagenize_orf('ATGAA', num_mutations=1)  # 5 bp, not divisible by 3
    
    def test_num_mutations_exceeds_eligible(self):
        """Error when num_mutations > number of eligible positions."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations.*exceeds.*eligible"):
                mutagenize_orf('ATGAAA', num_mutations=3)  # Only 2 codons


class TestMutagenizeOrfORFBoundaries:
    """Test ORF boundary handling."""
    
    def test_orf_extent(self):
        """Test orf_extent parameter."""
        # Sequence with 5' UTR (GGG) + ORF (ATGAAA) + 3' UTR (CCC)
        seq = 'GGGATGAAACCC'
        with pp.Party() as party:
            pool = mutagenize_orf(seq, num_mutations=1, orf_extent=(3, 9)).named('mutant')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df['seq']:
            # UTRs should be preserved
            assert mutant[:3] == 'GGG'
            assert mutant[-3:] == 'CCC'
            # Total length preserved
            assert len(mutant) == 12
    
    def test_orf_extent_validation(self):
        """Test orf_extent validation."""
        with pp.Party() as party:
            # orf_extent start out of range
            with pytest.raises(ValueError, match="orf_extent start must be >= 0"):
                mutagenize_orf('ATGAAA', num_mutations=1, orf_extent=(-1, 6))
            
            # orf_extent end exceeds length
            with pytest.raises(ValueError, match="orf_extent end.*cannot exceed sequence length"):
                mutagenize_orf('ATGAAA', num_mutations=1, orf_extent=(0, 10))
            
            # orf_extent start >= end
            with pytest.raises(ValueError, match="orf_extent start.*must be < end"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, orf_extent=(6, 3))
            
            # orf_extent must have exactly 2 elements
            with pytest.raises(ValueError, match="orf_extent must have exactly 2 elements"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, orf_extent=(0, 3, 6))


class TestMutagenizeOrfCodonPositions:
    """Test codon position selection."""
    
    def test_codon_positions_explicit(self):
        """Test explicit codon_positions parameter."""
        # 4 codons: ATG AAA TTT GGG
        seq = 'ATGAAATTTGGG'
        with pp.Party() as party:
            # Only allow mutations at codons 1 and 2 (AAA and TTT)
            pool = mutagenize_orf(
                seq, num_mutations=1, codon_positions=[1, 2], mode='sequential'
            ).named('mutant')
        
        df = pool.generate_library(num_cycles=1)
        # Should only mutate at positions 1 and 2
        for mutant in df['seq']:
            # First codon (ATG) should be unchanged
            assert mutant[:3] == 'ATG'
            # Last codon (GGG) should be unchanged
            assert mutant[-3:] == 'GGG'
    
    def test_codon_positions_slice(self):
        """Test codon_positions with slice parameter."""
        # 6 codons
        seq = 'ATGAAATTTGGGCCCAAA'
        with pp.Party() as party:
            # Only codons 0, 2, 4 (using slice with step=2)
            pool = mutagenize_orf(
                seq, num_mutations=1, codon_positions=slice(0, 6, 2)
            ).named('mutant')
            # Should have 3 eligible positions
            assert pool.operation.num_eligible == 3
            assert pool.operation.eligible_positions == [0, 2, 4]
    
    def test_codon_positions_validation(self):
        """Test codon position validation."""
        with pp.Party() as party:
            # Position out of range
            with pytest.raises(ValueError, match="codon_positions value.*is out of range"):
                mutagenize_orf('ATGAAA', num_mutations=1, codon_positions=[5])
            
            # Duplicate positions
            with pytest.raises(ValueError, match="must not contain duplicates"):
                mutagenize_orf('ATGAAATTT', num_mutations=1, codon_positions=[0, 0, 1])


class TestMutagenizeOrfMutationTypes:
    """Test different mutation types."""
    
    def test_any_codon_type(self):
        """Test any_codon mutation type."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAA', num_mutations=1, mutation_type='any_codon', mode='sequential'
            ).named('mutant')
        
        # 2 codons * 63 alternatives = 126 states
        assert pool.operation.num_values == 126
    
    def test_missense_only_first_type(self):
        """Test missense_only_first mutation type (default)."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAA', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_library(num_cycles=1, report_design_cards=True)
        ct = CodonTable('standard')
        
        for _, row in df.iterrows():
            wt_aas = row['mutate.key.wt_aas']
            mut_aas = row['mutate.key.mut_aas']
            for wt_aa, mut_aa in zip(wt_aas, mut_aas):
                # Should be different AA
                assert wt_aa != mut_aa
                # Should not be stop
                assert mut_aa != '*'
    
    def test_nonsense_type(self):
        """Test nonsense mutation type."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAA', num_mutations=1, mutation_type='nonsense', mode='sequential', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_cycles=1, report_design_cards=True)
        
        for _, row in df.iterrows():
            mut_aas = row['mutate.key.mut_aas']
            for mut_aa in mut_aas:
                # All mutations should be stops
                assert mut_aa == '*'
    
    def test_synonymous_type(self):
        """Test synonymous mutation type (random mode only)."""
        # Use a codon with synonymous options (Leucine CTG has 5 alternatives)
        with pp.Party() as party:
            pool = mutagenize_orf(
                'CTGCTG', num_mutations=1, mutation_type='synonymous', mode='random', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=20, seed=42, report_design_cards=True)
        ct = CodonTable('standard')
        
        for _, row in df.iterrows():
            wt_aas = row['mutate.key.wt_aas']
            mut_aas = row['mutate.key.mut_aas']
            for wt_aa, mut_aa in zip(wt_aas, mut_aas):
                # AA should be the same (synonymous)
                assert wt_aa == mut_aa


class TestMutagenizeOrfSequentialMode:
    """Test sequential mode enumeration."""
    
    def test_sequential_single_mutation_count(self):
        """Test correct number of single mutants in sequential mode."""
        with pp.Party() as party:
            # 3 codons, missense_only_first has 19 alternatives (20 AAs - 1 current - stop)
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='sequential'
            ).named('mutant')
        
        # 3 positions * 19 alternatives = 57
        assert pool.operation.num_values == 57
        
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 57
    
    def test_sequential_double_mutation_count(self):
        """Test correct number of double mutants in sequential mode."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=2, mode='sequential'
            ).named('mutant')
        
        # C(3,2) * 19^2 = 3 * 361 = 1083
        assert pool.operation.num_values == 1083
    
    def test_sequential_mutations_correctness(self):
        """Test that sequential mutations are applied correctly."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='sequential', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_cycles=1, report_design_cards=True)
        
        for _, row in df.iterrows():
            mutant = row['seq']
            positions = row['mutate.key.codon_positions']
            wt_codons = row['mutate.key.wt_codons']
            mut_codons = row['mutate.key.mut_codons']
            
            # Verify mutations are applied
            for pos, wt, mut in zip(positions, wt_codons, mut_codons):
                # Get codon from mutant sequence
                codon_start = pos * 3
                mutant_codon = mutant[codon_start:codon_start + 3]
                assert mutant_codon == mut
                assert wt != mut  # Should be different


class TestMutagenizeOrfRandomMode:
    """Test random mode."""
    
    def test_random_mode_with_num_mutations(self):
        """Test random mode with fixed num_mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTTGGG', num_mutations=2, mode='random', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=50, seed=42, report_design_cards=True)
        
        for _, row in df.iterrows():
            positions = row['mutate.key.codon_positions']
            # Should have exactly 2 mutations
            assert len(positions) == 2
    
    def test_random_mode_with_mutation_rate(self):
        """Test random mode with mutation_rate."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTTGGGCCCAAA', mutation_rate=0.5, mode='random', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=100, seed=42, report_design_cards=True)
        
        # Should have variable number of mutations
        num_mutations_list = [len(row['mutate.key.codon_positions']) for _, row in df.iterrows()]
        # With 6 codons and 50% rate, should see some variation
        assert len(set(num_mutations_list)) > 1
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTTGGG', num_mutations=1, mode='random'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=50, seed=42)
        unique_mutants = df['seq'].nunique()
        assert unique_mutants > 5  # Should have variety


class TestMutagenizeOrfHybridMode:
    """Test hybrid mode."""
    
    def test_random_uses_num_states(self):
        """Random mode with num_states sets num_states correctly."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='random', num_states=100
            )
            assert pool.operation.num_values == 100
    
    def test_random_generates_correct_count(self):
        """Random mode with num_states generates num_states sequences per iteration."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='random', num_states=25
            ).named('mutant')
        
        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) == 25


class TestMutagenizeOrfDesignCards:
    """Test design card output."""
    
    def test_design_card_columns(self):
        """Design card contains expected columns."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='sequential', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=4, report_design_cards=True)
        
        assert 'mutate.key.codon_positions' in df.columns
        assert 'mutate.key.wt_codons' in df.columns
        assert 'mutate.key.mut_codons' in df.columns
        assert 'mutate.key.wt_aas' in df.columns
        assert 'mutate.key.mut_aas' in df.columns
    
    def test_design_card_consistency(self):
        """Design card values match actual mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                'ATGAAATTT', num_mutations=1, mode='sequential', op_name='mutate'
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=20, report_design_cards=True)
        ct = CodonTable('standard')
        
        for _, row in df.iterrows():
            mutant = row['seq']
            positions = row['mutate.key.codon_positions']
            wt_codons = row['mutate.key.wt_codons']
            mut_codons = row['mutate.key.mut_codons']
            wt_aas = row['mutate.key.wt_aas']
            mut_aas = row['mutate.key.mut_aas']
            
            for pos, wt_c, mut_c, wt_aa, mut_aa in zip(
                positions, wt_codons, mut_codons, wt_aas, mut_aas
            ):
                # Check codon in mutant sequence
                codon_start = pos * 3
                actual_codon = mutant[codon_start:codon_start + 3]
                assert actual_codon == mut_c
                
                # Check AA translations
                assert ct.codon_to_aa.get(wt_c.upper()) == wt_aa
                assert ct.codon_to_aa.get(mut_c.upper()) == mut_aa


class TestMutagenizeOrfPreservesLength:
    """Test that mutations preserve sequence length."""
    
    def test_length_preserved(self):
        """Mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTTGGG', num_mutations=2).named('mutant')
        
        df = pool.generate_library(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert len(mutant) == 12
    
    def test_length_preserved_with_flanks(self):
        """Mutations preserve length with ORF boundaries."""
        seq = 'GGGATGAAACCC'  # 3bp UTR + 6bp ORF + 3bp UTR
        with pp.Party() as party:
            pool = mutagenize_orf(
                seq, num_mutations=1, orf_extent=(3, 9)
            ).named('mutant')
        
        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df['seq']:
            assert len(mutant) == 12


class TestMutagenizeOrfCustomName:
    """Test name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTT', num_mutations=1)
            assert pool.operation.name.startswith('op[')
            assert ':mutagenize_orf' in pool.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = mutagenize_orf('ATGAAATTT', num_mutations=1, op_name='my_orf_mutations')
            assert pool.operation.name == 'my_orf_mutations'
