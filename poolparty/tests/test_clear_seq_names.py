"""Tests for Pool.clear_seq_names() functionality."""
import pytest
import pandas as pd
import poolparty as pp


class TestClearSeqNamesBasic:
    """Test basic clear_seq_names() behavior."""
    
    def test_clear_seq_names_returns_self(self):
        """Test that clear_seq_names() returns self for chaining."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'], seq_names=['first', 'second'])
            result = pool.clear_seq_names()
            assert result is pool
    
    def test_clear_seq_names_blocks_explicit_names(self):
        """Test that clear_seq_names() blocks explicit seq_names."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ['AAA', 'TTT'],
                seq_names=['first', 'second'],
                mode='sequential',
            ).named('pool').clear_seq_names()
        
        df = pool.generate_library(num_cycles=1)
        assert 'name' not in df.columns or df['name'].isna().all()
    
    def test_clear_seq_names_blocks_prefix(self):
        """Test that clear_seq_names() blocks prefix."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ['AAA', 'TTT'],
                prefix='seq_',
                mode='sequential',
            ).named('pool').clear_seq_names()
        
        df = pool.generate_library(num_cycles=1)
        assert 'name' not in df.columns or df['name'].isna().all()
    
    def test_clear_seq_names_sets_block_flag(self):
        """Test that clear_seq_names() sets the _block_seq_names flag."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation._block_seq_names is False
            pool.clear_seq_names()
            assert pool.operation._block_seq_names is True


class TestClearSeqNamesWithParentNames:
    """Test clear_seq_names() blocks parent name propagation."""
    
    def test_blocks_parent_names_in_mutagenize(self):
        """Test that clear_seq_names() blocks parent names from propagating."""
        with pp.Party() as party:
            parent = pp.from_seqs(
                ['ACGT'],
                seq_names=['parent_seq'],
                mode='sequential',
            )
            mutants = pp.mutagenize(parent, num_mutations=1, mode='sequential').named('mut')
            mutants.clear_seq_names()
        
        df = mutants.generate_library(num_cycles=1)
        # Names should be None (column absent or all NaN)
        assert 'name' not in df.columns or df['name'].isna().all()
    
    def test_parent_names_propagate_without_clear(self):
        """Test that parent names propagate without clear_seq_names()."""
        with pp.Party() as party:
            parent = pp.from_seqs(
                ['ACGT'],
                seq_names=['parent_seq'],
                mode='sequential',
            )
            mutants = pp.mutagenize(parent, num_mutations=1, mode='sequential').named('mut')
        
        df = mutants.generate_library(num_cycles=1)
        # Names should contain parent name
        assert 'name' in df.columns
        assert df['name'].notna().any()
        assert 'parent_seq' in df['name'].iloc[0]


class TestClearSeqNamesWithStack:
    """Test clear_seq_names() with stacked pools."""
    
    def test_clear_on_stacked_pool(self):
        """Test clear_seq_names() on a stacked pool."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'], seq_names=['name_a'], mode='sequential')
            b = pp.from_seqs(['TTT'], seq_names=['name_b'], mode='sequential')
            stacked = (a + b).named('stacked').clear_seq_names()
        
        df = stacked.generate_library(num_cycles=1)
        assert 'name' not in df.columns or df['name'].isna().all()
    
    def test_stacked_pool_names_without_clear(self):
        """Test that stacked pool propagates parent names without clear."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'], seq_names=['name_a'], mode='sequential')
            b = pp.from_seqs(['TTT'], seq_names=['name_b'], mode='sequential')
            stacked = (a + b).named('stacked')
        
        df = stacked.generate_library(num_cycles=1)
        assert 'name' in df.columns
        names = list(df['name'])
        assert 'name_a' in names
        assert 'name_b' in names


class TestClearSeqNamesChaining:
    """Test clear_seq_names() method chaining."""
    
    def test_chaining_with_named(self):
        """Test chaining clear_seq_names() with named()."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ['AAA'],
                seq_names=['test'],
                mode='sequential',
            ).named('pool').clear_seq_names()
            assert pool.name == 'pool'
            assert pool.operation._block_seq_names is True
    
    def test_clear_then_operations(self):
        """Test that clear_seq_names() doesn't affect downstream operations."""
        with pp.Party() as party:
            parent = pp.from_seqs(['ACGT'], seq_names=['src'], mode='sequential')
            parent.clear_seq_names()
            mutants = pp.mutagenize(parent, num_mutations=1, mode='sequential').named('mut')
        
        df = mutants.generate_library(num_cycles=1)
        # Parent cleared names, so mutants shouldn't have propagated names
        assert 'name' not in df.columns or df['name'].isna().all()



class TestClearSeqNamesDoesNotAffectSequences:
    """Test that clear_seq_names() only affects names, not sequences."""
    
    def test_sequences_unchanged(self):
        """Test that sequences are not affected by clear_seq_names()."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ['AAA', 'TTT', 'GGG'],
                seq_names=['a', 'b', 'c'],
                mode='sequential',
            ).named('pool').clear_seq_names()
        
        df = pool.generate_library(num_cycles=1)
        assert list(df['seq']) == ['AAA', 'TTT', 'GGG']
    
    def test_design_cards_unchanged(self):
        """Test that design cards are not affected by clear_seq_names()."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ['AAA', 'TTT'],
                seq_names=['a', 'b'],
                mode='sequential',
            ).named('pool').clear_seq_names()
        
        df = pool.generate_library(num_cycles=1, report_design_cards=True)
        # Design card columns should still be present (uses pool.op naming)
        key_cols = [c for c in df.columns if '.key.seq_index' in c]
        assert len(key_cols) == 1
        assert list(df[key_cols[0]]) == [0, 1]
