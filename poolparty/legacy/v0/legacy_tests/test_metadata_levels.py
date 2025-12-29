"""
Comprehensive tests for metadata level control in design cards.

Tests the three metadata levels:
- 'core': index, abs_start, abs_end only
- 'features': core + pool-specific fields (default)
- 'complete': features + value
"""

import pytest
from poolparty import Pool
from poolparty.kmer_pool import KmerPool
from poolparty.barcode_pool import BarcodePool
from poolparty.iupac_pool import IUPACPool
from poolparty.mixed_pool import MixedPool
from poolparty.insertion_scan_pool import InsertionScanPool
from poolparty.deletion_scan_pool import DeletionScanPool
from poolparty.subseq_pool import SubseqPool
from poolparty.shuffle_scan_pool import ShuffleScanPool
from poolparty.k_mutation_pool import KMutationPool
from poolparty.random_mutation_pool import RandomMutationPool
from poolparty.insertion_scan_orf_pool import InsertionScanORFPool
from poolparty.deletion_scan_orf_pool import DeletionScanORFPool
from poolparty.k_mutation_orf_pool import KMutationORFPool
from poolparty.random_mutation_orf_pool import RandomMutationORFPool


class TestMetadataLevelValidation:
    """Test validation of metadata level parameter."""
    
    def test_valid_metadata_levels(self):
        """All three levels should be accepted."""
        for level in ['core', 'features', 'complete']:
            pool = Pool(["AAAA"], name="test", metadata=level)
            assert pool._metadata_level == level
    
    def test_invalid_metadata_level_raises(self):
        """Invalid metadata level should raise ValueError."""
        with pytest.raises(ValueError, match="metadata must be one of"):
            Pool(["AAAA"], name="test", metadata='invalid')
    
    def test_default_is_features(self):
        """Default metadata level should be 'features'."""
        pool = Pool(["AAAA"], name="test")
        assert pool._metadata_level == 'features'


class TestCoreMetadataLevel:
    """Test 'core' metadata level returns only essential fields."""
    
    def test_base_pool_core(self):
        """Base Pool with core level should have index, abs_start, abs_end only."""
        pool = Pool(["AAAA", "TTTT"], name="test", metadata='core')
        metadata = pool.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'value' not in metadata
    
    def test_insertion_scan_pool_core(self):
        """InsertionScanPool with core level should NOT include pos, pos_abs, insert."""
        pool = InsertionScanPool("AAAA", "GG", name="scanner", metadata='core')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'pos' not in metadata
        assert 'pos_abs' not in metadata
        assert 'insert' not in metadata
        assert 'value' not in metadata
    
    def test_k_mutation_pool_core(self):
        """KMutationPool with core level should NOT include mutation details."""
        pool = KMutationPool("AAAA", k=1, name="mutator", metadata='core')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'mut_pos' not in metadata
        assert 'mut_pos_abs' not in metadata
        assert 'mut_from' not in metadata
        assert 'mut_to' not in metadata
        assert 'value' not in metadata
    
    def test_mixed_pool_core(self):
        """MixedPool with core level should NOT include selected info."""
        A = Pool(["AAAA"], name="A")
        B = Pool(["TTTT"], name="B")
        mixed = MixedPool([A, B], name="mixed", metadata='core')
        mixed.set_state(0)
        metadata = mixed.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'selected' not in metadata
        assert 'selected_name' not in metadata
        assert 'value' not in metadata


class TestFeaturesMetadataLevel:
    """Test 'features' metadata level (default) includes pool-specific fields."""
    
    def test_base_pool_features(self):
        """Base Pool with features level should have index, abs_start, abs_end, but NO value."""
        pool = Pool(["AAAA", "TTTT"], name="test", metadata='features')
        metadata = pool.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'value' not in metadata  # Not in 'features' level
    
    def test_insertion_scan_pool_features(self):
        """InsertionScanPool with features level should include pos, pos_abs, insert."""
        pool = InsertionScanPool("AAAA", "GG", name="scanner", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 6)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'pos' in metadata
        assert 'pos_abs' in metadata
        assert 'insert' in metadata
        assert 'value' not in metadata
    
    def test_deletion_scan_pool_features(self):
        """DeletionScanPool with features level should include pos, pos_abs, del_len."""
        pool = DeletionScanPool("AAAAAA", deletion_size=2, name="deleter", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 6)
        
        assert 'pos' in metadata
        assert 'pos_abs' in metadata
        assert 'del_len' in metadata
        assert 'value' not in metadata
    
    def test_subseq_pool_features(self):
        """SubseqPool with features level should include pos, width."""
        pool = SubseqPool("AAAATTTT", width=4, name="subseq", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'pos' in metadata
        assert 'width' in metadata
        assert 'value' not in metadata
    
    def test_shuffle_scan_pool_features(self):
        """ShuffleScanPool with features level should include pos, pos_abs, window_size."""
        pool = ShuffleScanPool("AAAATTTT", shuffle_size=4, name="shuffler", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 8)
        
        assert 'pos' in metadata
        assert 'pos_abs' in metadata
        assert 'window_size' in metadata
        assert 'value' not in metadata
    
    def test_k_mutation_pool_features(self):
        """KMutationPool with features level should include mutation details."""
        pool = KMutationPool("AAAA", k=1, name="mutator", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'mut_pos' in metadata
        assert 'mut_pos_abs' in metadata
        assert 'mut_from' in metadata
        assert 'mut_to' in metadata
        assert 'value' not in metadata
    
    def test_random_mutation_pool_features(self):
        """RandomMutationPool with features level should include mutation details."""
        pool = RandomMutationPool("AAAA", mutation_rate=0.5, name="random_mut", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'mut_count' in metadata
        assert 'mut_pos' in metadata
        assert 'mut_pos_abs' in metadata
        assert 'mut_from' in metadata
        assert 'mut_to' in metadata
        assert 'value' not in metadata
    
    def test_mixed_pool_features(self):
        """MixedPool with features level should include selected info."""
        A = Pool(["AAAA"], name="A")
        B = Pool(["TTTT"], name="B")
        mixed = MixedPool([A, B], name="mixed", metadata='features')
        mixed.set_state(0)
        metadata = mixed.get_metadata(0, 4)
        
        assert 'selected' in metadata
        assert 'selected_name' in metadata
        assert 'value' not in metadata


class TestCompleteMetadataLevel:
    """Test 'complete' metadata level includes value field."""
    
    def test_base_pool_complete(self):
        """Base Pool with complete level should have index, abs_start, abs_end, and value."""
        pool = Pool(["AAAA", "TTTT"], name="test", metadata='complete', mode='sequential')
        pool.set_sequential_op_states(0)
        seq = pool.seq  # This triggers computation
        metadata = pool.get_metadata(0, 4)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_insertion_scan_pool_complete(self):
        """InsertionScanPool with complete level should include pos, pos_abs, insert, AND value."""
        pool = InsertionScanPool("AAAA", "GG", name="scanner", metadata='complete')
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 6)
        
        assert 'pos' in metadata
        assert 'pos_abs' in metadata
        assert 'insert' in metadata
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_k_mutation_pool_complete(self):
        """KMutationPool with complete level should include mutation details AND value."""
        pool = KMutationPool("AAAA", k=1, name="mutator", metadata='complete')
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'mut_pos' in metadata
        assert 'mut_pos_abs' in metadata
        assert 'mut_from' in metadata
        assert 'mut_to' in metadata
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_mixed_pool_complete(self):
        """MixedPool with complete level should include selected info AND value."""
        A = Pool(["AAAA"], name="A")
        B = Pool(["TTTT"], name="B")
        mixed = MixedPool([A, B], name="mixed", metadata='complete')
        mixed.set_state(0)
        seq = mixed.seq
        metadata = mixed.get_metadata(0, 4)
        
        assert 'selected' in metadata
        assert 'selected_name' in metadata
        assert 'value' in metadata
        assert metadata['value'] == seq


class TestMetadataLevelInheritance:
    """Test that metadata level is properly propagated through pool types."""
    
    def test_kmer_pool_inherits_metadata(self):
        """KmerPool should respect metadata parameter."""
        pool = KmerPool(length=4, name="kmer", metadata='complete')
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_barcode_pool_inherits_metadata(self):
        """BarcodePool should respect metadata parameter."""
        pool = BarcodePool(num_barcodes=5, length=6, name="barcode", metadata='complete', seed=42)
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 6)
        
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_iupac_pool_inherits_metadata(self):
        """IUPACPool should respect metadata parameter."""
        pool = IUPACPool("NNNN", name="iupac", metadata='complete')
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 4)
        
        assert 'value' in metadata
        assert metadata['value'] == seq


class TestMetadataLevelsInComposites:
    """Test metadata levels work correctly in composite pools."""
    
    def test_concatenation_with_different_levels(self):
        """Pools in concatenation can have different metadata levels."""
        core_pool = Pool(["AAAA"], name="core", metadata='core')
        complete_pool = Pool(["TTTT"], name="complete", metadata='complete')
        
        library = core_pool + complete_pool
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        cards = result['design_cards']
        
        row = cards.get_row(0)
        
        # Core pool should NOT have value
        assert 'core_value' not in row
        
        # Complete pool should have value
        assert row['complete_value'] == 'TTTT'
    
    def test_mixed_pool_children_preserve_their_levels(self):
        """Children in MixedPool keep their own metadata levels."""
        A = Pool(["AAAA"], name="A", metadata='core')
        B = Pool(["TTTT"], name="B", metadata='complete')
        
        mixed = MixedPool([A, B], name="mixed", mode='sequential')
        
        library = mixed
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        cards = result['design_cards']
        
        # Row 0 selects A (core level) - should NOT have value
        row0 = cards.get_row(0)
        assert 'A_value' not in row0 or row0.get('A_value') is None
        
        # Row 1 selects B (complete level) - should have value
        row1 = cards.get_row(1)
        assert row1['B_value'] == 'TTTT'
    
    def test_transformer_parent_with_different_level(self):
        """Transformer and its parent can have different metadata levels."""
        parent = Pool(["AAAA"], name="parent", metadata='core')
        transformer = KMutationPool(parent, k=1, name="mutator", metadata='complete')
        
        result = transformer.generate_seqs(num_seqs=1, return_design_cards=True)
        cards = result['design_cards']
        row = cards.get_row(0)
        
        # Parent should NOT have value (core level)
        assert 'parent_value' not in row or row.get('parent_value') is None
        
        # Transformer should have mutation details and value (complete level)
        assert row['mutator_mut_pos'] is not None
        assert row['mutator_value'] is not None


class TestORFPoolMetadataLevels:
    """Test metadata levels for ORF-based pools."""
    
    def test_k_mutation_orf_pool_core(self):
        """KMutationORFPool with core level should NOT include codon details."""
        pool = KMutationORFPool("ATGAAATTT", mutation_type='any_codon', k=1, 
                                name="kmut", metadata='core')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 9)
        
        assert 'index' in metadata
        assert 'abs_start' in metadata
        assert 'abs_end' in metadata
        assert 'codon_pos' not in metadata
        assert 'codon_pos_abs' not in metadata
        assert 'aa_from' not in metadata
        assert 'value' not in metadata
    
    def test_k_mutation_orf_pool_features(self):
        """KMutationORFPool with features level should include codon details."""
        pool = KMutationORFPool("ATGAAATTT", mutation_type='any_codon', k=1,
                                name="kmut", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 9)
        
        assert 'codon_pos' in metadata
        assert 'codon_pos_abs' in metadata
        assert 'codon_from' in metadata
        assert 'codon_to' in metadata
        assert 'aa_from' in metadata
        assert 'aa_to' in metadata
        assert 'value' not in metadata
    
    def test_k_mutation_orf_pool_complete(self):
        """KMutationORFPool with complete level should include codon details AND value."""
        pool = KMutationORFPool("ATGAAATTT", mutation_type='any_codon', k=1,
                                name="kmut", metadata='complete')
        pool.set_state(0)
        seq = pool.seq
        metadata = pool.get_metadata(0, 9)
        
        assert 'codon_pos' in metadata
        assert 'aa_from' in metadata
        assert 'value' in metadata
        assert metadata['value'] == seq
    
    def test_random_mutation_orf_pool_features(self):
        """RandomMutationORFPool with features level should include codon details."""
        pool = RandomMutationORFPool("ATGAAATTT", mutation_type='any_codon', 
                                     mutation_rate=0.5, name="rmut", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 9)
        
        assert 'mut_count' in metadata
        assert 'codon_pos' in metadata
        assert 'value' not in metadata
    
    def test_insertion_scan_orf_pool_features(self):
        """InsertionScanORFPool with features level should include codon position details."""
        pool = InsertionScanORFPool("ATGAAATTT", "GAA", name="iscan", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 12)  # Length changes with insertion
        
        assert 'codon_pos' in metadata
        assert 'codon_pos_abs' in metadata
        assert 'insert' in metadata
        assert 'insert_aa' in metadata
        assert 'value' not in metadata
    
    def test_deletion_scan_orf_pool_features(self):
        """DeletionScanORFPool with features level should include codon position details."""
        pool = DeletionScanORFPool("ATGAAATTTGGG", deletion_size=1, name="dscan", metadata='features')
        pool.set_state(0)
        _ = pool.seq
        metadata = pool.get_metadata(0, 12)  # Length changes based on mark_changes
        
        assert 'codon_pos' in metadata
        assert 'codon_pos_abs' in metadata
        assert 'del_codons' in metadata
        assert 'del_aa' in metadata
        assert 'value' not in metadata


class TestDesignCardsGenerationWithLevels:
    """Test generate_seqs() with return_design_cards respects metadata levels."""
    
    def test_complete_level_has_value_in_cards(self):
        """Design cards from pool with 'complete' level should have _value column."""
        pool = Pool(["AAAA", "TTTT"], name="test", metadata='complete', mode='sequential')
        result = pool.generate_library(num_seqs=2, return_design_cards=True)
        cards = result['design_cards']
        
        assert 'test_value' in cards
        assert cards['test_value'][0] == 'AAAA'
        assert cards['test_value'][1] == 'TTTT'
    
    def test_features_level_no_value_in_cards(self):
        """Design cards from pool with 'features' level should NOT have _value column."""
        pool = Pool(["AAAA", "TTTT"], name="test", metadata='features', mode='sequential')
        result = pool.generate_library(num_seqs=2, return_design_cards=True)
        cards = result['design_cards']
        
        assert 'test_value' not in cards
        assert 'test_index' in cards
        assert 'test_abs_start' in cards
    
    def test_core_level_minimal_columns(self):
        """Design cards from pool with 'core' level should have minimal columns."""
        scanner = InsertionScanPool("AAAA", "GG", name="scanner", metadata='core', mode='sequential')
        result = scanner.generate_seqs(num_seqs=3, return_design_cards=True)
        cards = result['design_cards']
        
        assert 'scanner_index' in cards
        assert 'scanner_abs_start' in cards
        assert 'scanner_abs_end' in cards
        assert 'scanner_pos' not in cards
        assert 'scanner_pos_abs' not in cards
        assert 'scanner_insert' not in cards
        assert 'scanner_value' not in cards
    
    def test_mixed_levels_in_composite(self):
        """Composite with pools of different levels should respect each level."""
        core = Pool(["AAAA"], name="core", metadata='core')
        features = InsertionScanPool("TTTT", "GG", name="features", metadata='features')
        complete = Pool(["CCCC"], name="complete", metadata='complete')
        
        library = core + features + complete
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        cards = result['design_cards']
        
        row = cards.get_row(0)
        
        # Core pool: index, abs_start, abs_end only
        assert row['core_index'] is not None
        assert row['core_abs_start'] == 0
        assert row['core_abs_end'] == 4
        assert 'core_value' not in row
        
        # Features pool: includes pos, pos_abs, insert but not value
        assert row['features_pos'] is not None
        assert row['features_pos_abs'] is not None
        assert row['features_insert'] == 'GG'
        assert 'features_value' not in row
        
        # Complete pool: includes value
        assert row['complete_value'] == 'CCCC'


class TestValueFieldConsistency:
    """Test that value field in 'complete' level is consistent with sequence."""
    
    def test_value_matches_sequence(self):
        """Value field should match the sequence produced by the pool."""
        pool = Pool(["AAAA", "TTTT", "CCCC"], name="test", metadata='complete', mode='sequential')
        
        for i in range(3):
            pool.set_sequential_op_states(i)
            seq = pool.seq
            metadata = pool.get_metadata(0, 4)
            assert metadata['value'] == seq
    
    def test_mutation_value_matches_mutated_sequence(self):
        """Value field for mutation pool should reflect mutations."""
        pool = KMutationPool("AAAA", k=1, name="mutator", metadata='complete')
        
        for i in range(5):
            pool.set_state(i)
            seq = pool.seq
            metadata = pool.get_metadata(0, 4)
            assert metadata['value'] == seq
            # Value should differ from original at mutation positions
            if metadata['mut_pos']:
                assert metadata['value'] != 'AAAA'
    
    def test_insertion_scan_value_matches_modified_sequence(self):
        """Value field for insertion scan pool should reflect insertion."""
        pool = InsertionScanPool("AAAA", "GG", name="scanner", metadata='complete', mode='sequential')
        
        for i in range(3):
            pool.set_state(i)
            seq = pool.seq
            metadata = pool.get_metadata(0, len(seq))
            assert metadata['value'] == seq
            assert 'GG' in metadata['value']

