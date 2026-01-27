"""Integration tests for sequence name construction through DAG execution."""
import pytest
import poolparty as pp


def test_simple_operation_naming():
    """Test that a simple operation generates correct names."""
    with pp.Party():
        pool = pp.from_seq('ACGTACGT').mutagenize(
            num_mutations=1, mode='sequential', prefix='mut'
        )
    
    df = pool.generate_library(num_cycles=1)
    
    # Check that names follow the pattern: mut_0, mut_1, mut_2, ...
    for idx, name in enumerate(df['name']):
        assert name == f'mut_{idx}', f"Expected 'mut_{idx}', got '{name}'"
    
    # Verify no duplicate segments
    for name in df['name']:
        parts = name.split('.')
        # Check for redundant consecutive segments
        for i in range(len(parts) - 1):
            assert parts[i] != parts[i+1], f"Name has duplicate consecutive segments: {name}"


def test_operation_with_region_naming():
    """Test that operations with region parameter generate correct names."""
    with pp.Party():
        pool = pp.from_seq('AAA<reg>CCCCCC</reg>GGG').mutagenize(
            region='reg', num_mutations=1, mode='sequential', prefix='mut'
        )
    
    df = pool.generate_library(num_cycles=1)
    
    # Names should be mut_X format (NOT mut_X.mut_X or similar)
    for name in df['name']:
        # Should start with mut_
        assert name.startswith('mut_'), f"Expected name to start with 'mut_', got '{name}'"
        
        # Should have no dots (single segment only)
        assert '.' not in name, f"Expected single segment name, got '{name}'"
        
        # Ensure no duplicate segments like mut_0.mut_0
        assert 'mut_' not in name[4:], f"Name appears to have duplicate 'mut_' segment: {name}"


def test_chained_operations_naming():
    """Test name construction through chained operations."""
    with pp.Party():
        pool = pp.from_seq('ACGT<bc/>TGCA').mutagenize(
            num_mutations=1, mode='random', num_states=5, prefix='mut'
        ).insert_kmers(
            region='bc', length=3, mode='sequential', prefix='bc'
        )
    
    df = pool.generate_library(num_cycles=1, seed=42)
    
    # Names should follow pattern: mut_X.bc_Y
    for name in df['name']:
        parts = name.split('.')
        assert len(parts) == 2, f"Expected 2 name parts, got {len(parts)}: {name}"
        assert parts[0].startswith('mut_'), f"First part should start with 'mut_': {name}"
        assert parts[1].startswith('bc_'), f"Second part should start with 'bc_': {name}"
        
        # Check no duplicates
        assert parts[0] != parts[1], f"Name has duplicate parts: {name}"


def test_stack_then_insert_kmers_naming():
    """Test the specific pattern from the MPRA example: stack -> insert_kmers."""
    with pp.Party():
        # Create two simple pools
        pool1 = pp.from_seq('AAAA<bc/>TTTT').mutagenize(
            num_mutations=1, mode='sequential', prefix='mut1'
        )
        pool2 = pp.from_seq('CCCC<bc/>GGGG').mutagenize(
            num_mutations=1, mode='sequential', prefix='mut2'
        )
        
        # Stack them and insert barcodes
        stacked = pp.stack([pool1, pool2]).insert_kmers(
            region='bc', length=2, mode='sequential', prefix='bc'
        )
    
    df = stacked.generate_library(num_cycles=1)
    
    # Names should be like: mut1_X.bc_Y or mut2_X.bc_Y
    for name in df['name']:
        parts = name.split('.')
        assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}: {name}"
        
        # First part should be mut1_X or mut2_X
        assert parts[0].startswith('mut1_') or parts[0].startswith('mut2_'), \
            f"First part should start with 'mut1_' or 'mut2_': {name}"
        
        # Second part should be bc_Y
        assert parts[1].startswith('bc_'), f"Second part should start with 'bc_': {name}"
        
        # Check no duplicate segments (e.g., mut1_0.mut1_0.bc_0)
        name_str = str(name)
        # Split and check for any repeated segments
        segments = name_str.split('.')
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i+1], \
                f"Name has duplicate consecutive segments: {name}"


def test_no_name_duplication_complex_pipeline():
    """Test complex pipeline similar to MPRA example doesn't produce duplicate names."""
    with pp.Party():
        template = pp.from_seq('AA<cre>CCCCCC</cre>GG<bc/>TT')
        
        mutated = template.mutagenize(
            region='cre', mutation_rate=0.1, mode='random', num_states=3, prefix='mut'
        )
        
        deleted = template.deletion_scan(
            region='cre', deletion_length=2, mode='sequential', prefix='del'
        )
        
        combined = pp.stack([mutated, deleted]).insert_kmers(
            region='bc', length=2, mode='sequential', prefix='bc'
        )
    
    df = combined.generate_library(num_cycles=1, seed=42)
    
    # Check all names for any duplicate segments
    for name in df['name']:
        name_str = str(name)
        segments = name_str.split('.')
        
        # Check for consecutive duplicates
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i+1], \
                f"Found duplicate consecutive segments in: {name}"
        
        # Check for any repeated segment at all (stricter test)
        # e.g., "mut_0.bc_0.mut_0" would fail this
        segment_set = set()
        for seg in segments:
            # Check if we've seen this exact segment before
            base_segment = seg.split('_')[0] if '_' in seg else seg
            if base_segment in ['mut', 'del', 'bc']:
                # For operation names, the full segment should be unique
                assert seg not in segment_set, \
                    f"Found repeated segment '{seg}' in: {name}"
                segment_set.add(seg)


def test_repeat_states_naming():
    """Test that repeat_states generates correct names."""
    with pp.Party():
        pool = pp.from_seq('ACGT').mutagenize(
            num_mutations=1, mode='sequential', prefix='mut'
        ).repeat_states(3, prefix='rep')
    
    df = pool.generate_library(num_cycles=1)
    
    # Names should be like: mut_X.rep_Y
    for name in df['name']:
        parts = name.split('.')
        assert len(parts) == 2, f"Expected 2 parts: {name}"
        assert parts[0].startswith('mut_'), f"First part should be 'mut_X': {name}"
        assert parts[1].startswith('rep_'), f"Second part should be 'rep_Y': {name}"
        
        # No duplicates
        assert parts[0] != parts[1]


def test_insertion_scan_naming():
    """Test insertion_scan generates correct names without duplication."""
    with pp.Party():
        sites = pp.from_seqs(['AAA', 'TTT'], mode='sequential', prefix='site')
        pool = pp.from_seq('GGG<reg>CCCCCC</reg>GGG').insertion_scan(
            region='reg',
            ins_pool=sites,
            positions=[0, 3],
            mode='sequential',
            prefix='ins',
            prefix_position='pos',
            prefix_insert='site'
        )
    
    df = pool.generate_library(num_cycles=1)
    
    # Check names don't have duplicate segments
    for name in df['name']:
        segments = name.split('.')
        # Should have pattern like: ins_X.pos_Y.site_Z (or similar)
        # Main check: no segment appears twice
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i+1], \
                f"Duplicate consecutive segments in: {name}"


def test_shuffle_scan_naming():
    """Test shuffle_scan generates correct names."""
    with pp.Party():
        pool = pp.from_seq('AAA<reg>CCCCCC</reg>GGG').shuffle_scan(
            region='reg',
            shuffle_length=3,
            positions=[0, 3],
            mode='sequential',
            prefix='shuf',
            prefix_position='pos',
            prefix_shuffle='s'
        )
    
    df = pool.generate_library(num_cycles=1, seed=42)
    
    # Check no duplicate segments
    for name in df['name']:
        segments = name.split('.')
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i+1], \
                f"Duplicate consecutive segments in: {name}"
