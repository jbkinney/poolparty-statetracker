"""Integration tests based on the examples from the original skeleton file."""

import pytest
from poolparty import (
    Pool, KmerPool
)


def test_example1_combinatorial_complete_iteration():
    """Test Example 1: Combinatorially complete iteration with Pools."""
    # Two selection pools - should give all combinations
    pool1 = Pool(seqs=['A', 'B', 'C'])
    pool2 = Pool(seqs=['1', '2'])
    combined = pool1 + '-' + pool2
    
    assert combined.num_states == 6  # 3 * 2
    # Ancestors now includes all pools (including composite ones)
    # Count primitive pools only (those with no parents or op='selection'/'kmer')
    primitive_ancestors = [a for a in combined.ancestors if not a.parents or a.op in ('selection', 'kmer', 'subseq')]
    assert len(primitive_ancestors) == 2
    
    # Set both pools to sequential mode
    pool1.set_mode('sequential')
    pool2.set_mode('sequential')
    
    # Generate all sequences
    all_seqs = combined.generate_seqs(num_complete_iterations=1)
    assert len(all_seqs) == 6
    
    # Check that we get all combinations (order may vary based on id() sorting)
    expected = {'A-1', 'A-2', 'B-1', 'B-2', 'C-1', 'C-2'}
    assert set(all_seqs) == expected


def test_example2_reusing_same_pool():
    """Test Example 2: Reusing the same Pool multiple times in the graph."""
    # Same pool appears twice - should be counted only once
    a = Pool(seqs=['X', 'Y'])
    repeated = a + '-' + a + '-' + a
    
    assert repeated.num_states == 2  # Not 2^3=8, because same pool reused
    # Count unique primitive pools
    primitive_ancestors = [anc for anc in repeated.ancestors if not anc.parents or anc.op in ('selection', 'kmer', 'subseq')]
    assert len(set(primitive_ancestors)) == 1  # Only one unique primitive ancestor
    
    # Set pool to sequential mode
    a.set_mode('sequential')
    
    # All sequences should have the same value repeated 3 times
    all_seqs = repeated.generate_seqs(num_complete_iterations=1)
    assert len(all_seqs) == 2
    assert set(all_seqs) == {'X-X-X', 'Y-Y-Y'}


def test_example3_complex_composition():
    """Test Example 3: Complex composition with multiple Pools."""
    prefix = Pool(seqs=['AAA', 'TTT'])
    middle = Pool(seqs=['GGG', 'CCC'])
    suffix = Pool(seqs=['XX', 'YY', 'ZZ'])
    complex_pool = prefix + '.' + middle + '.' + suffix
    
    assert complex_pool.num_states == 12  # 2×2×3=12
    # Count primitive pools
    primitive_ancestors = [a for a in complex_pool.ancestors if not a.parents or a.op in ('selection', 'kmer', 'subseq')]
    assert len(primitive_ancestors) == 3
    
    # Set all pools to sequential mode
    prefix.set_mode('sequential')
    middle.set_mode('sequential')
    suffix.set_mode('sequential')
    
    all_seqs = complex_pool.generate_seqs(num_complete_iterations=1)
    assert len(all_seqs) == 12
    
    # Verify format: all should have AAA or TTT, GGG or CCC, XX or YY or ZZ
    for seq in all_seqs:
        parts = seq.split('.')
        assert len(parts) == 3
        assert parts[0] in ['AAA', 'TTT']
        assert parts[1] in ['GGG', 'CCC']
        assert parts[2] in ['XX', 'YY', 'ZZ']


def test_example4_kmer_pool_dimers():
    """Test Example 4: KmerPool - all possible sequences (dimers)."""
    # Generate all possible 2-mers from alphabet 'AB'
    dimers = KmerPool(length=2, alphabet=['A', 'B'])
    
    assert dimers.num_states == 4  # 2^2=4
    
    # Set to sequential mode
    dimers.set_mode('sequential')
    
    all_dimers = dimers.generate_seqs(num_complete_iterations=1)
    assert len(all_dimers) == 4
    assert set(all_dimers) == {'AA', 'AB', 'BA', 'BB'}


def test_example4_kmer_pool_trimers():
    """Test Example 4: KmerPool - all possible sequences (trimers)."""
    # Generate all possible 3-mers from alphabet 'XYZ'
    trimers = KmerPool(length=3, alphabet=['X', 'Y', 'Z'])
    
    assert trimers.num_states == 27  # 3^3=27
    
    # Set to sequential mode
    trimers.set_mode('sequential')
    
    all_trimers = trimers.generate_seqs(num_complete_iterations=1)
    assert len(all_trimers) == 27
    
    # Check all are length 3 and use only XYZ
    assert all(len(seq) == 3 for seq in all_trimers)
    assert all(all(c in 'XYZ' for c in seq) for seq in all_trimers)


def test_example5_combining_kmer_pools():
    """Test Example 5: Combining KmerPools."""
    # Combine two KmerPools
    pool_a = KmerPool(length=2, alphabet=['A', 'B'])
    pool_b = KmerPool(length=1, alphabet=['X', 'Y'])
    combined = pool_a + '-' + pool_b
    
    assert combined.num_states == 8  # 4×2=8
    # Count primitive pools
    primitive_ancestors = [a for a in combined.ancestors if not a.parents or a.op in ('selection', 'kmer', 'subseq')]
    assert len(primitive_ancestors) == 2
    
    # Set both to sequential mode
    pool_a.set_mode('sequential')
    pool_b.set_mode('sequential')
    
    all_seqs = combined.generate_seqs(num_complete_iterations=1)
    assert len(all_seqs) == 8
    
    # Check format: 2-mer from AB, dash, 1-mer from XY
    for seq in all_seqs:
        parts = seq.split('-')
        assert len(parts) == 2
        assert len(parts[0]) == 2
        assert all(c in 'AB' for c in parts[0])
        assert len(parts[1]) == 1
        assert parts[1] in 'XY'


def test_example6_all_pools_now_finite():
    """Test Example 6: All pools are now finite."""
    # Both pools are now finite
    finite_part = Pool(seqs=['A', 'B'])
    kmer_part = KmerPool(4, 'dna')
    mixed = finite_part + '-' + kmer_part
    
    assert mixed.num_states == 2 * (4**4)  # 2 * 256 = 512
    assert mixed.is_sequential_compatible()
    # Count primitive pools
    primitive_ancestors = [a for a in mixed.ancestors if not a.parents or a.op in ('selection', 'kmer', 'subseq')]
    assert len(primitive_ancestors) == 2
    
    # Generate some sequences using random generation
    pool_samples = mixed.generate_seqs(num_seqs=10)
    assert len(pool_samples) == 10
    
    # Check format: A or B, dash, 4-char sequence
    for seq in pool_samples:
        parts = seq.split('-')
        assert len(parts) == 2
        assert parts[0] in ['A', 'B']
        assert len(parts[1]) == 4
        assert all(c in 'ACGT' for c in parts[1])


def test_generate_seqs_with_num_seqs():
    """Test generate_seqs with num_seqs parameter for random generation."""
    pool = Pool(seqs=['A', 'B', 'C'])
    seqs = pool.generate_library(num_seqs=10)
    assert len(seqs) == 10
    # All should be from the pool
    assert all(seq in ['A', 'B', 'C'] for seq in seqs)


def test_generate_seqs_without_num_seqs_raises():
    """Test generate_seqs without num_seqs raises error."""
    pool = KmerPool(5)
    with pytest.raises(ValueError, match="Must specify either num_seqs or num_complete_iterations"):
        pool.generate_seqs()


def test_generate_seqs_with_combinatorially_complete():
    """Test generate_seqs with one pool sequential and one random."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = KmerPool(1, ['X', 'Y'])
    combined = pool1 + '-' + pool2
    
    # Set pool1 to sequential, pool2 stays random
    pool1.set_mode('sequential')
    
    # Generate with pool1 complete, pool2 random
    seqs = combined.generate_seqs(num_complete_iterations=3)
    
    # Should get 2 (pool1 states) * 3 (iterations) = 6 sequences
    assert len(seqs) == 6
    
    # Check format
    for seq in seqs:
        parts = seq.split('-')
        assert len(parts) == 2
        assert parts[0] in ['A', 'B']
        assert parts[1] in ['X', 'Y']
    
    # Both A and B should appear
    prefixes = [seq.split('-')[0] for seq in seqs]
    assert 'A' in prefixes
    assert 'B' in prefixes


def test_generate_seqs_with_all_pools_complete():
    """Test generate_seqs with all pools in sequential mode."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = KmerPool(1, ['X', 'Y'])
    combined = pool1 + '-' + pool2
    
    # Set both pools to sequential mode
    pool1.set_mode('sequential')
    pool2.set_mode('sequential')
    
    # Generate with both pools complete
    seqs = combined.generate_seqs(num_complete_iterations=2)
    
    # Should get 2 * 2 (complete states) * 2 (iterations) = 8 sequences
    assert len(seqs) == 8
    
    # Check all combinations appear
    unique_seqs = set(seqs)
    assert 'A-X' in unique_seqs
    assert 'A-Y' in unique_seqs
    assert 'B-X' in unique_seqs
    assert 'B-Y' in unique_seqs


def test_generate_seqs_without_num_complete_iterations_raises():
    """Test generate_seqs with sequential pools but no num_complete_iterations raises."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = KmerPool(1, ['X', 'Y'])
    combined = pool1 + '-' + pool2
    
    # Set pool1 to sequential mode
    pool1.set_mode('sequential')
    
    with pytest.raises(ValueError, match="Must specify either num_seqs or num_complete_iterations"):
        combined.generate_seqs()


def test_generate_seqs_with_num_seqs_and_sequential_pools():
    """Test generate_seqs with num_seqs parameter when sequential pools exist."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['X', 'Y', 'Z'])
    combined = pool1 + '-' + pool2
    
    # Set both pools to sequential mode
    pool1.set_mode('sequential')
    pool2.set_mode('sequential')
    
    # Generate 4 sequences (less than total 6 states)
    seqs = combined.generate_seqs(num_seqs=4)
    
    assert len(seqs) == 4
    # Should get first 4 states in iteration order
    for seq in seqs:
        parts = seq.split('-')
        assert len(parts) == 2
        assert parts[0] in ['A', 'B']
        assert parts[1] in ['X', 'Y', 'Z']


def test_generate_seqs_with_num_seqs_wrapping():
    """Test generate_seqs wrapping behavior when num_seqs exceeds total states."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['1', '2'])
    combined = pool1 + '-' + pool2
    
    # Set both pools to sequential mode
    pool1.set_mode('sequential')
    pool2.set_mode('sequential')
    
    # Total states = 2 * 2 = 4
    # Request 10 sequences, should wrap around
    seqs = combined.generate_seqs(num_seqs=10)
    
    assert len(seqs) == 10
    
    # Get unique sequences to check all 4 states appear
    unique_seqs = set(seqs)
    assert len(unique_seqs) == 4  # Only 4 unique combinations
    
    # All 4 combinations should be present
    expected = {'A-1', 'A-2', 'B-1', 'B-2'}
    assert unique_seqs == expected
    
    # First 4 should match last 4 (one full wrap-around starts at seq index 8)
    # However, due to wrapping, we should see the pattern repeat
    # seqs[0] == seqs[4] == seqs[8]
    assert seqs[0] == seqs[4]
    assert seqs[0] == seqs[8]


def test_generate_seqs_both_parameters_raises():
    """Test generate_seqs raises error when both num_seqs and num_complete_iterations provided."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['X', 'Y'])
    combined = pool1 + '-' + pool2
    
    # Set both pools to sequential mode
    pool1.set_mode('sequential')
    pool2.set_mode('sequential')
    
    with pytest.raises(ValueError, match="Cannot specify both num_seqs and num_complete_iterations"):
        combined.generate_seqs(num_seqs=5, num_complete_iterations=2)


def test_generate_seqs_num_seqs_with_mixed_modes():
    """Test generate_seqs with num_seqs when one pool is sequential and one is random."""
    pool1 = Pool(seqs=['A', 'B', 'C'])
    pool2 = KmerPool(2, ['X', 'Y'])
    combined = pool1 + '-' + pool2
    
    # Set pool1 to sequential, pool2 stays random
    pool1.set_mode('sequential')
    
    # Generate 9 sequences (3 complete cycles through pool1)
    seqs = combined.generate_seqs(num_seqs=9)
    
    assert len(seqs) == 9
    
    # Check format
    for seq in seqs:
        parts = seq.split('-')
        assert len(parts) == 2
        assert parts[0] in ['A', 'B', 'C']
        assert len(parts[1]) == 2
        assert all(c in 'XY' for c in parts[1])
    
    # All three prefixes from pool1 should appear (since we cycle through 3 times)
    prefixes = [seq.split('-')[0] for seq in seqs]
    assert 'A' in prefixes
    assert 'B' in prefixes
    assert 'C' in prefixes

