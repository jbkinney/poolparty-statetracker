"""Regression tests for the stale cache bug in get_metadata.

The bug: Pools with cached metadata (like _cached_pos, _cached_orientation, etc.)
would return stale cached data when get_metadata() was called after set_state()
without accessing .seq first.

The fix: Each pool now tracks _cached_state and recomputes if state has changed.

These tests verify that get_metadata returns correct values after state changes
even when .seq is not explicitly accessed.
"""

import pytest
import pandas as pd
from poolparty import (
    Pool,
    InsertionScanPool,
    DeletionScanPool,
    SubseqPool,
    ShuffleScanPool,
    KMutationPool,
    RandomMutationPool,
    MotifPool,
)


class TestStaleCacheBug:
    """Tests that verify the stale cache bug is fixed."""
    
    def test_insertion_scan_pool_stale_cache(self):
        """InsertionScanPool get_metadata should return current position after state change."""
        pool = InsertionScanPool(
            background_seq='NNNNNNNN',
            insert_seq='AAA',
            mode='sequential',
            name='scan'
        )
        
        # Access state 0
        pool.set_state(0)
        _ = pool.seq  # Populate cache
        meta0 = pool.get_metadata(0, 11)
        assert meta0['pos'] == 0
        
        # Change to state 2 WITHOUT accessing .seq
        pool.set_state(2)
        meta2 = pool.get_metadata(0, 11)
        
        # BUG: Without the fix, meta2['pos'] would still be 0 (stale)
        assert meta2['pos'] == 2, f"Expected pos=2, got pos={meta2['pos']} (stale cache bug!)"
    
    def test_deletion_scan_pool_stale_cache(self):
        """DeletionScanPool get_metadata should return current position after state change."""
        pool = DeletionScanPool(
            background_seq='NNNNNNNNNNNN',
            deletion_size=3,
            mode='sequential',
            name='del'
        )
        
        pool.set_state(0)
        _ = pool.seq
        meta0 = pool.get_metadata(0, 12)
        assert meta0['pos'] == 0
        
        pool.set_state(3)
        meta3 = pool.get_metadata(0, 12)
        assert meta3['pos'] == 3, f"Expected pos=3, got pos={meta3['pos']} (stale cache bug!)"
    
    def test_subseq_pool_stale_cache(self):
        """SubseqPool get_metadata should return current position after state change."""
        pool = SubseqPool(
            seq='NNNNNNNNNN',
            width=4,
            mode='sequential',
            name='sub'
        )
        
        pool.set_state(0)
        _ = pool.seq
        meta0 = pool.get_metadata(0, 4)
        assert meta0['pos'] == 0
        
        pool.set_state(3)
        meta3 = pool.get_metadata(0, 4)
        assert meta3['pos'] == 3, f"Expected pos=3, got pos={meta3['pos']} (stale cache bug!)"
    
    def test_shuffle_scan_pool_stale_cache(self):
        """ShuffleScanPool get_metadata should return current position after state change."""
        pool = ShuffleScanPool(
            background_seq='NNNNNNNNNN',
            shuffle_size=3,
            mode='sequential',
            name='shuf'
        )
        
        pool.set_state(0)
        _ = pool.seq
        meta0 = pool.get_metadata(0, 10)
        assert meta0['pos'] == 0
        
        pool.set_state(3)
        meta3 = pool.get_metadata(0, 10)
        assert meta3['pos'] == 3, f"Expected pos=3, got pos={meta3['pos']} (stale cache bug!)"
    
    def test_k_mutation_pool_stale_cache(self):
        """KMutationPool get_metadata should return current mutations after state change."""
        pool = KMutationPool(
            seq='AAAA',
            k=1,
            mode='sequential',
            name='kmut'
        )
        
        # State 0: mutation at position 0
        pool.set_state(0)
        _ = pool.seq
        meta0 = pool.get_metadata(0, 4)
        pos0 = meta0['mut_pos']
        
        # State 3: mutation at position 1 (different position in the cycle)
        pool.set_state(3)
        meta3 = pool.get_metadata(0, 4)
        pos3 = meta3['mut_pos']
        
        # The positions should be different for different states
        # (Exact values depend on enumeration order, but they shouldn't be identical)
        assert pos0 != pos3 or meta0['mut_to'] != meta3['mut_to'], \
            f"Metadata should differ between states (stale cache bug!)"
    
    def test_random_mutation_pool_stale_cache(self):
        """RandomMutationPool get_metadata should return current mutations after state change."""
        pool = RandomMutationPool(
            seq='AAAAAAAAAA',
            mutation_rate=0.5,
            name='rand'
        )
        
        # Different states produce different random mutations
        pool.set_state(42)
        _ = pool.seq
        meta42 = pool.get_metadata(0, 10)
        
        pool.set_state(12345)
        meta12345 = pool.get_metadata(0, 10)
        
        # With different seeds, mutations should (very likely) be different
        assert meta42['mut_pos'] != meta12345['mut_pos'] or \
               meta42['mut_count'] != meta12345['mut_count'], \
            "Different states should produce different mutations (stale cache bug!)"
    
    def test_motif_pool_stale_cache(self):
        """MotifPool get_metadata should return current orientation after state change."""
        # PWM with 50/50 probability at each position
        pwm = pd.DataFrame({
            'A': [0.25, 0.25, 0.25],
            'C': [0.25, 0.25, 0.25],
            'G': [0.25, 0.25, 0.25],
            'T': [0.25, 0.25, 0.25]
        })
        
        pool = MotifPool(
            probability_df=pwm,
            orientation='both',
            forward_prob=0.5,
            name='motif'
        )
        
        # Try multiple states to find ones with different orientations
        orientations = {}
        for state in range(100):
            pool.set_state(state)
            meta = pool.get_metadata(0, 3)
            orientations[state] = meta['orientation']
        
        # With 50/50 probability, we should see both orientations
        unique_orientations = set(orientations.values())
        assert len(unique_orientations) == 2, \
            f"Expected both orientations, got {unique_orientations}"
        
        # Now verify the actual bug scenario: metadata should update without .seq
        # Find a state with 'forward' and one with 'reverse'
        forward_state = next(s for s, o in orientations.items() if o == 'forward')
        reverse_state = next(s for s, o in orientations.items() if o == 'reverse')
        
        # Access forward state
        pool.set_state(forward_state)
        _ = pool.seq  # Populate cache
        meta_fwd = pool.get_metadata(0, 3)
        assert meta_fwd['orientation'] == 'forward'
        
        # Change to reverse state WITHOUT accessing .seq
        pool.set_state(reverse_state)
        meta_rev = pool.get_metadata(0, 3)
        
        assert meta_rev['orientation'] == 'reverse', \
            f"Expected 'reverse', got '{meta_rev['orientation']}' (stale cache bug!)"


class TestCacheStateTracking:
    """Tests that verify _cached_state is properly tracked."""
    
    def test_cached_state_none_initially(self):
        """_cached_state should be None before first .seq access."""
        pool = InsertionScanPool('NNNNNN', 'AA', mode='sequential')
        assert pool._cached_state is None
    
    def test_cached_state_updated_after_seq(self):
        """_cached_state should be updated after .seq access."""
        pool = InsertionScanPool('NNNNNN', 'AA', mode='sequential')
        pool.set_state(3)
        _ = pool.seq
        assert pool._cached_state == pool.get_state()
    
    def test_cached_state_triggers_recompute(self):
        """get_metadata should recompute when _cached_state differs from current state."""
        pool = InsertionScanPool('NNNNNN', 'AA', mode='sequential')
        
        pool.set_state(0)
        _ = pool.seq
        initial_state = pool._cached_state
        
        # Change state - _cached_state is now stale
        pool.set_state(2)
        assert pool._cached_state != pool.get_state()
        
        # get_metadata should trigger recompute
        pool.get_metadata(0, 8)
        
        # Now _cached_state should match current state
        assert pool._cached_state == pool.get_state()

