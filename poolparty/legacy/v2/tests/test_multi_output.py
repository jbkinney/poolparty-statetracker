"""Tests for MultiPool and OutputSelectorOp infrastructure."""

import pytest
from poolparty import (
    Pool,
    MultiPool,
    OutputSelectorOp,
    Operation,
    from_seqs,
    breakpoint_scan,
)
from poolparty.utils import reset_op_id_counter


@pytest.fixture(autouse=True)
def reset_counter():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()


class TestOutputSelectorOp:
    """Tests for OutputSelectorOp class."""
    
    def test_output_selector_extracts_correct_column(self):
        """OutputSelectorOp should extract the correct seq_N column."""
        # Create a multi-output operation via breakpoint_scan
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        
        # The MultiPool should have 2 outputs
        assert len(multi) == 2
        
        # Get the underlying operations
        left_pool = multi[0]
        right_pool = multi[1]
        
        # Each pool should have an OutputSelectorOp
        assert isinstance(left_pool.operation, OutputSelectorOp)
        assert isinstance(right_pool.operation, OutputSelectorOp)
        
        # Check output indices
        assert left_pool.operation._output_index == 0
        assert right_pool.operation._output_index == 1
    
    def test_output_selector_invalid_index_raises(self):
        """OutputSelectorOp should raise for invalid output_index."""
        # Create a simple single-output operation
        pool = from_seqs(["ACGT"])
        
        with pytest.raises(ValueError, match="out of range"):
            OutputSelectorOp(pool.operation, output_index=1)
        
        with pytest.raises(ValueError, match="out of range"):
            OutputSelectorOp(pool.operation, output_index=-1)
    
    def test_output_selector_inherits_mode(self):
        """OutputSelectorOp should inherit mode from source operation."""
        multi_seq = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        assert multi_seq[0].operation.mode == 'sequential'
        
        multi_rand = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='random')
        assert multi_rand[0].operation.mode == 'random'
    
    def test_output_selector_inherits_num_states(self):
        """OutputSelectorOp should inherit num_states from source operation."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        source_op = multi.operation
        
        for pool in multi:
            assert pool.operation.num_states == source_op.num_states


class TestMultiPool:
    """Tests for MultiPool class."""
    
    def test_multipool_requires_multi_output_op(self):
        """MultiPool should raise for single-output operations."""
        pool = from_seqs(["ACGT"])
        
        with pytest.raises(ValueError, match="requires num_outputs > 1"):
            MultiPool(pool.operation)
    
    def test_multipool_length(self):
        """MultiPool length should match num_outputs."""
        for n in [1, 2, 3]:
            multi = breakpoint_scan("ACGTACGTACGT", num_breakpoints=n, mode='sequential')
            assert len(multi) == n + 1
    
    def test_multipool_indexing(self):
        """MultiPool should support integer indexing."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=2, mode='sequential')
        
        # Positive indexing
        assert isinstance(multi[0], Pool)
        assert isinstance(multi[1], Pool)
        assert isinstance(multi[2], Pool)
        
        # Negative indexing
        assert multi[-1] is multi[2]
        assert multi[-2] is multi[1]
        assert multi[-3] is multi[0]
    
    def test_multipool_index_out_of_range(self):
        """MultiPool should raise IndexError for out of range indices."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        
        with pytest.raises(IndexError):
            multi[2]
        
        with pytest.raises(IndexError):
            multi[-3]
    
    def test_multipool_iteration(self):
        """MultiPool should support iteration."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=2, mode='sequential')
        
        pools = list(multi)
        assert len(pools) == 3
        for pool in pools:
            assert isinstance(pool, Pool)
    
    def test_multipool_unpacking(self):
        """MultiPool should support unpacking."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=2, mode='sequential')
        
        left, middle, right = multi
        assert isinstance(left, Pool)
        assert isinstance(middle, Pool)
        assert isinstance(right, Pool)
    
    def test_multipool_operation_property(self):
        """MultiPool.operation should return the source operation."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        
        assert multi.operation.num_outputs == 2
        assert multi.num_outputs == 2
    
    def test_multipool_repr(self):
        """MultiPool should have a readable repr."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        repr_str = repr(multi)
        
        assert "MultiPool" in repr_str
        assert "num_outputs=2" in repr_str


class TestMultiPoolIntegration:
    """Integration tests for MultiPool with operations."""
    
    def test_segments_are_independent_pools(self):
        """Each segment from MultiPool should be usable as an independent Pool."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, positions=[4], mode='sequential')
        left, right = multi
        
        # Each should generate sequences independently
        left_seq = left.seq
        right_seq = right.seq
        
        assert isinstance(left_seq, str)
        assert isinstance(right_seq, str)
        assert len(left_seq) + len(right_seq) == 8  # Original length
    
    def test_segments_concatenate_to_original(self):
        """Concatenating segments should reconstruct the original sequence."""
        original = "ACGTACGTACGT"
        multi = breakpoint_scan(original, num_breakpoints=2, positions=[4, 8], mode='sequential')
        left, middle, right = multi
        
        reconstructed = left + middle + right
        assert reconstructed.seq == original
    
    def test_segments_as_parents_to_other_operations(self):
        """Segments should work as parents to other operations like concatenate."""
        from poolparty import concatenate, from_seqs
        
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, positions=[4], mode='sequential')
        left, right = multi
        
        # Use segments in concatenation
        combined = concatenate([left, from_seqs(["XX"]), right])
        
        # Should be able to generate sequences
        seq = combined.seq
        assert isinstance(seq, str)
        assert seq == "ACGTXXACGT"  # left + XX + right
    
    def test_multipool_generate_library(self):
        """Each segment should support generate_library."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        left, right = multi
        
        # Generate library from left segment
        df = left.generate_library(num_seqs=5)
        
        assert len(df) == 5
        assert 'seq' in df.columns

