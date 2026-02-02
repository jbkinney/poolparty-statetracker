"""Tests for the from_fasta operation."""

import gc
import os
import tempfile
import time

import pytest
from poolparty.fixed_ops.fixed import FixedOp
from poolparty.fixed_ops.from_fasta import from_fasta


def _remove_file_with_retry(path, max_attempts=5, delay=0.1):
    """Remove a file with retry logic for Windows file locking issues."""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(path):
                os.unlink(path)
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                gc.collect()  # Force garbage collection to release file handles
                time.sleep(delay * (attempt + 1))  # Increasing delay
            else:
                raise


@pytest.fixture
def test_fasta():
    """Create a temporary FASTA file for testing."""
    content = """>chr1
ACGTACGTACGTACGT
>chr2
AAAACCCCGGGGTTTT
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(content)
        fasta_path = f.name
    yield fasta_path
    # Cleanup with retry logic for Windows
    gc.collect()  # Ensure any lingering file handles are released
    _remove_file_with_retry(fasta_path)
    _remove_file_with_retry(fasta_path + ".fai")


class TestFromFastaFactory:
    """Test from_fasta factory function."""

    def test_returns_pool(self, test_fasta):
        """Test that from_fasta returns a Pool."""
        pool = from_fasta(test_fasta, "chr1", 0, 8)
        assert pool is not None
        assert hasattr(pool, "operation")

    def test_creates_fixed_op(self, test_fasta):
        """Test that from_fasta creates a FixedOp."""
        pool = from_fasta(test_fasta, "chr1", 0, 8)
        assert isinstance(pool.operation, FixedOp)

    def test_mode_is_fixed(self, test_fasta):
        """Test that the operation mode is always 'fixed'."""
        pool = from_fasta(test_fasta, "chr1", 0, 8)
        assert pool.operation.mode == "fixed"


class TestFromFastaExtraction:
    """Test sequence extraction from FASTA."""

    def test_extracts_correct_sequence(self, test_fasta):
        """Test that the correct sequence is extracted."""
        pool = from_fasta(test_fasta, "chr1", 0, 8).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_extracts_subsequence(self, test_fasta):
        """Test extraction of a subsequence."""
        pool = from_fasta(test_fasta, "chr1", 4, 12).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_extracts_from_different_chrom(self, test_fasta):
        """Test extraction from a different chromosome."""
        pool = from_fasta(test_fasta, "chr2", 0, 4).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAAA"


class TestFromFastaStrand:
    """Test strand handling in from_fasta."""

    def test_plus_strand_no_change(self, test_fasta):
        """Test that '+' strand returns sequence unchanged."""
        pool = from_fasta(test_fasta, "chr1", 0, 4, strand="+").named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_minus_strand_reverse_complement(self, test_fasta):
        """Test that '-' strand returns reverse complement."""
        # ACGT -> reverse: TGCA -> complement: ACGT
        pool = from_fasta(test_fasta, "chr1", 0, 4, strand="-").named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_minus_strand_asymmetric(self, test_fasta):
        """Test reverse complement with asymmetric sequence."""
        # AAAA -> reverse: AAAA -> complement: TTTT
        pool = from_fasta(test_fasta, "chr2", 0, 4, strand="-").named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "TTTT"


class TestFromFastaOperationName:
    """Test from_fasta operation naming."""

    def test_factory_name_is_from_fasta(self, test_fasta):
        """Test that factory name is 'from_fasta'."""
        pool = from_fasta(test_fasta, "chr1", 0, 8)
        assert pool.operation.name.endswith(":from_fasta")

    def test_custom_pool_name(self, test_fasta):
        """Test custom pool name."""
        pool = from_fasta(test_fasta, "chr1", 0, 8).named("my_pool")
        assert pool.name == "my_pool"


class TestFromFastaErrors:
    """Test error handling in from_fasta."""

    def test_invalid_chrom_raises_error(self, test_fasta):
        """Test that invalid chromosome raises KeyError."""
        with pytest.raises(KeyError):
            from_fasta(test_fasta, "chrX", 0, 8)

    def test_invalid_strand_raises_error(self, test_fasta):
        """Test that invalid strand value raises error."""
        with pytest.raises(Exception):
            from_fasta(test_fasta, "chr1", 0, 8, strand="invalid")
