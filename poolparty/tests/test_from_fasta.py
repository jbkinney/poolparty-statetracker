"""Tests for the from_fasta operation."""

import gc
import os
import tempfile
import time

import pytest


def _remove_file_with_retry(path, max_retries=3, delay=0.1):
    """Remove a file with retry logic for Windows file locking issues."""
    for i in range(max_retries):
        try:
            if os.path.exists(path):
                os.unlink(path)
            return
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                raise

from poolparty.fixed_ops.fixed import FixedOp
from poolparty.fixed_ops.from_fasta import from_fasta


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
    # Cleanup
    os.unlink(fasta_path)
    if os.path.exists(fasta_path + ".fai"):
        os.unlink(fasta_path + ".fai")


@pytest.fixture
def circular_fasta():
    """Create a temporary FASTA file for circular genome testing."""
    # 20bp circular genome
    content = """>circ
ACGTACGTACGTACGTACGT
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(content)
        fasta_path = f.name
    yield fasta_path
    gc.collect()
    _remove_file_with_retry(fasta_path)
    _remove_file_with_retry(fasta_path + ".fai")


class TestFromFastaFactory:
    """Test from_fasta factory function."""

    def test_returns_pool(self, test_fasta):
        """Test that from_fasta returns a Pool."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+"))
        assert pool is not None
        assert hasattr(pool, "operation")

    def test_creates_fixed_op(self, test_fasta):
        """Test that from_fasta creates a FixedOp."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+"))
        assert isinstance(pool.operation, FixedOp)

    def test_mode_is_fixed(self, test_fasta):
        """Test that the operation mode is always 'fixed'."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+"))
        assert pool.operation.mode == "fixed"


class TestFromFastaExtraction:
    """Test sequence extraction from FASTA."""

    def test_extracts_correct_sequence(self, test_fasta):
        """Test that the correct sequence is extracted."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_extracts_subsequence(self, test_fasta):
        """Test extraction of a subsequence."""
        pool = from_fasta(test_fasta, ("chr1", 4, 12, "+")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_extracts_from_different_chrom(self, test_fasta):
        """Test extraction from a different chromosome."""
        pool = from_fasta(test_fasta, ("chr2", 0, 4, "+")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAAA"


class TestFromFastaStrand:
    """Test strand handling in from_fasta."""

    def test_plus_strand_no_change(self, test_fasta):
        """Test that '+' strand returns sequence unchanged."""
        pool = from_fasta(test_fasta, ("chr1", 0, 4, "+")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_minus_strand_reverse_complement(self, test_fasta):
        """Test that '-' strand returns reverse complement."""
        # ACGT -> reverse: TGCA -> complement: ACGT
        pool = from_fasta(test_fasta, ("chr1", 0, 4, "-")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_minus_strand_asymmetric(self, test_fasta):
        """Test reverse complement with asymmetric sequence."""
        # AAAA -> reverse: AAAA -> complement: TTTT
        pool = from_fasta(test_fasta, ("chr2", 0, 4, "-")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "TTTT"


class TestFromFastaOperationName:
    """Test from_fasta operation naming."""

    def test_factory_name_is_from_fasta(self, test_fasta):
        """Test that factory name is 'from_fasta'."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+"))
        assert pool.operation.name.endswith(":from_fasta")

    def test_custom_pool_name(self, test_fasta):
        """Test custom pool name."""
        pool = from_fasta(test_fasta, ("chr1", 0, 8, "+")).named("my_pool")
        assert pool.name == "my_pool"


class TestFromFastaErrors:
    """Test error handling in from_fasta."""

    def test_invalid_chrom_raises_error(self, test_fasta):
        """Test that invalid chromosome raises KeyError."""
        with pytest.raises(KeyError):
            from_fasta(test_fasta, ("chrX", 0, 8, "+"))

    def test_invalid_strand_raises_error(self, test_fasta):
        """Test that invalid strand value raises error."""
        with pytest.raises(Exception):
            from_fasta(test_fasta, ("chr1", 0, 8, "invalid"))

    def test_invalid_coordinate_length_raises_error(self, test_fasta):
        """Test that coordinate with wrong number of elements raises error."""
        with pytest.raises(Exception):
            from_fasta(test_fasta, [("chr1", 0, 8)])  # Missing strand


class TestFromFastaBatchMode:
    """Test batch extraction of multiple regions."""

    def test_batch_extracts_multiple_sequences(self, test_fasta):
        """Test extracting multiple regions at once."""
        pool = from_fasta(
            test_fasta,
            [
                ("chr1", 0, 4, "+"),
                ("chr1", 4, 8, "+"),
                ("chr1", 8, 12, "+"),
            ],
        )
        df = pool.generate_library(num_seqs=3)
        assert len(df) == 3
        seqs = set(df["seq"])
        assert "ACGT" in seqs

    def test_batch_with_different_strands(self, test_fasta):
        """Test batch extraction with mixed strands."""
        pool = from_fasta(
            test_fasta,
            [
                ("chr2", 0, 4, "+"),
                ("chr2", 0, 4, "-"),
            ],
        )
        # Sequential mode iterates through sequences
        df = pool.generate_library(num_seqs=2)
        seqs = set(df["seq"])
        assert "AAAA" in seqs
        assert "TTTT" in seqs

    def test_batch_with_different_chroms(self, test_fasta):
        """Test batch extraction from different chromosomes."""
        pool = from_fasta(
            test_fasta,
            [
                ("chr1", 0, 4, "+"),
                ("chr2", 0, 4, "+"),
            ],
        )
        df = pool.generate_library(num_seqs=2)
        seqs = set(df["seq"])
        assert "ACGT" in seqs
        assert "AAAA" in seqs

    def test_batch_generates_correct_names(self, test_fasta):
        """Test that batch mode generates chrom:start-stop(strand) names."""
        pool = from_fasta(
            test_fasta,
            [
                ("chr1", 0, 4, "+"),
                ("chr1", 4, 8, "-"),
            ],
        )
        # Sequential mode cycles through sequences
        df = pool.generate_library(num_seqs=2)
        assert "name" in df.columns
        names = set(df["name"])
        assert "chr1:0-4(+)" in names
        assert "chr1:4-8(-)" in names

    def test_batch_with_prefix(self, test_fasta):
        """Test that prefix is prepended to names."""
        pool = from_fasta(test_fasta, [("chr1", 0, 4, "+"), ("chr1", 4, 8, "+")], prefix="myprefix")
        df = pool.generate_library(num_seqs=2)
        names = set(df["name"])
        assert "myprefix_chr1:0-4(+)" in names
        assert "myprefix_chr1:4-8(+)" in names

    def test_batch_sequential_mode(self, test_fasta):
        """Test that batch mode uses sequential mode (names cycle in order)."""
        pool = from_fasta(
            test_fasta,
            [
                ("chr1", 0, 4, "+"),
                ("chr1", 4, 8, "+"),
            ],
        )
        df = pool.generate_library(num_seqs=4)
        # Should cycle through sequences in order: 0, 1, 0, 1
        names = df["name"].tolist()
        assert names == ["chr1:0-4(+)", "chr1:4-8(+)", "chr1:0-4(+)", "chr1:4-8(+)"]


class TestFromFastaCircular:
    """Test circular genome handling."""

    def test_wrap_around_extraction(self, circular_fasta):
        """Test extraction across circular genome origin (start > stop)."""
        # Genome: ACGTACGTACGTACGTACGT (20bp)
        # Extract from position 16 to 4 (wraps around)
        # Positions 16-19: ACGT, positions 0-3: ACGT -> ACGTACGT
        pool = from_fasta(circular_fasta, ("circ", 16, 4, "+")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_wrap_around_with_reverse_strand(self, circular_fasta):
        """Test wrap-around extraction with reverse complement."""
        # Genome: ACGTACGTACGTACGTACGT (20bp)
        # Positions 16-19: ACGT, positions 0-3: ACGT -> ACGTACGT
        # Reverse complement of ACGTACGT is ACGTACGT (palindromic)
        pool = from_fasta(circular_fasta, ("circ", 16, 4, "-")).named("seq")
        df = pool.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTACGT"

    def test_batch_with_wrap_around(self, circular_fasta):
        """Test batch mode with mix of normal and wrap-around regions."""
        pool = from_fasta(
            circular_fasta,
            [
                ("circ", 0, 4, "+"),  # Normal
                ("circ", 16, 4, "+"),  # Wrap-around
            ],
        )
        # Sequential mode iterates through sequences
        df = pool.generate_library(num_seqs=2)
        seqs = set(df["seq"])
        assert "ACGT" in seqs
        assert "ACGTACGT" in seqs
