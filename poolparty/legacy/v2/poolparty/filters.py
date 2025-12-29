"""Filter factory functions for FilterOp."""
from collections.abc import Sequence
from .types import FilterFunc, beartype
from .utils import gc_content, max_homopolymer_length, hamming_distance, edit_distance


@beartype
def gc_range_filter(gc_range: tuple[float, float]) -> FilterFunc:
    """Return filter that passes sequences with GC content within (min_gc, max_gc)."""
    min_gc, max_gc = gc_range
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        gc = gc_content(seq)
        return min_gc <= gc <= max_gc
    return _filter


@beartype
def max_homopolymer_filter(max_homopolymer: int) -> FilterFunc:
    """Return filter that passes sequences with max homopolymer run <= limit."""
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        return max_homopolymer_length(seq) <= max_homopolymer
    return _filter


@beartype
def min_hamming_dist_filter(min_dist: int) -> FilterFunc:
    """Return filter requiring Hamming distance >= min_dist from all filtered_seqs."""
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        if filtered_seqs is None:
            return True
        for existing in filtered_seqs:
            if len(seq) == len(existing):
                if hamming_distance(seq, existing) < min_dist:
                    return False
        return True
    return _filter


@beartype
def min_edit_distance_filter(min_dist: int) -> FilterFunc:
    """Return filter requiring edit distance >= min_dist from all filtered_seqs."""
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        if filtered_seqs is None:
            return True
        for existing in filtered_seqs:
            if edit_distance(seq, existing) < min_dist:
                return False
        return True
    return _filter


@beartype
def avoid_seqs_filter(
    avoid_seqs: Sequence[str],
    avoid_min_hamming_dist: int | None = None,
    avoid_min_edit_dist: int | None = None,
) -> FilterFunc:
    """Return filter rejecting sequences too similar to avoid_seqs (exact match if no distances specified)."""
    avoid_seqs_list = list(avoid_seqs)
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        for avoid_seq in avoid_seqs_list:
            if avoid_min_hamming_dist is not None:
                if len(seq) == len(avoid_seq):
                    if hamming_distance(seq, avoid_seq) < avoid_min_hamming_dist:
                        return False
            if avoid_min_edit_dist is not None:
                if edit_distance(seq, avoid_seq) < avoid_min_edit_dist:
                    return False
            if avoid_min_hamming_dist is None and avoid_min_edit_dist is None:
                if seq == avoid_seq:
                    return False
        return True
    return _filter


@beartype
def avoid_subseqs_filter(avoid_subseqs: Sequence[str]) -> FilterFunc:
    """Return filter rejecting sequences containing any of the specified subsequences."""
    avoid_subseqs_list = list(avoid_subseqs)
    def _filter(seq: str, filtered_seqs: list[str] | None = None) -> bool:
        for subseq in avoid_subseqs_list:
            if subseq in seq:
                return False
        return True
    return _filter
