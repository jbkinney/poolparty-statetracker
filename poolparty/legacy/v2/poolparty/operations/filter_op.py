from ..types import Union, Optional, Literal, Sequence, Real, ModeType, FilterFunc, beartype
from ..operation import Operation
from ..pool import Pool
from ..filters import (
    gc_range_filter,
    max_homopolymer_filter,
    min_hamming_dist_filter,
    min_edit_distance_filter,
    avoid_seqs_filter,
)
import numpy as np
import pandas as pd


@beartype
class FilterOp(Operation):
    """Filter sequences from a pool using user-provided filter functions."""
    design_card_keys = ['filtered_seq_idx', 'source_seq_idx']
  
    #########################################################
    # Constructor
    #########################################################
  
    def __init__(
        self,
        pool: Pool,
        num_candidate_seqs: int,
        filter_funcs: Sequence[FilterFunc] = (),
        min_num_filtered_seqs: int = 1,
        # Built-in filter parameters
        gc_range: tuple[float, float] | None = None,
        max_homopolymer: int | None = None,
        min_hamming_dist: int | None = None,
        min_edit_distance: int | None = None,
        avoid_seqs: Sequence[str] | None = None,
        avoid_min_hamming_dist: int | None = None,
        avoid_min_edit_dist: int | None = None,
        # Standard parameters
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ) -> None:
        # Validate inputs
        if num_candidate_seqs <= 0:
            raise ValueError(f"num_candidate_seqs must be positive; got {num_candidate_seqs}")
        if min_num_filtered_seqs <= 0:
            raise ValueError(f"min_num_filtered_seqs must be positive; got {min_num_filtered_seqs}")
        
        # Validate gc_range
        if gc_range is not None:
            if len(gc_range) != 2:
                raise ValueError("gc_range must be a tuple of (min_gc, max_gc)")
            min_gc, max_gc = gc_range
            if not (0 <= min_gc <= 1 and 0 <= max_gc <= 1):
                raise ValueError(f"gc_range values must be in [0, 1], got {gc_range}")
            if min_gc > max_gc:
                raise ValueError(f"gc_range min ({min_gc}) cannot exceed max ({max_gc})")
        
        # Build combined filter list: built-in filters first, then user filters
        all_filters: list[FilterFunc] = []
        
        # Add built-in filters based on parameters
        if gc_range is not None:
            all_filters.append(gc_range_filter(gc_range))
        
        if max_homopolymer is not None:
            all_filters.append(max_homopolymer_filter(max_homopolymer))
        
        if avoid_seqs is not None:
            all_filters.append(avoid_seqs_filter(
                avoid_seqs, 
                avoid_min_hamming_dist, 
                avoid_min_edit_dist
            ))
        
        if min_hamming_dist is not None:
            all_filters.append(min_hamming_dist_filter(min_hamming_dist))
        
        if min_edit_distance is not None:
            all_filters.append(min_edit_distance_filter(min_edit_distance))
        
        # Add user-provided filters
        all_filters.extend(filter_funcs)
        
        # Require at least one filter
        if len(all_filters) == 0:
            raise ValueError(
                "At least one filter must be specified: either via filter_funcs "
                "or built-in filter parameters (gc_range, max_homopolymer, etc.)"
            )
        self.filter_funcs = all_filters
        self.num_candidate_seqs = num_candidate_seqs
        self.min_num_filtered_seqs = min_num_filtered_seqs
        
        # Generate candidate sequences from input pool
        candidate_df = pool.generate_library(num_seqs=num_candidate_seqs)
        
        # Apply filters to each candidate sequence
        filtered_rows = []
        filtered_seqs: list[str] = []
        
        for idx, row in candidate_df.iterrows():
            seq = row['seq']
            
            # Check if sequence passes all filters
            passes_all = True
            for filter_func in self.filter_funcs:
                # Pass filtered_seqs (may be empty list initially)
                if not filter_func(seq, filtered_seqs if filtered_seqs else None):
                    passes_all = False
                    break
            
            if passes_all:
                # Add tracking columns
                row_dict = row.to_dict()
                row_dict['filtered_seq_idx'] = len(filtered_rows)
                row_dict['source_seq_idx'] = idx
                filtered_rows.append(row_dict)
                filtered_seqs.append(seq)
        
        # Check minimum requirement
        if len(filtered_rows) < min_num_filtered_seqs:
            raise ValueError(
                f"Only {len(filtered_rows)} sequences passed filters, "
                f"but min_num_filtered_seqs={min_num_filtered_seqs} required"
            )
        
        # Build master DataFrame from filtered sequences
        self._master_results_df = pd.DataFrame(filtered_rows)
        
        # Update design card keys to include all columns from master DataFrame
        self.design_card_keys = list(self._master_results_df.columns)
        
        # Compute seq_length: fixed if all same length, None if variable
        seq_lengths = set(len(s) for s in self._master_results_df['seq'])
        seq_length = seq_lengths.pop() if len(seq_lengths) == 1 else None
    
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=len(self._master_results_df),
            mode=mode,
            seq_length=seq_length,
            name=name,
            design_card_keys=self.design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int
    ) -> dict:
        """Return a dict with seq and design card data for one sequence."""
        total_seqs = len(self._master_results_df)
        
        if self.mode == 'sequential':
            index = sequential_state % total_seqs
        elif self.mode == 'random':
            index = self.rng.integers(0, total_seqs)
        else:
            raise ValueError(f"{self.mode=} is not 'sequential' or 'random'.")
        
        # Get row from master DataFrame
        row = self._master_results_df.iloc[index]
        return row.to_dict()


#########################################################
# Public factory function
#########################################################

@beartype
def filter(
    pool: Pool,
    num_candidate_seqs: int,
    filter_funcs: Sequence[FilterFunc] = (),
    min_num_filtered_seqs: int = 1,
    # Built-in filter parameters
    gc_range: tuple[float, float] | None = None,
    max_homopolymer: int | None = None,
    min_hamming_dist: int | None = None,
    min_edit_distance: int | None = None,
    avoid_seqs: Sequence[str] | None = None,
    avoid_min_hamming_dist: int | None = None,
    avoid_min_edit_dist: int | None = None,
    # Standard parameters
    mode: ModeType = 'random',
    name: str = 'filter',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool by filtering sequences from another pool.
    
    Args:
        pool: Input pool to filter sequences from
        num_candidate_seqs: Number of sequences to generate from input pool
        filter_funcs: List of filter functions. Each function takes
            (seq: str, filtered_seqs: list[str] | None) -> bool.
            A sequence passes if ALL filter functions return True.
            The filtered_seqs argument receives the cumulative list of
            already-accepted sequences, enabling uniqueness/diversity checks.
        min_num_filtered_seqs: Minimum required passing sequences (error if not met)
        gc_range: Optional tuple of (min_gc, max_gc) as fractions between 0 and 1.
            E.g., gc_range=(0.4, 0.6) requires 40-60% GC content.
        max_homopolymer: Optional maximum allowed consecutive identical characters.
            E.g., max_homopolymer=3 means no runs of 4+ identical bases.
        min_hamming_dist: Optional minimum Hamming distance between accepted sequences.
            Only compares equal-length sequences.
        min_edit_distance: Optional minimum edit (Levenshtein) distance between
            accepted sequences.
        avoid_seqs: Optional list of sequences to avoid similarity with (e.g., adapters).
        avoid_min_hamming_dist: If specified with avoid_seqs, requires Hamming distance
            >= this from all equal-length avoid_seqs.
        avoid_min_edit_dist: If specified with avoid_seqs, requires edit distance
            >= this from all avoid_seqs.
        mode: 'random' or 'sequential'
        name: Name for this operation
        design_card_keys: Which design card keys to include
    
    Returns:
        A Pool that serves sequences from the filtered set
    
    Example:
        >>> from poolparty import get_kmers, filter
        >>> 
        >>> # Filter using custom function
        >>> def has_atg(seq, filtered_seqs=None):
        ...     return 'ATG' in seq
        >>> pool = get_kmers(length=10, alphabet='dna')
        >>> filtered = filter(pool, num_candidate_seqs=1000, filter_funcs=[has_atg])
        >>> 
        >>> # Filter using built-in filters
        >>> filtered = filter(
        ...     pool, 
        ...     num_candidate_seqs=1000,
        ...     gc_range=(0.4, 0.6),
        ...     max_homopolymer=3,
        ...     min_edit_distance=3
        ... )
    """
    return Pool(
        operation=FilterOp(
            pool=pool,
            num_candidate_seqs=num_candidate_seqs,
            filter_funcs=filter_funcs,
            min_num_filtered_seqs=min_num_filtered_seqs,
            gc_range=gc_range,
            max_homopolymer=max_homopolymer,
            min_hamming_dist=min_hamming_dist,
            min_edit_distance=min_edit_distance,
            avoid_seqs=avoid_seqs,
            avoid_min_hamming_dist=avoid_min_hamming_dist,
            avoid_min_edit_dist=avoid_min_edit_dist,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        ),
    )

