"""BarcodePool operation - generate DNA barcodes with constraints."""

from __future__ import annotations
import random
from typing import List, Optional, Union, Tuple, Literal, TYPE_CHECKING

from ..operation import Operation
from ..utils import validate_alphabet, hamming_distance, edit_distance, max_homopolymer_length, gc_content

if TYPE_CHECKING:
    from ..pool import Pool


def _generate_barcodes(
    num_barcodes: int,
    lengths: list[int],
    length_proportions: Optional[list[float]],
    min_edit_distance: Optional[int],
    min_hamming_distance: Optional[int],
    max_homopolymer: Optional[int],
    gc_range: Optional[tuple[float, float]],
    avoid_sequences: list[str],
    avoid_min_distance: Optional[int],
    padding_char: str,
    padding_side: str,
    generation_seed: Optional[int],
    max_attempts: int,
    alphabet: list[str],
) -> tuple[list[str], int]:
    """Generate barcodes using greedy random algorithm."""
    rng = random.Random(generation_seed)
    max_length = max(lengths)
    
    # Track unpadded barcodes for distance calculations
    unpadded_barcodes = []
    
    # Calculate per-length quotas based on proportions
    if len(lengths) > 1:
        if length_proportions is not None:
            length_quotas = {}
            remaining = num_barcodes
            for i, L in enumerate(lengths[:-1]):
                quota = round(length_proportions[i] * num_barcodes)
                length_quotas[L] = quota
                remaining -= quota
            length_quotas[lengths[-1]] = remaining
        else:
            base_quota = num_barcodes // len(lengths)
            remainder = num_barcodes % len(lengths)
            length_quotas = {}
            for i, L in enumerate(lengths):
                length_quotas[L] = base_quota + (1 if i < remainder else 0)
        length_counts = {L: 0 for L in lengths}
    else:
        length_quotas = None
        length_counts = None

    attempts = 0
    while len(unpadded_barcodes) < num_barcodes and attempts < max_attempts:
        attempts += 1
        
        # Pick a length
        if length_quotas is not None:
            available_lengths = [L for L in lengths if length_counts[L] < length_quotas[L]]
            if not available_lengths:
                break
            chosen_length = rng.choice(available_lengths)
        else:
            chosen_length = lengths[0]
        
        # Generate random candidate
        candidate = ''.join(rng.choice(alphabet) for _ in range(chosen_length))
        
        # Check constraints
        if max_homopolymer is not None and max_homopolymer_length(candidate) > max_homopolymer:
            continue
        
        if gc_range is not None:
            gc_frac = gc_content(candidate)
            if not (gc_range[0] <= gc_frac <= gc_range[1]):
                continue
        
        # Check distance from avoid_sequences
        if avoid_sequences and avoid_min_distance is not None:
            skip = False
            for avoid_seq in avoid_sequences:
                if edit_distance(candidate, avoid_seq) < avoid_min_distance:
                    skip = True
                    break
            if skip:
                continue
        
        # Check distance from existing barcodes
        skip = False
        for existing in unpadded_barcodes:
            if min_edit_distance is not None:
                if edit_distance(candidate, existing) < min_edit_distance:
                    skip = True
                    break
            
            if min_hamming_distance is not None and len(candidate) == len(existing):
                if hamming_distance(candidate, existing) < min_hamming_distance:
                    skip = True
                    break
        
        if skip:
            continue
        
        # Accept the candidate
        unpadded_barcodes.append(candidate)
        if length_counts is not None:
            length_counts[chosen_length] += 1
    
    # Check if we generated enough
    if len(unpadded_barcodes) < num_barcodes:
        raise ValueError(
            f"Could only generate {len(unpadded_barcodes)} barcodes satisfying constraints "
            f"within {max_attempts} attempts (requested {num_barcodes}). "
            "Try relaxing constraints (lower min_edit_distance, wider gc_range, etc.) "
            "or increasing max_attempts."
        )
    
    # Sort by unpadded length, then alphabetically for stability
    unpadded_barcodes.sort(key=lambda bc: (len(bc), bc))
    
    # Pad barcodes to max length
    def pad_barcode(barcode: str) -> str:
        if len(barcode) >= max_length:
            return barcode
        padding = padding_char * (max_length - len(barcode))
        if padding_side == 'right':
            return barcode + padding
        else:
            return padding + barcode
    
    padded_barcodes = [pad_barcode(bc) for bc in unpadded_barcodes]
    
    return padded_barcodes


class GetBarcodesOp(Operation):
    """Generate DNA barcodes with specified constraints.
    
    This is a generator operation - it has no parent pools.
    Pre-generates all barcodes at construction time using a greedy random
    algorithm that satisfies distance and quality constraints.
    """
    op_name = 'get_barcodes_op'
    
    def __init__(
        self,
        barcodes: list[str],
        max_length: int,
        padding_char: str,
        mode: str = 'random',
        name: Optional[str] = None,
    ):
        """Initialize GetBarcodesOp.
        
        Args:
            barcodes: List of pre-generated barcode strings
            max_length: Maximum barcode length (after padding)
            padding_char: Character used for padding
            mode: Either 'random' or 'sequential' (default: 'random')
            name: Optional name for this operation
        """
        self.barcodes = barcodes
        self.padding_char = padding_char
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=len(barcodes),
            mode=mode,
            seq_length=max_length,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Return the barcode at the given state index.
        
        Args:
            input_strings: Empty list (generator has no parents)
            state: Internal state number
        
        Returns:
            Barcode sequence
        """
        index = state % len(self.barcodes)
        return self.barcodes[index]


def get_barcodes_op(
    num_barcodes: int,
    length: Union[int, list[int]],
    length_proportions: Optional[list[float]] = None,
    min_edit_distance: Optional[int] = None,
    min_hamming_distance: Optional[int] = None,
    max_homopolymer: Optional[int] = None,
    gc_range: Optional[tuple[float, float]] = None,
    avoid_sequences: Optional[list[str]] = None,
    avoid_min_distance: Optional[int] = None,
    padding_char: str = '-',
    padding_side: Literal['left', 'right'] = 'right',
    seed: Optional[int] = None,
    max_attempts: int = 100000,
    alphabet: Union[str, list[str]] = 'dna',
    mode: str = 'sequential',
    name: Optional[str] = None,
) -> 'Pool':
    """Generate DNA barcodes with specified constraints.
    
    Pre-generates all barcodes using a greedy random algorithm that satisfies
    distance and quality constraints. Supports both fixed-length and
    variable-length barcodes.
    
    Args:
        num_barcodes: Number of barcodes to generate (must be positive)
        length: Barcode length (int) or list of allowed lengths for variable-length
        length_proportions: Proportions for each length when using variable-length.
            Must have same length as `length` list. Values are normalized to sum to 1.
            If None (default), equal distribution across all lengths.
        min_edit_distance: Minimum Levenshtein distance between any two barcodes.
            Recommended for most applications. Works with variable-length barcodes.
        min_hamming_distance: Minimum Hamming distance between barcodes. Only valid
            for fixed-length barcodes.
        max_homopolymer: Maximum consecutive identical characters allowed.
            E.g., max_homopolymer=3 means no runs of 4+ identical bases.
        gc_range: Tuple of (min_gc, max_gc) as fractions between 0 and 1.
            E.g., gc_range=(0.4, 0.6) requires 40-60% GC content.
        avoid_sequences: List of sequences to avoid similarity with (e.g., adapters).
        avoid_min_distance: Minimum edit distance from avoid_sequences. Required if
            avoid_sequences is provided.
        padding_char: Character for padding variable-length barcodes. Default: '-'
        padding_side: Which side to pad shorter barcodes. Default: 'right'
        seed: Random seed for reproducible barcode generation. Default: None
        max_attempts: Maximum candidate generation attempts. Default: 100000
        alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
            or a list of single-character strings. Default: 'dna'
        mode: 'random' or 'sequential'. Default: 'sequential'
        name: Optional pool name
    
    Returns:
        A Pool that generates barcodes.
    
    Example:
        >>> pool = get_barcodes_op(
        ...     num_barcodes=100,
        ...     length=8,
        ...     min_edit_distance=3,
        ...     gc_range=(0.4, 0.6),
        ...     max_homopolymer=3,
        ...     seed=42
        ... )
        >>> pool.operation.num_states
        100
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    
    # Validate num_barcodes
    if not isinstance(num_barcodes, int) or num_barcodes <= 0:
        raise ValueError(f"num_barcodes must be a positive integer, got {num_barcodes}")
    
    # Normalize length to list
    if isinstance(length, int):
        lengths = [length]
    else:
        lengths = list(length)
    max_length = max(lengths)
    
    # Validate lengths
    if not lengths:
        raise ValueError("length must be a non-empty int or list of ints")
    for L in lengths:
        if not isinstance(L, int) or L <= 0:
            raise ValueError(f"All lengths must be positive integers, got {L}")
    
    # Check variable length constraints
    is_variable_length = len(lengths) > 1
    
    if is_variable_length and min_hamming_distance is not None:
        raise ValueError(
            "min_hamming_distance cannot be used with variable-length barcodes. "
            "Use min_edit_distance instead."
        )
    
    # Validate length_proportions
    if length_proportions is not None:
        if len(length_proportions) != len(lengths):
            raise ValueError(
                f"length_proportions length ({len(length_proportions)}) must match "
                f"length list length ({len(lengths)})"
            )
        if any(p <= 0 for p in length_proportions):
            raise ValueError("All length_proportions values must be positive")
        # Normalize to sum to 1
        total = sum(length_proportions)
        length_proportions = [p / total for p in length_proportions]
    
    # Validate gc_range
    if gc_range is not None:
        if len(gc_range) != 2:
            raise ValueError("gc_range must be a tuple of (min_gc, max_gc)")
        min_gc, max_gc = gc_range
        if not (0 <= min_gc <= 1 and 0 <= max_gc <= 1):
            raise ValueError(f"gc_range values must be in [0, 1], got {gc_range}")
        if min_gc > max_gc:
            raise ValueError(f"gc_range min ({min_gc}) cannot exceed max ({max_gc})")
    
    # Validate avoid_sequences
    if avoid_sequences is not None and avoid_min_distance is None:
        raise ValueError("avoid_min_distance is required when avoid_sequences is provided")
    
    # Validate padding_side
    if padding_side not in ('left', 'right'):
        raise ValueError(f"padding_side must be 'left' or 'right', got '{padding_side}'")
    
    # Validate and normalize alphabet
    alphabet_list = validate_alphabet(alphabet)
    
    # Generate barcodes
    barcodes = _generate_barcodes(
        num_barcodes=num_barcodes,
        lengths=lengths,
        length_proportions=length_proportions,
        min_edit_distance=min_edit_distance,
        min_hamming_distance=min_hamming_distance,
        max_homopolymer=max_homopolymer,
        gc_range=gc_range,
        avoid_sequences=avoid_sequences or [],
        avoid_min_distance=avoid_min_distance,
        padding_char=padding_char,
        padding_side=padding_side,
        generation_seed=seed,
        max_attempts=max_attempts,
        alphabet=alphabet_list,
    )
    
    return Pool(
        operation=GetBarcodesOp(
            barcodes, max_length, padding_char, mode=mode, name=name
        ),
    )
