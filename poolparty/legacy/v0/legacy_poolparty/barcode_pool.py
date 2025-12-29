import random
from typing import Union, List, Tuple, Literal
from .pool import Pool


class BarcodePool(Pool):
    """A pool for generating DNA barcodes with specified constraints.
    
    Pre-generates all barcodes at construction time using a greedy random
    algorithm that satisfies distance and quality constraints. Supports
    both fixed-length and variable-length barcodes.
    
    The pool has finite states equal to num_barcodes, making it compatible
    with both random and sequential modes.
    
    Example:
        # Generate 100 8-mer barcodes with min edit distance of 3
        pool = BarcodePool(
            num_barcodes=100,
            length=8,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=3,
            seed=42
        )
        
        # Variable length barcodes
        pool = BarcodePool(
            num_barcodes=50,
            length=[6, 8, 10],
            min_edit_distance=3,
            padding_char='-'
        )
    """
    
    # DNA alphabet (hardcoded - no alphabet parameter)
    ALPHABET = ['A', 'C', 'G', 'T']
    
    def __init__(self,
                 num_barcodes: int,
                 length: Union[int, List[int]],
                 length_proportions: List[float] = None,
                 
                 # Distance constraints
                 min_edit_distance: int = None,
                 min_hamming_distance: int = None,
                 
                 # Sequence quality constraints
                 max_homopolymer: int = None,
                 gc_range: Tuple[float, float] = None,
                 
                 # External avoidance
                 avoid_sequences: List[str] = None,
                 avoid_min_distance: int = None,
                 
                 # Variable length padding
                 padding_char: str = '-',
                 padding_side: Literal['left', 'right'] = 'right',
                 
                 # Generation control
                 seed: int = None,
                 max_attempts: int = 100000,
                 
                 # Standard Pool args
                 mode: str = 'sequential',
                 max_num_states: int = None,
                 iteration_order: int = None,
                 name: str = None,
                 metadata: str = 'features'):
        """Initialize a BarcodePool.
        
        Args:
            num_barcodes: Number of barcodes to generate (must be positive)
            length: Barcode length (int) or list of allowed lengths for variable-length
            length_proportions: Proportions for each length when using variable-length.
                Must have same length as `length` list. Values are normalized to sum to 1.
                If None (default), equal distribution across all lengths.
                E.g., length=[6,8,10], length_proportions=[0.5, 0.3, 0.2] means 50% 6-mers,
                30% 8-mers, 20% 10-mers. Ignored if length is a single int.
            
            min_edit_distance: Minimum Levenshtein distance between any two barcodes.
                Recommended for most applications. Works with variable-length barcodes.
            min_hamming_distance: Minimum Hamming distance between barcodes. Only valid
                for fixed-length barcodes (raises error if used with variable length).
            
            max_homopolymer: Maximum consecutive identical characters allowed.
                E.g., max_homopolymer=3 means no runs of 4+ identical bases (no AAAA).
            gc_range: Tuple of (min_gc, max_gc) as fractions between 0 and 1.
                E.g., gc_range=(0.4, 0.6) requires 40-60% GC content.
            
            avoid_sequences: List of sequences to avoid similarity with (e.g., adapters).
                Each generated barcode must have at least avoid_min_distance from all
                sequences in this list.
            avoid_min_distance: Minimum edit distance from avoid_sequences. Required if
                avoid_sequences is provided.
            
            padding_char: Character for padding variable-length barcodes to max length.
                Default: '-'
            padding_side: Which side to pad shorter barcodes. 'left' or 'right'.
                Default: 'right'
            
            seed: Random seed for reproducible barcode generation. Default: None
            max_attempts: Maximum candidate generation attempts before raising error.
                Default: 100000
            
            mode: 'random' or 'sequential'. Default: 'sequential'
            max_num_states: Maximum states before treating as infinite
            iteration_order: Order for sequential iteration
            name: Optional pool name
            
        Raises:
            ValueError: If num_barcodes <= 0
            ValueError: If length is empty or contains non-positive values
            ValueError: If min_hamming_distance is used with variable length
            ValueError: If avoid_sequences is provided without avoid_min_distance
            ValueError: If gc_range values are not in [0, 1] or min > max
            ValueError: If length_proportions length doesn't match length list
            ValueError: If length_proportions values are not positive
            ValueError: If cannot generate num_barcodes within max_attempts
        """
        # Validate num_barcodes
        if not isinstance(num_barcodes, int) or num_barcodes <= 0:
            raise ValueError(f"num_barcodes must be a positive integer, got {num_barcodes}")
        
        # Normalize length to list
        if isinstance(length, int):
            lengths = [length]
        else:
            lengths = list(length)
        
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
                "Use min_edit_distance instead, which handles different lengths correctly."
            )
        
        # Validate and normalize length_proportions
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
        
        # Store parameters
        self.num_barcodes = num_barcodes
        self.lengths = lengths
        self.max_length = max(lengths)
        self.length_proportions = length_proportions
        self.min_edit_distance = min_edit_distance
        self.min_hamming_distance = min_hamming_distance
        self.max_homopolymer = max_homopolymer
        self.gc_range = gc_range
        self.avoid_sequences = avoid_sequences or []
        self.avoid_min_distance = avoid_min_distance
        self.padding_char = padding_char
        self.padding_side = padding_side
        self.generation_seed = seed
        self.max_attempts = max_attempts
        
        # Generate barcodes
        self.barcodes = self._generate_barcodes()
        
        # Call parent constructor
        super().__init__(
            op='barcode',
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            metadata=metadata
        )
    
    def _generate_barcodes(self) -> List[str]:
        """Generate barcodes using greedy random algorithm.
        
        Returns:
            List of padded barcode strings, sorted by unpadded length
        """
        rng = random.Random(self.generation_seed)
        
        # Track unpadded barcodes for distance calculations
        unpadded_barcodes = []
        
        # Calculate per-length quotas based on proportions
        if len(self.lengths) > 1:
            if self.length_proportions is not None:
                # Custom proportions: calculate quotas from proportions
                length_quotas = {}
                remaining = self.num_barcodes
                for i, L in enumerate(self.lengths[:-1]):
                    quota = round(self.length_proportions[i] * self.num_barcodes)
                    length_quotas[L] = quota
                    remaining -= quota
                # Last length gets the remainder to ensure exact total
                length_quotas[self.lengths[-1]] = remaining
            else:
                # Equal proportions (None = equal)
                base_quota = self.num_barcodes // len(self.lengths)
                remainder = self.num_barcodes % len(self.lengths)
                length_quotas = {}
                for i, L in enumerate(self.lengths):
                    # Distribute remainder to first few lengths based on proportions
                    length_quotas[L] = base_quota + (1 if i < remainder else 0)
            
            length_counts = {L: 0 for L in self.lengths}
        else:
            length_quotas = None
            length_counts = None
        
        attempts = 0
        while len(unpadded_barcodes) < self.num_barcodes and attempts < self.max_attempts:
            attempts += 1
            
            # Pick a length
            if length_quotas is not None:
                # Pick from lengths that haven't met quota
                available_lengths = [L for L in self.lengths if length_counts[L] < length_quotas[L]]
                if not available_lengths:
                    break  # All quotas met
                chosen_length = rng.choice(available_lengths)
            else:
                # Single length
                chosen_length = self.lengths[0]
            
            # Generate random candidate
            candidate = ''.join(rng.choice(self.ALPHABET) for _ in range(chosen_length))
            
            # Check all constraints
            if not self._passes_all_constraints(candidate, unpadded_barcodes):
                continue
            
            # Accept the candidate
            unpadded_barcodes.append(candidate)
            if length_counts is not None:
                length_counts[chosen_length] += 1
        
        # Check if we generated enough
        if len(unpadded_barcodes) < self.num_barcodes:
            raise ValueError(
                f"Could only generate {len(unpadded_barcodes)} barcodes satisfying constraints "
                f"within {self.max_attempts} attempts (requested {self.num_barcodes}). "
                "Try relaxing constraints (lower min_edit_distance, wider gc_range, etc.) "
                "or increasing max_attempts."
            )
        
        # Sort by unpadded length (shorter first), then alphabetically for stability
        unpadded_barcodes.sort(key=lambda bc: (len(bc), bc))
        
        # Pad barcodes to max length
        padded_barcodes = [self._pad_barcode(bc) for bc in unpadded_barcodes]
        
        return padded_barcodes
    
    def _passes_all_constraints(self, candidate: str, existing_barcodes: List[str]) -> bool:
        """Check if a candidate barcode passes all constraints.
        
        Args:
            candidate: The candidate barcode (unpadded)
            existing_barcodes: List of already-accepted barcodes (unpadded)
            
        Returns:
            True if candidate passes all constraints, False otherwise
        """
        # Check homopolymer constraint
        if self.max_homopolymer is not None:
            if not self._check_homopolymer(candidate):
                return False
        
        # Check GC content constraint
        if self.gc_range is not None:
            if not self._check_gc_content(candidate):
                return False
        
        # Check distance from avoid_sequences
        if self.avoid_sequences:
            for avoid_seq in self.avoid_sequences:
                dist = self._edit_distance(candidate, avoid_seq)
                if dist < self.avoid_min_distance:
                    return False
        
        # Check distance from existing barcodes
        for existing in existing_barcodes:
            # Check edit distance if specified
            if self.min_edit_distance is not None:
                if self._edit_distance(candidate, existing) < self.min_edit_distance:
                    return False
            
            # Check Hamming distance if specified (only for same-length)
            if self.min_hamming_distance is not None:
                if len(candidate) == len(existing):
                    if self._hamming_distance(candidate, existing) < self.min_hamming_distance:
                        return False
        
        return True
    
    def _check_homopolymer(self, seq: str) -> bool:
        """Check if sequence has no homopolymer runs exceeding max_homopolymer.
        
        Args:
            seq: DNA sequence to check
            
        Returns:
            True if no runs exceed max_homopolymer, False otherwise
        """
        if len(seq) <= self.max_homopolymer:
            return True
        
        run_length = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                run_length += 1
                if run_length > self.max_homopolymer:
                    return False
            else:
                run_length = 1
        
        return True
    
    def _check_gc_content(self, seq: str) -> bool:
        """Check if sequence GC content is within gc_range.
        
        Args:
            seq: DNA sequence to check
            
        Returns:
            True if GC content is within range, False otherwise
        """
        if not seq:
            return True
        
        gc_count = sum(1 for base in seq.upper() if base in ('G', 'C'))
        gc_fraction = gc_count / len(seq)
        
        min_gc, max_gc = self.gc_range
        return min_gc <= gc_fraction <= max_gc
    
    @staticmethod
    def _hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two equal-length strings.
        
        Args:
            s1: First string
            s2: Second string (must be same length as s1)
            
        Returns:
            Number of positions where characters differ
        """
        if len(s1) != len(s2):
            raise ValueError("Hamming distance requires equal-length strings")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein (edit) distance between two strings.
        
        Uses dynamic programming with O(min(m,n)) space optimization.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Minimum number of single-character edits (insert, delete, substitute)
        """
        # Ensure s1 is the shorter string for space optimization
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        m, n = len(s1), len(s2)
        
        # Previous and current row of distances
        prev_row = list(range(m + 1))
        curr_row = [0] * (m + 1)
        
        for j in range(1, n + 1):
            curr_row[0] = j
            for i in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    curr_row[i] = prev_row[i-1]
                else:
                    curr_row[i] = 1 + min(
                        prev_row[i],      # deletion
                        curr_row[i-1],    # insertion
                        prev_row[i-1]     # substitution
                    )
            prev_row, curr_row = curr_row, prev_row
        
        return prev_row[m]
    
    def _pad_barcode(self, barcode: str) -> str:
        """Pad a barcode to max_length.
        
        Args:
            barcode: Unpadded barcode string
            
        Returns:
            Padded barcode string of length max_length
        """
        if len(barcode) >= self.max_length:
            return barcode
        
        padding = self.padding_char * (self.max_length - len(barcode))
        
        if self.padding_side == 'right':
            return barcode + padding
        else:
            return padding + barcode
    
    def _calculate_num_internal_states(self) -> int:
        """BarcodePool has finite internal states = num_barcodes."""
        return self.num_barcodes
    
    def _calculate_seq_length(self) -> int:
        """BarcodePool produces sequences of max_length (after padding)."""
        return self.max_length
    
    def _compute_seq(self) -> str:
        """Return the barcode at the current state index."""
        state = self.get_state() % self.num_barcodes
        return self.barcodes[state]
    
    def get_unpadded_barcode(self, index: int = None) -> str:
        """Get an unpadded barcode by index or current state.
        
        Args:
            index: Barcode index (0 to num_barcodes-1). If None, uses current state.
            
        Returns:
            Unpadded barcode string (with padding characters stripped)
        """
        if index is None:
            index = self.get_state() % self.num_barcodes
        
        barcode = self.barcodes[index]
        return barcode.replace(self.padding_char, '')
    
    def get_all_barcodes(self, padded: bool = True) -> List[str]:
        """Get all generated barcodes.
        
        Args:
            padded: If True (default), return padded barcodes. If False, strip padding.
            
        Returns:
            List of barcode strings
        """
        if padded:
            return list(self.barcodes)
        else:
            return [bc.replace(self.padding_char, '') for bc in self.barcodes]
    
    def __repr__(self) -> str:
        lengths_str = self.lengths[0] if len(self.lengths) == 1 else self.lengths
        constraints = []
        if self.min_edit_distance:
            constraints.append(f"edit≥{self.min_edit_distance}")
        if self.min_hamming_distance:
            constraints.append(f"hamming≥{self.min_hamming_distance}")
        if self.gc_range:
            constraints.append(f"gc={self.gc_range}")
        if self.max_homopolymer:
            constraints.append(f"homopoly≤{self.max_homopolymer}")
        
        constraints_str = ", ".join(constraints) if constraints else "none"
        return f"BarcodePool(n={self.num_barcodes}, L={lengths_str}, constraints=[{constraints_str}])"

