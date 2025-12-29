import math
import random
import warnings
from typing import Union, Dict, Any
from itertools import permutations
from collections import defaultdict
from .pool import Pool


# Maximum sequence length for caching permutations (8! = 40,320 permutations)
_MAX_CACHE_LENGTH = 8


class ShufflePool(Pool):
    """A class for generating shuffled versions of an input sequence.
    
    Generates shuffled permutations of the input sequence (or a specified region).
    Supports both random and sequential iteration through shuffled variants.
    
    **Regular shuffle** (preserve_dinucleotides=False, default):
        Has finite states equal to L! where L = length of region to shuffle.
        
        - Sequential mode: Systematically iterates through all L! permutations.
          Only practical for short sequences (L <= 9 or so, since 10! > 3 million).
        - Random mode: Uses seeded random shuffles. Each state maps deterministically
          to one shuffle, but does not enumerate all permutations systematically.
        - Caching: For string inputs with region length <= 8, permutations are cached.
    
    **Dinucleotide-preserving shuffle** (preserve_dinucleotides=True):
        Preserves the counts of adjacent character pairs (dinucleotides) using
        Hierholzer's algorithm for finding Eulerian paths.
        
        - Only supports random mode (cannot enumerate all valid shuffles).
        - Has infinite states (returns float('inf')).
        - Falls back to regular shuffle with a warning if no Eulerian path exists.
        - Note: Only dinucleotides within the shuffled region are preserved;
          boundary dinucleotides (between flanks and region) may change.
    
    **Flank shuffle mode** (shuffle_flanks=True):
        Instead of shuffling the region [start:end], shuffles the flanking sequences
        while keeping the region [start:end] fixed. The left flank [0:start] and
        right flank [end:] are shuffled independently.
        
        - Useful for controls where you want to preserve a central motif/feature.
        - Each flank is shuffled independently (not pooled together).
        - State space is L_left! × L_right! for regular shuffle.
        - With preserve_dinucleotides=True, each flank is shuffled independently
          using Hierholzer's algorithm.
    
    Example:
        # Enumerate all 24 permutations of "ACGT"
        pool = ShufflePool("ACGT", mode='sequential')
        all_shuffles = pool.generate_seqs(num_complete_iterations=1)
        
        # Random shuffles of a longer sequence
        pool = ShufflePool("ACGTACGTACGT", mode='random')
        shuffles = pool.generate_seqs(num_seqs=100, seed=42)
        
        # Shuffle only a region (positions 4-8)
        pool = ShufflePool("ACGTACGTACGT", start=4, end=8)
        shuffles = pool.generate_seqs(num_seqs=10, seed=42)
        
        # Keep region [4:8] fixed, shuffle the flanks independently
        pool = ShufflePool("ACGTACGTACGT", start=4, end=8, shuffle_flanks=True)
        shuffles = pool.generate_seqs(num_seqs=10, seed=42)
        
        # Dinucleotide-preserving shuffles
        pool = ShufflePool("ACGTACGTACGT", preserve_dinucleotides=True)
        shuffles = pool.generate_seqs(num_seqs=100, seed=42)
    """
    
    def __init__(self, 
                 seq: Union[Pool, str],
                 start: int = 0,
                 end: int = None,
                 preserve_dinucleotides: bool = False,
                 shuffle_flanks: bool = False,
                 mark_changes: bool = False,
                 max_num_states: int = None, 
                 mode: str = 'random', 
                 iteration_order: int | None = None, 
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a ShufflePool.
        
        Args:
            seq: Input sequence to shuffle. Can be:
                - str: A sequence string to shuffle
                - Pool: A Pool whose current sequence will be shuffled (transformer pattern)
            start: Starting position of region boundary (inclusive, 0-indexed).
                Default: 0 (beginning of sequence)
            end: Ending position of region boundary (exclusive, like Python slicing).
                Default: None (end of sequence)
            preserve_dinucleotides: If True, use Hierholzer's algorithm to preserve
                adjacent character pair (dinucleotide) frequencies within the shuffled
                region(s). Only supports mode='random'. If False (default), use regular shuffle.
            shuffle_flanks: If True, keep the region [start:end] fixed and shuffle the
                flanking sequences [0:start] and [end:] independently. If False (default),
                shuffle the region [start:end] and keep flanks fixed.
            mark_changes: If True, apply swapcase() to the shuffled region(s) for
                visualization. Default: False
            max_num_states: Maximum number of states before treating as infinite.
                Default: uses global DEFAULT_MAX_NUM_STATES (1,000,000)
            mode: Either 'random' or 'sequential'. Default: 'random'
                - 'sequential': Enumerate all permutations in order (only for regular shuffle)
                - 'random': Use seeded random shuffles
            iteration_order: Order for sequential iteration when combined with other
                sequential pools. Default: auto-assigned based on creation order
            name: Optional name for this pool
            metadata: Metadata level for design cards. Default: 'features'
        
        Raises:
            ValueError: If preserve_dinucleotides=True with mode='sequential'
            ValueError: If start < 0 or end > len(seq) or start > end
        
        Note:
            Sequential mode is only practical for short sequences. For L=10,
            there are 3,628,800 permutations. For L=13, there are 6+ billion.
            When shuffle_flanks=True with mode='sequential', the state space is
            L_left! × L_right!, which can be very large.
        """
        # Validate mode for dinucleotide-preserving shuffle
        if preserve_dinucleotides and mode == 'sequential':
            raise ValueError(
                "preserve_dinucleotides=True only supports mode='random'. "
                "Cannot systematically enumerate all dinucleotide-preserving shuffles."
            )
        
        self.preserve_dinucleotides = preserve_dinucleotides
        self.shuffle_flanks = shuffle_flanks
        self.mark_changes = mark_changes
        
        # Instance-level warning flag for Eulerian path fallback
        self._instance_warning_issued = False
        # Separate warning flags for left and right flanks (for shuffle_flanks mode)
        self._left_flank_warning_issued = False
        self._right_flank_warning_issued = False
        
        # Track if parent is a Pool (affects caching strategy)
        self._parent_is_pool = isinstance(seq, Pool)
        
        # Get initial sequence for validation
        if self._parent_is_pool:
            initial_seq = seq.seq
        else:
            initial_seq = seq
        
        # Validate and normalize start/end
        self.start = start
        self.end = end if end is not None else len(initial_seq)
        
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end > len(initial_seq):
            raise ValueError(
                f"end ({self.end}) cannot exceed sequence length ({len(initial_seq)})"
            )
        if self.start > self.end:
            raise ValueError(
                f"start ({self.start}) must be <= end ({self.end})"
            )
        
        # Store lengths for state calculations
        self._region_length = self.end - self.start
        self._left_flank_length = self.start
        self._right_flank_length = len(initial_seq) - self.end
        
        # Warn if shuffle_flanks=True with sequential mode (exponential state space)
        if shuffle_flanks and mode == 'sequential':
            total_flank_states = (
                math.factorial(self._left_flank_length) * 
                math.factorial(self._right_flank_length)
            )
            if total_flank_states > 1000000:
                warnings.warn(
                    f"shuffle_flanks=True with mode='sequential' has {total_flank_states:,} states "
                    f"(left_flank={self._left_flank_length}! × right_flank={self._right_flank_length}!). "
                    "Consider using mode='random' for large flank lengths.",
                    UserWarning
                )
        
        # Cache permutations for string parents with small regions (regular shuffle only)
        self._cached_perms = None
        self._cached_left_perms = None
        self._cached_right_perms = None
        
        if not preserve_dinucleotides and not self._parent_is_pool:
            if shuffle_flanks:
                # Cache flank permutations if small enough
                if self._left_flank_length <= _MAX_CACHE_LENGTH and self._left_flank_length > 0:
                    left_flank = initial_seq[:self.start]
                    self._cached_left_perms = list(permutations(left_flank))
                if self._right_flank_length <= _MAX_CACHE_LENGTH and self._right_flank_length > 0:
                    right_flank = initial_seq[self.end:]
                    self._cached_right_perms = list(permutations(right_flank))
            else:
                # Cache region permutations if small enough
                if self._region_length <= _MAX_CACHE_LENGTH:
                    region = initial_seq[self.start:self.end]
                    self._cached_perms = list(permutations(region))
        
        super().__init__(
            parents=(seq,), 
            op='shuffle', 
            max_num_states=max_num_states, 
            mode=mode, 
            iteration_order=iteration_order, 
            name=name,
            metadata=metadata
        )
    
    def _calculate_num_internal_states(self) -> int | float:
        """Calculate the number of internal states.
        
        For regular shuffle: L! where L = length of region to shuffle
        For shuffle_flanks: L_left! × L_right!
        For dinucleotide-preserving: float('inf')
        """
        if self.preserve_dinucleotides:
            # Cannot enumerate dinucleotide-preserving shuffles
            return float('inf')
        
        if self.shuffle_flanks:
            # State space is the product of left and right flank permutations
            left_states = math.factorial(self._left_flank_length)
            right_states = math.factorial(self._right_flank_length)
            return left_states * right_states
        
        return math.factorial(self._region_length)
    
    def _calculate_seq_length(self) -> int:
        """Shuffling doesn't change length - inherit from parent."""
        parent = self.parents[0]
        if isinstance(parent, Pool):
            return parent.seq_length
        return len(parent)
    
    # =========================================================================
    # Dinucleotide-preserving shuffle methods (Hierholzer's algorithm)
    # =========================================================================
    
    def _build_digraph(self, sequence: str) -> dict:
        """Build a directed graph from the sequence.
        
        Nodes are characters, edges represent adjacencies.
        
        Args:
            sequence: String to build graph from
            
        Returns:
            Dictionary mapping each character to list of its successors
        """
        graph = defaultdict(list)
        for i in range(len(sequence) - 1):
            graph[sequence[i]].append(sequence[i + 1])
        return dict(graph)
    
    def _has_eulerian_path(self, graph: dict, sequence: str) -> bool:
        """Check if the directed graph has an Eulerian path.
        
        An Eulerian path exists if:
        - All vertices have equal in-degree and out-degree, OR
        - Exactly one vertex has out-degree - in-degree = 1 (start)
          and exactly one vertex has in-degree - out-degree = 1 (end)
          and all other vertices have equal in-degree and out-degree
        
        Args:
            graph: Adjacency list representation
            sequence: Original sequence string
            
        Returns:
            True if Eulerian path exists, False otherwise
        """
        if len(sequence) <= 1:
            return True
        
        # Calculate in-degrees and out-degrees
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for u, neighbors in graph.items():
            out_degree[u] += len(neighbors)
            for v in neighbors:
                in_degree[v] += 1
        
        # Get all vertices (including sinks with no outgoing edges)
        all_vertices = set(in_degree.keys()) | set(out_degree.keys())
        
        # Count vertices with degree imbalance
        start_vertices = 0  # out > in
        end_vertices = 0    # in > out
        
        for v in all_vertices:
            diff = out_degree[v] - in_degree[v]
            if diff == 1:
                start_vertices += 1
            elif diff == -1:
                end_vertices += 1
            elif diff != 0:
                return False
        
        # Valid if all balanced or exactly one start and one end
        return (start_vertices == 0 and end_vertices == 0) or \
               (start_vertices == 1 and end_vertices == 1)
    
    def _hierholzer_shuffle_with_flag(self, sequence: str, seed_value: int,
                                       warning_flag_name: str = '_instance_warning_issued') -> str:
        """Generate a dinucleotide-preserving shuffle using Hierholzer's algorithm.
        
        Uses Hierholzer's algorithm with seeded randomness to find an Eulerian path
        through the directed graph representation of the sequence.
        
        Args:
            sequence: String to shuffle
            seed_value: Seed for random number generator
            warning_flag_name: Name of instance attribute to track if warning was issued
            
        Returns:
            Shuffled string preserving dinucleotide counts
        """
        if len(sequence) <= 1:
            return sequence
        
        # Build the directed graph
        graph = self._build_digraph(sequence)
        
        # Check if Eulerian path exists
        if not self._has_eulerian_path(graph, sequence):
            # Fall back to regular shuffle
            if not getattr(self, warning_flag_name, False):
                warnings.warn(
                    f"Sequence does not have an Eulerian path. "
                    "Falling back to regular shuffle that preserves only monomer counts.",
                    UserWarning
                )
                setattr(self, warning_flag_name, True)
            
            # Regular shuffle with seed
            seq_list = list(sequence)
            rng = random.Random(seed_value)
            rng.shuffle(seq_list)
            return ''.join(seq_list)
        
        # Create mutable adjacency list for edge removal
        adj = defaultdict(list)
        for u, neighbors in graph.items():
            adj[u] = list(neighbors)
        
        # Calculate in/out degrees to find start vertex
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for u, neighbors in adj.items():
            out_degree[u] = len(neighbors)
            for v in neighbors:
                in_degree[v] += 1
        
        # Find starting vertex (vertex with out > in, or any vertex with edges)
        start = None
        for v in adj.keys():
            if out_degree[v] > in_degree[v]:
                start = v
                break
        if start is None:
            start = sequence[0]  # Start from first character if all balanced
        
        # Hierholzer's algorithm with randomization
        rng = random.Random(seed_value)
        stack = [start]
        path = []
        
        while stack:
            curr = stack[-1]
            if curr in adj and adj[curr]:
                # Randomly select next edge
                next_idx = rng.randint(0, len(adj[curr]) - 1)
                next_vertex = adj[curr].pop(next_idx)
                stack.append(next_vertex)
            else:
                path.append(stack.pop())
        
        # Path is in reverse order
        path.reverse()
        
        return ''.join(path)
    
    def _hierholzer_shuffle(self, sequence: str, seed_value: int) -> str:
        """Generate a dinucleotide-preserving shuffle using Hierholzer's algorithm.
        
        Wrapper around _hierholzer_shuffle_with_flag using the default warning flag.
        
        Args:
            sequence: String to shuffle
            seed_value: Seed for random number generator
            
        Returns:
            Shuffled string preserving dinucleotide counts
        """
        return self._hierholzer_shuffle_with_flag(sequence, seed_value, '_instance_warning_issued')
    
    # =========================================================================
    # Core sequence computation
    # =========================================================================
    
    def _shuffle_single_region(self, seq: str, state: int, 
                                cached_perms: list = None,
                                warning_flag_name: str = '_instance_warning_issued') -> str:
        """Shuffle a single sequence region.
        
        Helper method to shuffle a region using the configured method.
        
        Args:
            seq: The sequence to shuffle
            state: The state value for deterministic shuffling
            cached_perms: Pre-computed permutations (or None)
            warning_flag_name: Name of instance warning flag for Eulerian path fallback
        
        Returns:
            Shuffled sequence
        """
        if len(seq) == 0:
            return seq
        
        if self.preserve_dinucleotides:
            # Dinucleotide-preserving shuffle using Hierholzer's algorithm
            return self._hierholzer_shuffle_with_flag(seq, state, warning_flag_name)
        
        elif self.mode == 'sequential':
            # Sequential mode: enumerate all L! permutations deterministically
            if cached_perms is not None:
                perm = cached_perms[state % len(cached_perms)]
            else:
                all_perms = list(permutations(seq))
                if len(all_perms) == 0:
                    return seq
                perm = all_perms[state % len(all_perms)]
            return ''.join(perm)
        
        else:
            # Random mode: seeded shuffle (deterministic per state)
            seq_list = list(seq)
            rng = random.Random(state)
            rng.shuffle(seq_list)
            return ''.join(seq_list)
    
    def _compute_seq(self) -> str:
        """Compute shuffled sequence based on current state and settings."""
        if isinstance(self.parents[0], str):
            base_seq = self.parents[0]
        else:
            base_seq = self.parents[0].seq
        
        # Extract flanks and region
        left_flank = base_seq[:self.start]
        region = base_seq[self.start:self.end]
        right_flank = base_seq[self.end:]
        
        if self.shuffle_flanks:
            # Shuffle flanks independently, keep region fixed
            state = self.get_state()
            
            if self.mode == 'sequential' and not self.preserve_dinucleotides:
                # Decompose state for sequential mode: left varies slower, right varies faster
                # state = left_state * right_states + right_state
                left_states = math.factorial(self._left_flank_length) if self._left_flank_length > 0 else 1
                right_states = math.factorial(self._right_flank_length) if self._right_flank_length > 0 else 1
                
                left_state = (state // right_states) % left_states
                right_state = state % right_states
            else:
                # Random mode: use different seeds for each flank
                # Use a simple hash to get independent but deterministic seeds
                left_state = hash((state, 'left')) & 0x7FFFFFFF
                right_state = hash((state, 'right')) & 0x7FFFFFFF
            
            # Shuffle each flank
            shuffled_left = self._shuffle_single_region(
                left_flank, left_state, 
                self._cached_left_perms,
                '_left_flank_warning_issued'
            )
            shuffled_right = self._shuffle_single_region(
                right_flank, right_state,
                self._cached_right_perms,
                '_right_flank_warning_issued'
            )
            
            # Apply mark_changes if requested (to the shuffled flanks)
            if self.mark_changes:
                shuffled_left = shuffled_left.swapcase()
                shuffled_right = shuffled_right.swapcase()
            
            return shuffled_left + region + shuffled_right
        
        else:
            # Standard mode: shuffle the region, keep flanks fixed
            shuffled_region = self._shuffle_single_region(
                region, self.get_state(),
                self._cached_perms,
                '_instance_warning_issued'
            )
            
            # Apply mark_changes if requested
            if self.mark_changes:
                shuffled_region = shuffled_region.swapcase()
            
            return left_flank + shuffled_region + right_flank
    
    # =========================================================================
    # Design Card Metadata
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this ShufflePool at the current state.
        
        Extends base Pool metadata with shuffle configuration information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + start, end, shuffle_mode (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add ShufflePool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['start'] = self.start
            metadata['end'] = self.end
            metadata['shuffle_mode'] = 'flanks' if self.shuffle_flanks else 'region'
        
        return metadata
    
    def __repr__(self) -> str:
        parent_seq = self.parents[0].seq if isinstance(self.parents[0], Pool) else self.parents[0]
        
        # Build optional parameter strings
        extra_parts = []
        
        # Show start/end if not defaults (0 and len(seq))
        seq_len = len(parent_seq)
        if self.start != 0 or self.end != seq_len:
            extra_parts.append(f"start={self.start}, end={self.end}")
        
        if self.shuffle_flanks:
            extra_parts.append("shuffle_flanks=True")
        
        if self.preserve_dinucleotides:
            extra_parts.append("preserve_dinucleotides=True")
        
        if self.mark_changes:
            extra_parts.append("mark_changes=True")
        
        extra_str = ", " + ", ".join(extra_parts) if extra_parts else ""
        return f"ShufflePool(seq={parent_seq}{extra_str})"
