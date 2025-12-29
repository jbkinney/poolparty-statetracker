"""Shuffle operation - generate shuffled versions of a sequence."""

from __future__ import annotations
import math
import random
from typing import List, Optional, Union, TYPE_CHECKING
from itertools import permutations
from collections import defaultdict

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


# Maximum sequence length for caching permutations (8! = 40,320 permutations)
_MAX_CACHE_LENGTH = 8


class ShuffleOp(Operation):
    """Generate shuffled versions of a parent sequence.
    
    This is a transformer operation - it has one parent pool.
    Shuffles the specified region of the input sequence.
    
    For regular shuffle: finite states = L! where L = region length
    For dinucleotide-preserving: infinite states
    """
    op_name = 'shuffle'
    
    def __init__(
        self,
        parent: 'Pool',
        start: int,
        end: int,
        preserve_dinucleotides: bool,
        shuffle_flanks: bool,
        mark_changes: bool,
        region_length: int,
        left_flank_length: int,
        right_flank_length: int,
        cached_perms: Optional[List] = None,
        mode: str = 'random',
        name: Optional[str] = None,
    ):
        """Initialize ShuffleOp."""
        self.start = start
        self.end = end
        self.preserve_dinucleotides = preserve_dinucleotides
        self.shuffle_flanks = shuffle_flanks
        self.mark_changes = mark_changes
        self._region_length = region_length
        self._left_flank_length = left_flank_length
        self._right_flank_length = right_flank_length
        self._cached_perms = cached_perms
        
        # Compute num_states
        if preserve_dinucleotides:
            num_states = -1
        elif shuffle_flanks:
            left_states = math.factorial(left_flank_length) if left_flank_length > 0 else 1
            right_states = math.factorial(right_flank_length) if right_flank_length > 0 else 1
            num_states = left_states * right_states
        else:
            num_states = math.factorial(region_length) if region_length > 0 else 1
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent],
            num_states=num_states,
            mode=mode,
            seq_length=parent.seq_length,
            name=name,
        )
    
    def _hierholzer_shuffle(self, sequence: str, seed_value: int) -> str:
        """Generate a dinucleotide-preserving shuffle using Hierholzer's algorithm."""
        if len(sequence) <= 1:
            return sequence
        
        # Build directed graph
        graph = defaultdict(list)
        for i in range(len(sequence) - 1):
            graph[sequence[i]].append(sequence[i + 1])
        
        # Calculate degrees
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for u, neighbors in graph.items():
            out_degree[u] = len(neighbors)
            for v in neighbors:
                in_degree[v] += 1
        
        # Check for Eulerian path
        all_vertices = set(in_degree.keys()) | set(out_degree.keys())
        start_vertices = sum(1 for v in all_vertices if out_degree[v] - in_degree[v] == 1)
        end_vertices = sum(1 for v in all_vertices if in_degree[v] - out_degree[v] == 1)
        
        has_path = (start_vertices == 0 and end_vertices == 0) or \
                   (start_vertices == 1 and end_vertices == 1)
        
        if not has_path:
            # Fall back to regular shuffle
            seq_list = list(sequence)
            rng = random.Random(seed_value)
            rng.shuffle(seq_list)
            return ''.join(seq_list)
        
        # Find start vertex
        start = None
        adj = {u: list(neighbors) for u, neighbors in graph.items()}
        for v in adj.keys():
            if out_degree[v] > in_degree[v]:
                start = v
                break
        if start is None:
            start = sequence[0]
        
        # Hierholzer's algorithm
        rng = random.Random(seed_value)
        stack = [start]
        path = []
        
        while stack:
            curr = stack[-1]
            if curr in adj and adj[curr]:
                next_idx = rng.randint(0, len(adj[curr]) - 1)
                next_vertex = adj[curr].pop(next_idx)
                stack.append(next_vertex)
            else:
                path.append(stack.pop())
        
        path.reverse()
        return ''.join(path)
    
    def _shuffle_region(self, seq: str, state: int) -> str:
        """Shuffle a sequence region."""
        if len(seq) == 0:
            return seq
        
        if self.preserve_dinucleotides:
            return self._hierholzer_shuffle(seq, state)
        
        if self._cached_perms is not None and len(self._cached_perms) > 0:
            perm = self._cached_perms[state % len(self._cached_perms)]
            return ''.join(perm)
        
        # Random shuffle
        seq_list = list(seq)
        rng = random.Random(state)
        rng.shuffle(seq_list)
        return ''.join(seq_list)
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute shuffled sequence.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number
        
        Returns:
            Shuffled sequence
        """
        base_seq = input_strings[0]
        
        left_flank = base_seq[:self.start]
        region = base_seq[self.start:self.end]
        right_flank = base_seq[self.end:]
        
        if self.shuffle_flanks:
            # Shuffle flanks, keep region fixed
            if self.preserve_dinucleotides:
                left_state = hash((state, 'left')) & 0x7FFFFFFF
                right_state = hash((state, 'right')) & 0x7FFFFFFF
            else:
                # Decompose state for sequential mode
                left_states = math.factorial(self._left_flank_length) if self._left_flank_length > 0 else 1
                right_states = math.factorial(self._right_flank_length) if self._right_flank_length > 0 else 1
                left_state = (state // right_states) % left_states
                right_state = state % right_states
            
            shuffled_left = self._shuffle_region(left_flank, left_state) if left_flank else ''
            shuffled_right = self._shuffle_region(right_flank, right_state) if right_flank else ''
            
            if self.mark_changes:
                shuffled_left = shuffled_left.swapcase()
                shuffled_right = shuffled_right.swapcase()
            
            return shuffled_left + region + shuffled_right
        else:
            # Shuffle region, keep flanks fixed
            shuffled_region = self._shuffle_region(region, state)
            
            if self.mark_changes:
                shuffled_region = shuffled_region.swapcase()
            
            return left_flank + shuffled_region + right_flank


def shuffle_op(
    seq: Union['Pool', str],
    start: int = 0,
    end: Optional[int] = None,
    preserve_dinucleotides: bool = False,
    shuffle_flanks: bool = False,
    mark_changes: bool = False,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Generate shuffled versions of a sequence.
    
    Shuffles the specified region of the input sequence (or shuffles the flanks
    while keeping the region fixed if shuffle_flanks=True).
    
    Args:
        seq: Input sequence (string or Pool) to shuffle
        start: Starting position of region boundary (default: 0)
        end: Ending position of region boundary (default: len(seq))
        preserve_dinucleotides: If True, preserve dinucleotide frequencies using
            Hierholzer's algorithm. Only supports mode='random'. Default: False
        shuffle_flanks: If True, keep region [start:end] fixed and shuffle flanks.
            Default: False
        mark_changes: If True, apply swapcase() to shuffled region(s). Default: False
        mode: Either 'random' or 'sequential'. Default: 'random'
        name: Optional name for this pool
    
    Returns:
        A Pool that generates shuffled sequences.
    
    Example:
        >>> pool = shuffle('ACGT', mode='sequential')
        >>> pool.operation.num_states
        24  # 4! = 24 permutations
        >>> seqs = pool.generate_library(num_complete_iterations=1)
        >>> len(seqs)
        24
    
    Raises:
        ValueError: If preserve_dinucleotides=True with mode='sequential'
        ValueError: If start < 0 or end > len(seq) or start > end
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # Validate mode for dinucleotide-preserving shuffle
    if preserve_dinucleotides and mode == 'sequential':
        raise ValueError(
            "preserve_dinucleotides=True only supports mode='random'. "
            "Cannot systematically enumerate all dinucleotide-preserving shuffles."
        )
    
    # If seq is a string, wrap it in from_seqs_op first
    if isinstance(seq, str):
        parent = from_seqs_op([seq])
        initial_seq = seq
    else:
        parent = seq
        # Get initial sequence for validation
        initial_seq = 'X' * parent.seq_length  # Placeholder for length
    
    seq_len = parent.seq_length
    
    # Validate and normalize start/end
    if end is None:
        end = seq_len
    
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end > seq_len:
        raise ValueError(f"end ({end}) cannot exceed sequence length ({seq_len})")
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    
    # Compute lengths
    region_length = end - start
    left_flank_length = start
    right_flank_length = seq_len - end
    
    # Cache permutations for small regions (if string input)
    cached_perms = None
    if not preserve_dinucleotides and isinstance(seq, str):
        if shuffle_flanks:
            pass  # Don't cache for flank mode (more complex)
        elif region_length <= _MAX_CACHE_LENGTH:
            region = seq[start:end]
            cached_perms = list(permutations(region))
    
    return Pool(
        operation=ShuffleOp(
            parent=parent,
            start=start,
            end=end,
            preserve_dinucleotides=preserve_dinucleotides,
            shuffle_flanks=shuffle_flanks,
            mark_changes=mark_changes,
            region_length=region_length,
            left_flank_length=left_flank_length,
            right_flank_length=right_flank_length,
            cached_perms=cached_perms,
            mode=mode,
            name=name,
        ),
    )
