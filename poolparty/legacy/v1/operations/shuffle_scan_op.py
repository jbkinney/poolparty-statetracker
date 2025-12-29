"""ShuffleScan operation - scan shuffled windows across a sequence."""

from __future__ import annotations
import random
from collections import defaultdict
from typing import List, Optional, Union, TYPE_CHECKING

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


def _regular_shuffle(seq: str, seed: int) -> str:
    """Regular random shuffle."""
    seq_list = list(seq)
    rng = random.Random(seed)
    rng.shuffle(seq_list)
    return ''.join(seq_list)


def _dinucleotide_shuffle(seq: str, seed: int) -> str:
    """Dinucleotide-preserving shuffle using Hierholzer's algorithm."""
    if len(seq) <= 1:
        return seq
    
    # Build directed graph
    graph = defaultdict(list)
    for i in range(len(seq) - 1):
        graph[seq[i]].append(seq[i + 1])
    
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
        return _regular_shuffle(seq, seed)
    
    # Find start vertex
    start = None
    adj = {u: list(neighbors) for u, neighbors in graph.items()}
    for v in adj.keys():
        if out_degree[v] > in_degree[v]:
            start = v
            break
    if start is None:
        start = seq[0]
    
    # Hierholzer's algorithm
    rng = random.Random(seed)
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


class ShuffleScanOp(Operation):
    """Scan shuffled windows across a parent sequence.
    
    This is a transformer operation - it has one parent pool.
    Systematically shuffles windows at specified positions.
    """
    op_name = 'shuffle_scan'
    
    def __init__(
        self,
        parent: 'Pool',
        shuffle_size: int,
        positions: list[int],
        num_shuffles: int,
        shuffle_seed: Optional[int],
        preserve_dinucleotides: bool,
        mark_changes: bool,
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize ShuffleScanOp."""
        self.shuffle_size = shuffle_size
        self.positions = positions
        self.num_shuffles = num_shuffles
        self.shuffle_seed = shuffle_seed
        self.preserve_dinucleotides = preserve_dinucleotides
        self.mark_changes = mark_changes
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent],
            num_states=len(positions) * num_shuffles,
            mode=mode,
            seq_length=parent.seq_length,
            name=name,
        )
    
    def _get_shuffle_seed(self, shuffle_idx: int) -> int:
        """Get the seed for a specific shuffle index."""
        if self.shuffle_seed is None:
            return shuffle_idx
        return hash((shuffle_idx, self.shuffle_seed)) & 0x7FFFFFFF
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute sequence with shuffled window.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number
        
        Returns:
            Sequence with shuffle
        """
        base_seq = input_strings[0]
        
        # Decompose state into position and shuffle indices
        idx = state % self.num_states
        pos_idx = idx // self.num_shuffles
        shuffle_idx = idx % self.num_shuffles
        
        pos = self.positions[pos_idx]
        
        # Extract and shuffle window
        window = base_seq[pos:pos + self.shuffle_size]
        seed = self._get_shuffle_seed(shuffle_idx) + hash(pos) & 0x7FFFFFFF
        
        if self.preserve_dinucleotides:
            shuffled = _dinucleotide_shuffle(window, seed)
        else:
            shuffled = _regular_shuffle(window, seed)
        
        if self.mark_changes:
            shuffled = shuffled.swapcase()
        
        return base_seq[:pos] + shuffled + base_seq[pos + self.shuffle_size:]


def shuffle_scan_op(
    seq: Union['Pool', str],
    shuffle_size: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[list[int]] = None,
    num_shuffles: int = 1,
    shuffle_seed: Optional[int] = None,
    preserve_dinucleotides: bool = False,
    mark_changes: bool = True,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Scan shuffled windows across a sequence.
    
    Performs scanning mutagenesis by randomly shuffling characters within a sliding
    window at specified positions.
    
    Args:
        seq: Input sequence (string or Pool) to scan across
        shuffle_size: Size of the window to shuffle
        start: Starting position for first shuffle (default: 0)
        end: Ending position (default: len(seq))
        step_size: Step between adjacent shuffles (default: 1)
        positions: List of explicit positions to shuffle at
        num_shuffles: Number of shuffle permutations per position (default: 1)
        shuffle_seed: Seed for shuffle permutation generation. Default: None
        preserve_dinucleotides: If True, preserve dinucleotide frequencies. Default: False
        mark_changes: If True, swapcase() shuffled region. Default: True
        mode: Either 'random' or 'sequential'. Default: 'random'
        name: Optional name for this pool
    
    Returns:
        A Pool that generates shuffle-scanned variants.
    
    Example:
        >>> pool = shuffle_scan('ACGTACGT', shuffle_size=4, mode='sequential')
        >>> pool.operation.num_states
        5  # Positions 0-4
    
    Raises:
        ValueError: If shuffle_size > sequence length
        ValueError: If num_shuffles < 1
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # If seq is a string, wrap it in from_seqs_op first
    if isinstance(seq, str):
        parent = from_seqs_op([seq])
        seq_len = len(seq)
    else:
        parent = seq
        seq_len = parent.seq_length
    
    # Validate parameters
    if shuffle_size <= 0:
        raise ValueError(f"shuffle_size must be > 0, got {shuffle_size}")
    if shuffle_size > seq_len:
        raise ValueError(
            f"shuffle_size ({shuffle_size}) cannot exceed sequence length ({seq_len})"
        )
    if num_shuffles < 1:
        raise ValueError(f"num_shuffles must be >= 1, got {num_shuffles}")
    
    # Determine positions
    range_params_provided = any(p is not None for p in [start, end, step_size])
    position_params_provided = positions is not None
    
    if range_params_provided and position_params_provided:
        raise ValueError(
            "Cannot specify both range-based parameters (start/end/step_size) "
            "and position-based parameters (positions)."
        )
    
    if position_params_provided:
        if not positions:
            raise ValueError("positions must be a non-empty list")
        for pos in positions:
            if pos < 0 or pos + shuffle_size > seq_len:
                raise ValueError(f"Position {pos} is invalid")
        computed_positions = list(positions)
    else:
        start_val = start if start is not None else 0
        end_val = end if end is not None else seq_len
        step_val = step_size if step_size is not None else 1
        
        end_val = min(end_val, seq_len)
        computed_positions = list(range(start_val, end_val - shuffle_size + 1, step_val))
        
        if len(computed_positions) == 0:
            raise ValueError("No valid shuffle positions for given parameters")
    
    return Pool(
        operation=ShuffleScanOp(
            parent=parent,
            shuffle_size=shuffle_size,
            positions=computed_positions,
            num_shuffles=num_shuffles,
            shuffle_seed=shuffle_seed,
            preserve_dinucleotides=preserve_dinucleotides,
            mark_changes=mark_changes,
            mode=mode,
            name=name,
        ),
    )
