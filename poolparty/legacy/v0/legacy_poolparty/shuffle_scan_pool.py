from typing import Union, List, Dict, Any
import random
from .pool import Pool
from collections import defaultdict
import warnings

class ShuffleScanPool(Pool):
    """ShuffleScanPool with support for explicit position lists and multiple shuffle variants.

    Performs scanning mutagenesis by randomly shuffling characters within a sliding window
    at specified positions. Supports both random and sequential iteration through positions,
    with optional importance weighting and multiple shuffle variants per position.

    The pool supports two mutually exclusive interfaces:

    **Range-based interface:**
        Systematically scans shuffle positions using start, end, and step_size.
        
        Given L=len(background), W=shuffle_size:
        - Scans positions where window [pos, pos+W) fits within [start, end)
        - num_internal_states = len(range(start, min(end, L) - W + 1, step_size)) × num_shuffles
        - Defaults: start=0, end=L, step_size=1, num_shuffles=1

    **Position-based interface:**
        Directly specifies explicit positions with optional importance weights.
        
        - Parameters: positions (required), position_weights (optional)
        - num_internal_states = len(positions) × num_shuffles
        - Sequential mode: iterates through positions in order
        - Random mode: samples positions with optional weights (uniform by default)

    **Multiple shuffle variants (num_shuffles > 1):**
        Creates num_shuffles distinct shuffled patterns per scanned window.
        Internal state space = num_positions × num_shuffles.
        Use shuffle_seed to explore different shuffled patterns per window.
    
    **Shuffle algorithms:**
        - Regular shuffle (default): Randomly permutes all characters
        - Dinucleotide-preserving (preserve_dinucleotides=True): Uses Hierholzer's algorithm
          to find Eulerian paths that preserve adjacent character pair frequencies
    """
    
    def __init__(self, 
                 background_seq: Union[Pool, str],
                 shuffle_size: int,
                 start: int = None,
                 end: int = None,
                 step_size: int = None,
                 positions: List[int] = None,
                 position_weights: List[float] = None,
                 num_shuffles: int = 1,
                 shuffle_seed: int = None,
                 preserve_dinucleotides: bool = False,
                 mark_changes: bool = True,
                 mode: str = 'random',
                 max_num_states: int = None,
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a ShuffleScanPool.

        Args:
            background_seq: Background sequence (string or Pool) to scan across
            shuffle_size: Size of the window to shuffle
            
            # Range-based interface (mutually exclusive with positions):
            start: Starting position for first shuffle (default: 0)
            end: Ending position (exclusive, default: len(background_seq))
            step_size: Step between adjacent shuffles (default: 1)
            
            # Position-based interface (mutually exclusive with start/end/step_size):
            positions: Explicit positions to shuffle at. Must be non-empty. (default: None)
            position_weights: Weights for random position sampling. Requires mode='random'.
                (default: uniform weights)
            
            # Shuffle control:
            num_shuffles: Number of shuffle permutations per position (default: 1)
            shuffle_seed: Seed for shuffle permutation generation. If None, uses 0,1,2,...
                If specified, uses hash((shuffle_index, shuffle_seed)). (default: None)
            preserve_dinucleotides: If True, use Hierholzer's algorithm to preserve adjacent
                character pair frequencies. If False, use regular random shuffle. (default: False)
            mark_changes: If True, swapcase() shuffled region (default: True)
            
            # Generation control:
            mode: 'random' or 'sequential' (default: 'random')
            max_num_states: Maximum states before treating as infinite
            iteration_order: Order for sequential iteration (default: auto-assigned)
            name: Optional pool name (default: None)

        Raises:
            ValueError: If both range and position parameters provided
            ValueError: If position_weights without positions
            ValueError: If position_weights length mismatches positions
            ValueError: If position_weights with mode='sequential'
            ValueError: If positions is empty
            ValueError: If positions out of bounds
            ValueError: If shuffle_size > background length
            ValueError: If num_shuffles < 1
        """
        self.background_seq = background_seq
        self.shuffle_size = shuffle_size
        self.mark_changes = mark_changes
        self.num_shuffles = num_shuffles
        self.shuffle_seed = shuffle_seed
        self.preserve_dinucleotides = preserve_dinucleotides
        self._instance_warning_issued = False  # For dinucleotide fallback warnings

        # Validate num_shuffles
        if num_shuffles < 1:
            raise ValueError("num_shuffles must be at least 1")

        # Determine which interface is being used
        range_params_provided = any(p is not None for p in [start, end, step_size])
        position_params_provided = positions is not None
        
        if range_params_provided and position_params_provided:
            raise ValueError(
                "Cannot specify both range-based parameters (start/end/step_size) "
                "and position-based parameters (positions). Choose one interface."
            )
        
        if position_weights is not None and positions is None:
            raise ValueError("position_weights requires positions to be specified")
        
        if mode == 'sequential' and position_weights is not None:
            raise ValueError(
                "Cannot specify position_weights with mode='sequential'. "
                "Sequential mode iterates through all positions deterministically, "
                "ignoring weights. Use mode='random' for weighted selection, or omit "
                "position_weights parameter for sequential mode."
            )
        
        if position_params_provided:
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            self.positions = positions
            self.use_positions = True
            
            if position_weights is None:
                self.position_weights = [1.0] * len(positions)
            else:
                if len(position_weights) != len(positions):
                    raise ValueError(
                        f"position_weights length ({len(position_weights)}) must match "
                        f"positions length ({len(positions)})"
                    )
                self.position_weights = list(position_weights)
            
            total_weight = sum(self.position_weights)
            if total_weight <= 0:
                raise ValueError("Sum of position_weights must be positive")
            self.position_probabilities = [w / total_weight for w in self.position_weights]
            
            self.start = None
            self.end = None
            self.step_size = None
        else:
            self.use_positions = False
            self.positions = None
            self.position_weights = None
            self.position_probabilities = None
            
            self.start = start if start is not None else 0
            self.end = end
            self.step_size = step_size if step_size is not None else 1
        
        parents = []
        if isinstance(background_seq, Pool):
            parents.append(background_seq)
        
        # Get sequences and lengths for position computation/validation
        background = self._get_background_seq()
        L = len(background)
        W = self.shuffle_size

        # Validate shuffle_size
        if W > L:
            raise ValueError(
                f"shuffle_size ({W}) cannot be longer than "
                f"background_seq length ({L})"
            )

        # Range-based: compute and cache positions BEFORE super().__init__()
        if not self.use_positions:
            end_boundary = self.end if self.end is not None else L
            end_boundary = min(end_boundary, L)
            
            # Compute and assign to self.positions
            self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
        
        super().__init__(
            parents=tuple(parents), 
            op='shuffle_scan', 
            max_num_states=max_num_states, 
            mode=mode, 
            iteration_order=iteration_order,
            name=name,
            metadata=metadata
        )
        
        # Design cards: cached position from last _compute_seq call
        self._cached_pos: int | None = None
        self._cached_state: int | None = None

        # Position-based: validate user-provided positions (needs parent pools initialized)
        if self.use_positions:
            for pos in self.positions:
                if pos < 0 or pos + W > L:
                    raise ValueError(
                        f"Position {pos} is invalid: shuffle window "
                        f"[{pos}, {pos + W}) must fit within "
                        f"background sequence of length {L}"
                    )
            if len(self.positions) != len(set(self.positions)):
                raise ValueError("positions must not contain duplicates")

    def _get_background_seq(self) -> str:
        if isinstance(self.background_seq, Pool):
            return self.background_seq.seq
        return self.background_seq
    
    def _calculate_seq_length(self) -> int:
        return len(self._get_background_seq())
    
    def _calculate_num_internal_states(self) -> int:
        # Both interfaces now have positions cached in self.positions after __init__
        num_positions = len(self.positions) if self.positions else 0
        return max(0, num_positions * self.num_shuffles)
    
    def _compute_seq(self) -> str:
        background = self._get_background_seq()
        W = self.shuffle_size
        
        state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0

        # Mixed-radix decomposition
        position_index = state // self.num_shuffles
        shuffle_index = state % self.num_shuffles

        # Determine position
        if self.use_positions:
            if self.mode == 'sequential' and self.is_sequential_compatible():
                # Sequential: use position_index
                pos = self.positions[position_index]
            else:
                # Random with weights: weighted sampling
                rng = random.Random(self.get_state())
                pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            # Range-based: use position_index
            pos = self.positions[position_index]
        
        # Cache for design cards
        self._cached_pos = pos
        self._cached_state = self.get_state()
        
        # Extract and shuffle window
        window = background[pos:pos + W]
        
        if self.mark_changes:
            window = window.swapcase()
        
        # Determine shuffle seed
        if self.shuffle_seed is None:
            # Default: use shuffle_index directly
            final_shuffle_seed = shuffle_index
        else:
            # With shuffle_seed: explore different pattern space
            final_shuffle_seed = hash((shuffle_index, self.shuffle_seed))

        # Choose shuffling algorithm
        if self.preserve_dinucleotides:
            shuffled_window = self._hierholzer_shuffle(window, final_shuffle_seed)
        else:
            window_list = list(window)
            rng = random.Random(final_shuffle_seed)
            rng.shuffle(window_list)
            shuffled_window = ''.join(window_list)
        
        result = background[:pos] + shuffled_window + background[pos + W:]
        return result
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this ShuffleScanPool at the current state.
        
        Extends base Pool metadata with shuffle position information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + pos, pos_abs, window_size (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Ensure cached values are current (recompute if state changed)
        if self._cached_state != self.get_state():
            _ = self.seq  # Trigger computation to populate cache
        
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add ShuffleScanPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['pos'] = self._cached_pos
            metadata['pos_abs'] = abs_start + self._cached_pos if abs_start is not None else None
            metadata['window_size'] = self.shuffle_size
        
        return metadata
    
    def __repr__(self) -> str:
        bg_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
        shuffle_info = f", num_shuffles={self.num_shuffles}" if self.num_shuffles > 1 else ""
        
        if self.use_positions:
            weights_info = "" if all(w == self.position_weights[0] for w in self.position_weights) else f", weights={self.position_weights}"
            return f"ShuffleScanPool(bg={bg_str}, shuffle_size={self.shuffle_size}, positions={self.positions}{weights_info}{shuffle_info})"
        else:
            return f"ShuffleScanPool(bg={bg_str}, shuffle_size={self.shuffle_size}, start={self.start}, end={self.end}, step={self.step_size}{shuffle_info})"
    
    # Dinucleotide-preserving shuffle methods (Hierholzer's algorithm)
    
    def _build_digraph(self, window: str) -> dict:
        """Build a directed graph from the window sequence."""
        graph = defaultdict(list)
        for i in range(len(window) - 1):
            graph[window[i]].append(window[i + 1])
        return dict(graph)
    
    def _has_eulerian_path(self, graph: dict, window: str) -> bool:
        """Check if the directed graph has an Eulerian path."""
        if len(window) <= 1:
            return True
        
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for u, neighbors in graph.items():
            out_degree[u] += len(neighbors)
            for v in neighbors:
                in_degree[v] += 1
        
        all_vertices = set(in_degree.keys()) | set(out_degree.keys())
        
        start_vertices = 0
        end_vertices = 0
        
        for v in all_vertices:
            diff = out_degree[v] - in_degree[v]
            if diff == 1:
                start_vertices += 1
            elif diff == -1:
                end_vertices += 1
            elif diff != 0:
                return False
        
        return (start_vertices == 0 and end_vertices == 0) or \
               (start_vertices == 1 and end_vertices == 1)
    
    def _hierholzer_shuffle(self, window: str, seed_value: int) -> str:
        """Generate a dinucleotide-preserving shuffle using Hierholzer's algorithm."""
        if len(window) <= 1:
            return window
        
        graph = self._build_digraph(window)
        
        if not self._has_eulerian_path(graph, window):
            if not self._instance_warning_issued:
                warnings.warn(
                    f"Window '{window}' does not have an Eulerian path. "
                    "Falling back to regular shuffle that preserves only monomer counts.",
                    UserWarning
                )
                self._instance_warning_issued = True
            
            window_list = list(window)
            rng = random.Random(seed_value)
            rng.shuffle(window_list)
            return ''.join(window_list)
        
        adj = defaultdict(list)
        for u, neighbors in graph.items():
            adj[u] = list(neighbors)
        
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for u, neighbors in adj.items():
            out_degree[u] = len(neighbors)
            for v in neighbors:
                in_degree[v] += 1
        
        start = None
        for v in adj.keys():
            if out_degree[v] > in_degree[v]:
                start = v
                break
        if start is None:
            start = window[0]
        
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


# Legacy: only random shuffle
# class ShuffleScanPool(Pool):
#     """ShuffleScanPool with support for explicit position lists and multiple shuffle variants.

#     Performs scanning mutagenesis by randomly shuffling characters within a sliding window
#     at specified positions. Supports both random and sequential iteration through positions,
#     with optional importance weighting and multiple shuffle variants per position.

#     The pool supports two mutually exclusive interfaces:

#     **Range-based interface:**
#         Systematically scans shuffle positions using start, end, and step_size.
        
#         Given L=len(background), W=shuffle_size:
#         - Scans positions where window [pos, pos+W) fits within [start, end)
#         - num_states = len(range(start, min(end, L) - W + 1, step_size)) × num_shuffles
#         - Defaults: start=0, end=L, step_size=1, num_shuffles=1

#     **Position-based interface:**
#         Directly specifies explicit positions with optional importance weights.
        
#         - Parameters: positions (required), position_weights (optional)
#         - num_states = len(positions) × num_shuffles
#         - Sequential mode: iterates through positions in order
#         - Random mode: samples positions with optional weights (uniform by default)

#     **Multiple shuffle variants (num_shuffles > 1):**
#         Creates num_shuffles distinct shuffled patterns per scanned window/position.
#         State space = num_positions × num_shuffles.
#         Use shuffle_seed to explore different shuffled patterns per window/position.
#     """
    
#     def __init__(self, 
#                  background_seq: Union[Pool, str],
#                  shuffle_size: int,
#                  start: int = None,
#                  end: int = None,
#                  step_size: int = None,
#                  positions: List[int] = None,
#                  position_weights: List[float] = None,
#                  num_shuffles: int = 1,
#                  shuffle_seed: int = None,
#                  mark_changes: bool = True,
#                  mode: str = 'random',
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a ShuffleScanPool.

#         Args:
#             background_seq: Background sequence (string or Pool) to scan across
#             shuffle_size: Size of the window to shuffle
            
#             # Range-based interface (mutually exclusive with positions):
#             start: Starting position for first shuffle (default: 0)
#             end: Ending position (exclusive, default: len(background_seq))
#             step_size: Step between adjacent shuffles (default: 1)
            
#             # Position-based interface (mutually exclusive with start/end/step_size):
#             positions: Explicit positions to shuffle at. Must be non-empty. (default: None)
#             position_weights: Weights for random position sampling. Requires mode='random'.
#                 (default: uniform weights)
            
#             # Shuffle control:  
#             num_shuffles: Number of shuffle permutations per position (default: 1)
#             shuffle_seed: Seed for shuffle permutation generation. If None, uses 0,1,2,...
#                 If specified, uses hash((shuffle_index, shuffle_seed)). (default: None)
#             mark_changes: If True, swapcase() shuffled region (default: True)
            
#             mode: 'random' or 'sequential' (default: 'random')
#             max_num_states: Maximum states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned)
#             name: Optional pool name (default: None)

#         Raises:
#             ValueError: If both range and position parameters provided
#             ValueError: If position_weights without positions
#             ValueError: If position_weights length mismatches positions
#             ValueError: If position_weights with mode='sequential'
#             ValueError: If positions is empty
#             ValueError: If positions out of bounds
#             ValueError: If shuffle_size > background length
#             ValueError: If num_shuffles < 1
#         """
#         self.background_seq = background_seq
#         self.shuffle_size = shuffle_size
#         self.mark_changes = mark_changes
#         self.num_shuffles = num_shuffles
#         self.shuffle_seed = shuffle_seed

#         # Validate num_shuffles
#         if num_shuffles < 1:
#             raise ValueError("num_shuffles must be at least 1")

#         # Determine which interface is being used
#         range_params_provided = any(p is not None for p in [start, end, step_size])
#         position_params_provided = positions is not None
        
#         if range_params_provided and position_params_provided:
#             raise ValueError(
#                 "Cannot specify both range-based parameters (start/end/step_size) "
#                 "and position-based parameters (positions). Choose one interface."
#             )
        
#         if position_weights is not None and positions is None:
#             raise ValueError("position_weights requires positions to be specified")
        
#         if mode == 'sequential' and position_weights is not None:
#             raise ValueError(
#                 "Cannot specify position_weights with mode='sequential'. "
#                 "Sequential mode iterates through all positions deterministically, "
#                 "ignoring weights. Use mode='random' for weighted selection, or omit "
#                 "position_weights parameter for sequential mode."
#             )
        
#         if position_params_provided:
#             if not positions or len(positions) == 0:
#                 raise ValueError("positions must be a non-empty list")
#             self.positions = positions
#             self.use_positions = True
            
#             if position_weights is None:
#                 self.position_weights = [1.0] * len(positions)
#             else:
#                 if len(position_weights) != len(positions):
#                     raise ValueError(
#                         f"position_weights length ({len(position_weights)}) must match "
#                         f"positions length ({len(positions)})"
#                     )
#                 self.position_weights = list(position_weights)
            
#             total_weight = sum(self.position_weights)
#             if total_weight <= 0:
#                 raise ValueError("Sum of position_weights must be positive")
#             self.position_probabilities = [w / total_weight for w in self.position_weights]
            
#             self.start = None
#             self.end = None
#             self.step_size = None
#         else:
#             self.use_positions = False
#             self.positions = None
#             self.position_weights = None
#             self.position_probabilities = None
            
#             self.start = start if start is not None else 0
#             self.end = end
#             self.step_size = step_size if step_size is not None else 1
        
#         parents = []
#         if isinstance(background_seq, Pool):
#             parents.append(background_seq)
        
#         # Get sequences and lengths for position computation/validation
#         background = self._get_background_seq()
#         L = len(background)
#         W = self.shuffle_size

#         # Validate shuffle_size
#         if W > L:
#             raise ValueError(
#                 f"shuffle_size ({W}) cannot be longer than "
#                 f"background_seq length ({L})"
#             )

#         # Range-based: compute and cache positions BEFORE super().__init__()
#         if not self.use_positions:
#             end_boundary = self.end if self.end is not None else L
#             end_boundary = min(end_boundary, L)
            
#             # Compute and assign to self.positions
#             self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
        
#         super().__init__(
#             parents=tuple(parents), 
#             op='shuffle_scan', 
#             max_num_states=max_num_states, 
#             mode=mode, 
#             iteration_order=iteration_order,
#             name=name
#         )

#         # Position-based: validate user-provided positions (needs parent pools initialized)
#         if self.use_positions:
#             for pos in self.positions:
#                 if pos < 0 or pos + W > L:
#                     raise ValueError(
#                         f"Position {pos} is invalid: shuffle window "
#                         f"[{pos}, {pos + W}) must fit within "
#                         f"background sequence of length {L}"
#                     )

#     def _get_background_seq(self) -> str:
#         if isinstance(self.background_seq, Pool):
#             return self.background_seq.seq
#         return self.background_seq
    
#     def _calculate_seq_length(self) -> int:
#         return len(self._get_background_seq())
    
#     def _calculate_num_internal_states(self) -> int:
#         # Both interfaces now have positions cached in self.positions after __init__
#         num_positions = len(self.positions) if self.positions else 0
#         return max(0, num_positions * self.num_shuffles)
    
#     def _compute_seq(self) -> str:
#         background = self._get_background_seq()
#         W = self.shuffle_size
        
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0

#         # Mixed-radix decomposition
#         position_index = state // self.num_shuffles
#         shuffle_index = state % self.num_shuffles

#         # Determine position
#         if self.use_positions:
#             if self.mode == 'sequential' and self.is_sequential_compatible():
#                 # Sequential: use position_index
#                 pos = self.positions[position_index]
#             else:
#                 # Random with weights: weighted sampling
#                 random.seed(self.get_state())
#                 pos = random.choices(self.positions, weights=self.position_probabilities)[0]
#         else:
#             # Range-based: use position_index
#             pos = self.positions[position_index]
        
#         # Extract and shuffle window
#         window = background[pos:pos + W]
        
#         if self.mark_changes:
#             window = window.swapcase()
        
#         window_list = list(window)

#         if self.shuffle_seed is None:
#             # Default: use shuffle_index directly
#             final_shuffle_seed = shuffle_index
#         else:
#             # With shuffle_seed: explore different pattern space
#             final_shuffle_seed = hash((shuffle_index, self.shuffle_seed))

#         rng = random.Random(final_shuffle_seed)
#         rng.shuffle(window_list)
#         shuffled_window = ''.join(window_list)
        
#         result = background[:pos] + shuffled_window + background[pos + W:]
#         return result
    
#     def __repr__(self) -> str:
#         bg_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
#         shuffle_info = f", num_shuffles={self.num_shuffles}" if self.num_shuffles > 1 else ""
        
#         if self.use_positions:
#             weights_info = "" if all(w == self.position_weights[0] for w in self.position_weights) else f", weights={self.position_weights}"
#             return f"ShuffleScanPool(bg={bg_str}, shuffle_size={self.shuffle_size}, positions={self.positions}{weights_info}{shuffle_info})"
#         else:
#             return f"ShuffleScanPool(bg={bg_str}, shuffle_size={self.shuffle_size}, start={self.start}, end={self.end}, step={self.step_size}{shuffle_info})"


# Legacy: original
# class ShuffleScanPool(Pool):
#     """A class for scanning shuffles across a background sequence.
    
#     Performs scanning mutagenesis by randomly shuffling characters within a sliding
#     window of specified size. Supports both random and sequential iteration through
#     all possible shuffle positions.
    
#     The pool always has a finite number of states determined by the background
#     length, shuffle size, shift, and offset parameters.
    
#     Given L=len(background), W=shuffle_size, S=shift, and O=offset%W:
#     num_states = (L - O - W + S) // S
#     """
    
#     def __init__(self, 
#                  background_seq: Union[Pool, str],
#                  shuffle_size: int,
#                  change_case_of_shuffle: bool = True,
#                  mode: str = 'random',
#                  shift: int = 1,
#                  offset: int = 0,
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a ShuffleScanPool.
        
#         Args:
#             background_seq: Background sequence (string or Pool object) to scan across
#             shuffle_size: Size of the shuffle window (number of positions to shuffle)
#             change_case_of_shuffle: If True, apply swapcase() before shuffling (default: True)
#             mode: Either 'random' or 'sequential' (default: 'random')
#             shift: Number of positions to shift between adjacent shuffles (default: 1)
#             offset: Starting position offset for first shuffle (default: 0)
#             max_num_states: Maximum number of states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
#         """
#         self.background_seq = background_seq
#         self.shuffle_size = shuffle_size
#         self.change_case_of_shuffle = change_case_of_shuffle
#         self.shift = shift
#         self.offset = offset
        
#         # Collect parents for computation graph
#         parents = []
#         if isinstance(background_seq, Pool):
#             parents.append(background_seq)
        
#         super().__init__(
#             parents=tuple(parents), 
#             op='shuffle_scan', 
#             max_num_states=max_num_states, 
#             mode=mode, 
#             iteration_order=iteration_order,
#             name=name
#         )
    
#     def _get_background_seq(self) -> str:
#         """Get the current background sequence from either a Pool or string."""
#         if isinstance(self.background_seq, Pool):
#             return self.background_seq.seq
#         return self.background_seq
    
#     def _calculate_seq_length(self) -> int:
#         """Calculate the output sequence length.
        
#         Length stays same as background since we're only shuffling in place.
#         """
#         return len(self._get_background_seq())
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of shuffle positions based on parameters.
        
#         Formula: (L - O - W + S) // S
#         where L=len(background), W=shuffle_size, S=shift, O=offset%W
#         """
#         background = self._get_background_seq()
#         L = len(background)
#         W = self.shuffle_size
#         S = self.shift
        
#         # Validate that shuffle_size is not longer than background
#         if W > L:
#             raise ValueError(
#                 f"shuffle_size ({W}) cannot be longer than "
#                 f"background_seq length ({L})"
#             )
        
#         O = self.offset % W
#         num_states = (L - O - W + S) // S
        
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute sequence with shuffle at current state position.
        
#         Maps state to a shuffle position, extracts the window of characters,
#         applies case change if requested, shuffles them, and replaces them
#         in the background sequence.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         background = self._get_background_seq()
#         W = self.shuffle_size
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate shuffle position
#         O = self.offset % W
#         pos = O + (state * self.shift)
        
#         # Extract the window to shuffle
#         window = background[pos:pos + W]
        
#         # Apply case change if requested (before shuffling)
#         if self.change_case_of_shuffle:
#             window = window.swapcase()
        
#         # Shuffle the characters in the window
#         window_list = list(window)
#         rng = random.Random(state)
#         rng.shuffle(window_list)
#         shuffled_window = ''.join(window_list)
        
#         # Replace the window in the background
#         result = background[:pos] + shuffled_window + background[pos + W:]
        
#         return result
    
#     def __repr__(self) -> str:
#         background_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
#         return f"ShuffleScanPool(bg={background_str}, shuffle_size={self.shuffle_size}, S={self.shift}, O={self.offset})"

