from typing import Union, List, Dict, Any
from .pool import Pool
import random

class SubseqPool(Pool):
    """A class for extracting subsequences from a sequence with configurable positions.
    
    Extracts fixed-width windows from an input sequence. Supports both systematic
    scanning and explicit position specification with optional importance weights.
    Supports both random subsequence selection and sequential iteration.
    
    The pool supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically extracts subsequences using start, end, and step_size.
        
        Given L=len(seq), W=width:
        - Extracts windows where [pos, pos+W) fits within [start, end)
        - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
        - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit extraction positions with optional importance weights.
        
        - Parameters: positions (required), position_weights (optional)
        - num_internal_states = len(positions)
        - Sequential mode: iterates through positions in order
        - Random mode: samples positions with optional weights (uniform by default)
    """
    
    def __init__(self, 
                 seq: Union[Pool, str],
                 width: int,
                 start: int = None,
                 end: int = None,
                 step_size: int = None,
                 positions: List[int] = None,
                 position_weights: List[float] = None,
                 max_num_states: int = None,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a SubseqPool.
        
        Args:
            seq: Input sequence (string or Pool object) to extract subsequences from
            width: Width of each subsequence window
            
            # Range-based interface (mutually exclusive with positions):
            start: Starting position for first subsequence (default: 0)
            end: Ending position (exclusive) for subsequences. The last valid window
                starts at end-width. (default: len(seq))
            step_size: Number of positions to step between adjacent subsequences (default: 1)
            
            # Position-based interface (mutually exclusive with start/end/step_size):
            positions: List of explicit positions to extract at. Each position must be
                valid (window must fit within sequence). Must be non-empty. (default: None)
            position_weights: Optional weights for random sampling of positions. Only valid
                with mode='random'. Weights are normalized to probabilities. Must have same
                length as positions if provided. (default: uniform weights)
            
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned)
            name: Optional name for this pool (default: None)
        
        Raises:
            ValueError: If both range-based and position-based parameters are provided
            ValueError: If position_weights is provided without positions
            ValueError: If position_weights length doesn't match positions length
            ValueError: If position_weights are provided with mode='sequential'
            ValueError: If positions list is empty
            ValueError: If any position in positions is out of valid bounds
            ValueError: If width is longer than input sequence
        """
        self.input_seq = seq
        self.width = width
        
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
            
            # Set up weights (default to equal probability)
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
        if isinstance(seq, Pool):
            parents.append(seq)
        
        # Get sequences and lengths BEFORE super().__init__() to cache positions
        seq_str = self._get_base_seq()
        L = len(seq_str)
        W = self.width

        # For range-based interface: compute and cache positions BEFORE super().__init__()
        if not self.use_positions:
            # Validate that width is not longer than sequence
            if W > L:
                raise ValueError(
                    f"width ({W}) cannot be longer than "
                    f"sequence length ({L})"
                )
            
            end_boundary = self.end if self.end is not None else L
            end_boundary = min(end_boundary, L)
            
            # Compute and assign to self.positions
            self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
        
        super().__init__(
            parents=tuple(parents) if parents else (), 
            op='subseq', 
            max_num_states=max_num_states, 
            mode=mode, 
            iteration_order=iteration_order,
            name=name,
            metadata=metadata
        )
        
        # Design cards: cached position from last _compute_seq call
        self._cached_pos: int | None = None
        self._cached_state: int | None = None

        # Position-based: validate user-provided positions
        if self.use_positions:
            # Subsequence window [pos, pos+W) must fit within sequence
            for pos in self.positions:
                if pos < 0 or pos + W > L:
                    raise ValueError(
                        f"Position {pos} is invalid: subsequence window "
                        f"[{pos}, {pos + W}) must fit within "
                        f"sequence of length {L}"
                    )
            if len(self.positions) != len(set(self.positions)):
                raise ValueError("positions must not contain duplicates")
    
    def _get_base_seq(self) -> str:
        if isinstance(self.input_seq, Pool):
            return self.input_seq.seq
        return self.input_seq
    
    def _calculate_seq_length(self) -> int:
        return self.width
    
    def _calculate_num_internal_states(self) -> int:
        """Calculate number of subsequence positions."""
        num_positions = len(self.positions) if self.positions else 0
        return max(0, num_positions)
    
    def _compute_seq(self) -> str:
        seq = self._get_base_seq()
        W = self.width
        
        # Determine position
        if self.use_positions:
            if self.mode == 'sequential' and self.is_sequential_compatible():
                state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
                pos = self.positions[state]
            else:
                rng = random.Random(self.get_state())
                pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            # Range-based interface: use cached positions
            state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
            pos = self.positions[state]
        
        # Cache for design cards
        self._cached_pos = pos
        self._cached_state = self.get_state()
        
        return seq[pos:pos + W]
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this SubseqPool at the current state.
        
        Extends base Pool metadata with extraction position information.
        Note: SubseqPool is length-changing, so no _pos_abs is provided.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + pos, width (default)
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
        
        # Add SubseqPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['pos'] = self._cached_pos
            metadata['width'] = self.width
        
        return metadata
    
    def __repr__(self) -> str:
        seq_str = self.input_seq.seq if isinstance(self.input_seq, Pool) else self.input_seq
        if self.use_positions:
            weights_info = "" if all(w == self.position_weights[0] for w in self.position_weights) else f", weights={self.position_weights}"
            return f"SubseqPool(seq={seq_str}, W={self.width}, positions={self.positions}{weights_info})"
        else:
            return f"SubseqPool(seq={seq_str}, W={self.width}, start={self.start}, end={self.end}, step={self.step_size})"


# Legacy
# class SubseqPool(Pool):
#     """A class for extracting subsequences from a sequence with configurable tiling.
    
#     Supports both random subsequence selection and sequential iteration through
#     all possible subsequences. The pool always has a finite number of states
#     determined by the sequence length, width, shift, and offset parameters.
    
#     Given L=len(seq), W=width, S=shift, and O=offset%W, the number of states is:
#     (L - O - W + S) // S
    
#     When iterated directly or called with next(), randomly selects subsequences.
#     When used with generate_seqs() and included in combinatorially_complete_pools,
#     iterates through all subsequences sequentially.
#     """
#     def __init__(self, seq: Union[Pool, str], width: int, shift: int, offset: int = 0, max_num_states: int = None, mode: str = 'random', iteration_order: int | None = None, name: str | None = None):
#         """Initialize a SubseqPool.
        
#         Args:
#             seq: Input sequence (string or Pool object) to tile
#             width: Width of each subsequence
#             shift: Number of positions to shift between adjacent subsequences
#             offset: Value that, when taken modulo width, gives the left-most coordinate of the first subseq
#             max_num_states: Maximum number of states before treating as infinite
#             mode: Either 'random' or 'sequential' (default: 'random')
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
#         """
#         self.input_seq = seq
#         self.width = width
#         self.shift = shift
#         self.offset = offset
#         super().__init__(parents=(seq,) if isinstance(seq, Pool) else (), op='subseq', max_num_states=max_num_states, mode=mode, iteration_order=iteration_order, name=name)
    
#     def _get_base_seq(self) -> str:
#         """Get the base sequence from either a Pool or string."""
#         if isinstance(self.input_seq, Pool):
#             return self.input_seq.seq
#         return self.input_seq
    
#     def _calculate_seq_length(self) -> int:
#         """SubseqPool always produces sequences of width W."""
#         return self.width
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of subsequences based on parameters.
        
#         Formula: (L - O - W + S) // S
#         where L=len(seq), W=width, S=shift, O=offset%width
#         """
#         seq = self._get_base_seq()
#         L = len(seq)
#         W = self.width
#         S = self.shift
#         O = self.offset % W
        
#         num_states = (L - O - W + S) // S
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute subsequence based on current state.
        
#         Maps state to a left-most position and extracts subsequence.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         seq = self._get_base_seq()
#         O = self.offset % self.width
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate left-most position
#         left = O + (state * self.shift)
        
#         # Extract and return subsequence
#         return seq[left:left + self.width]
    
#     def __repr__(self) -> str:
#         return f"SubseqPool(W={self.width}, S={self.shift}, O={self.offset})"

