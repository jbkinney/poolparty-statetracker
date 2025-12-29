from typing import Union, List, Dict, Any
from .pool import Pool
import random

class DeletionScanPool(Pool):
    """A class for scanning deletions across a background sequence.
    
    Performs deletion scanning mutagenesis by systematically removing or marking
    segments from the background sequence. Supports both marked deletions (with
    a character like '-') and actual removal. Supports both random and sequential
    iteration through deletion positions.
    
    The pool supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically scans deletion positions using start, end, and step_size.
        
        Given L=len(background), W=deletion_size:
        - Scans positions where window [pos, pos+W) fits within [start, end)
        - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
        - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit deletion positions with optional importance weights.
        
        - Parameters: positions (required), position_weights (optional)
        - num_internal_states = len(positions)
        - Sequential mode: iterates through positions in order
        - Random mode: samples positions with optional weights (uniform by default)
    """
    
    def __init__(self, 
                 background_seq: Union[Pool, str],
                 deletion_size: int,
                 start: int = None,
                 end: int = None,
                 step_size: int = None,
                 positions: List[int] = None,
                 position_weights: List[float] = None,
                 mark_changes: bool = True,
                 deletion_character: str = '-',
                 mode: str = 'random',
                 max_num_states: int = None,
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a DeletionScanPool.
        
        Args:
            background_seq: Background sequence (string or Pool object) to delete from
            deletion_size: Size of the deletion region (number of positions to delete)
            
            # Range-based interface (mutually exclusive with positions):
            start: Starting position for first deletion (default: 0)
            end: Ending position (exclusive) for deletions. The last valid deletion
                window starts at end-deletion_size. (default: len(background_seq))
            step_size: Number of positions to step between adjacent deletions (default: 1)
            
            # Position-based interface (mutually exclusive with start/end/step_size):
            positions: List of explicit positions to delete at. Each position must be
                valid (deletion window must fit within background). Must be non-empty.
                (default: None)
            position_weights: Optional weights for random sampling of positions. Only valid
                with mode='random'. Weights are normalized to probabilities. Must have same
                length as positions if provided. (default: uniform weights)
            
            mark_changes: If True, mark deletions with deletion_character; if False,
                actually remove them from sequence (default: True)
            deletion_character: Character to mark deletions when mark_changes=True
                (default: '-')
            mode: Either 'random' or 'sequential' (default: 'random')
            max_num_states: Maximum number of states before treating as infinite
            iteration_order: Order for sequential iteration (default: auto-assigned)
            name: Optional name for this pool (default: None)
        
        Raises:
            ValueError: If both range-based and position-based parameters are provided
            ValueError: If position_weights is provided without positions
            ValueError: If position_weights length doesn't match positions length
            ValueError: If position_weights are provided with mode='sequential'
            ValueError: If positions list is empty
            ValueError: If any position in positions is out of valid bounds
            ValueError: If deletion_size is longer than background_seq
        """
        self.background_seq = background_seq
        self.deletion_size = deletion_size
        self.mark_changes = mark_changes
        self.deletion_character = deletion_character
        
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
            # Position-based interface
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
            # Range-based interface
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

        # Get sequences and lengths BEFORE super().__init__() to cache positions
        background = self._get_background_seq()
        L = len(background)
        W = self.deletion_size

        # For range-based interface: compute and cache positions BEFORE super().__init__()
        if not self.use_positions:
            # Validate that deletion_size is not longer than background
            if W > L:
                raise ValueError(
                    f"deletion_size ({W}) cannot be longer than "
                    f"background_seq length ({L})"
                )
            
            end_boundary = self.end if self.end is not None else L
            end_boundary = min(end_boundary, L)
            
            # Compute and assign to self.positions
            self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
            
            if len(self.positions) == 0:
                raise ValueError(
                    f"Range [start={self.start}, end={end_boundary}) with deletion_size={W} "
                    f"and step_size={self.step_size} produces no valid deletion positions"
                )
        
        super().__init__(
            parents=tuple(parents), 
            op='deletion_scan', 
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
            # Position-based: validate user-provided positions (needs parent pools initialized)
            for pos in self.positions:
                if pos < 0 or pos + W > L:
                    raise ValueError(
                        f"Position {pos} is invalid: deletion window "
                        f"[{pos}, {pos + W}) must fit within "
                        f"background sequence of length {L}"
                    )
            if len(self.positions) != len(set(self.positions)):
                raise ValueError("positions must not contain duplicates")
            
    def _get_background_seq(self) -> str:
        """Get the current background sequence from either a Pool or string."""
        if isinstance(self.background_seq, Pool):
            return self.background_seq.seq
        return self.background_seq
    
    def _calculate_seq_length(self) -> int:
        """Calculate the output sequence length."""
        background_len = len(self._get_background_seq())
        if self.mark_changes:
            return background_len  # Marked deletions don't change length
        else:
            return background_len - self.deletion_size  # Actual deletions reduce length
    
    def _calculate_num_internal_states(self) -> int:
        """Calculate number of deletion positions."""
        num_positions = len(self.positions) if self.positions else 0
        return max(0, num_positions)
    
    def _compute_seq(self) -> str:
        """Compute sequence with deletion at current state position."""
        background = self._get_background_seq()
        W = self.deletion_size
        
        # Determine position (use weighted random only for position-based + random mode)
        if self.use_positions and self.mode == 'random':
            rng = random.Random(self.get_state())
            pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
            pos = self.positions[state]
        
        # Cache for design cards
        self._cached_pos = pos
        self._cached_state = self.get_state()
        
        # Perform deletion
        if self.mark_changes:
            deletion_marker = self.deletion_character * W
            result = background[:pos] + deletion_marker + background[pos + W:]
        else:
            # Actually remove the characters (don't use marker which could corrupt data)
            result = background[:pos] + background[pos + W:]
        
        return result
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this DeletionScanPool at the current state.
        
        Extends base Pool metadata with deletion position information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + pos, pos_abs, del_len (default)
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
        
        # Add DeletionScanPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['pos'] = self._cached_pos
            metadata['pos_abs'] = abs_start + self._cached_pos if abs_start is not None else None
            metadata['del_len'] = self.deletion_size
        
        return metadata
    
    def __repr__(self) -> str:
        bg_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
        if self.use_positions:
            weights_info = "" if all(w == self.position_weights[0] for w in self.position_weights) else f", weights={self.position_weights}"
            return f"DeletionScanPool(bg={bg_str}, del_size={self.deletion_size}, positions={self.positions}{weights_info})"
        else:
            return f"DeletionScanPool(bg={bg_str}, del_size={self.deletion_size}, start={self.start}, end={self.end}, step={self.step_size})"



# Legacy
# class DeletionScanPool(Pool):
#     """A class for scanning deletions across a background sequence.
    
#     Performs deletion scanning mutagenesis by systematically removing or marking
#     segments from the background sequence. Supports both marked deletions (with
#     a character like '-') and unmarked deletions (actual removal). Supports both
#     random and sequential iteration through all possible deletion positions.
    
#     The pool always has a finite number of states determined by the background
#     length, deletion size, shift, and offset parameters.
    
#     Given L=len(background), W=deletion_size, S=shift, and O=offset%W:
#     num_states = (L - O - W + S) // S
#     """
    
#     def __init__(self, 
#                  background_seq: Union[Pool, str],
#                  deletion_size: int,
#                  mark_deletion: bool = True,
#                  deletion_character: str = '-',
#                  mode: str = 'random',
#                  shift: int = 1,
#                  offset: int = 0,
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a DeletionScanPool.
        
#         Args:
#             background_seq: Background sequence (string or Pool object) to delete from
#             deletion_size: Size of the deletion region (number of positions to delete)
#             mark_deletion: If True, mark deletions with deletion_character; if False, actually remove them (default: True)
#             deletion_character: Character to mark deletions when mark_deletion=True (default: '-')
#             mode: Either 'random' or 'sequential' (default: 'random')
#             shift: Number of positions to shift between adjacent deletions (default: 1)
#             offset: Starting position offset for first deletion (default: 0)
#             max_num_states: Maximum number of states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
#         """
#         self.background_seq = background_seq
#         self.deletion_size = deletion_size
#         self.mark_deletion = mark_deletion
#         self.deletion_character = deletion_character
#         self.shift = shift
#         self.offset = offset
        
#         # Collect parents for computation graph
#         parents = []
#         if isinstance(background_seq, Pool):
#             parents.append(background_seq)
        
#         super().__init__(
#             parents=tuple(parents), 
#             op='deletion_scan', 
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
        
#         When mark_deletion=True: length stays same as background
#         When mark_deletion=False: length is background - deletion_size
#         """
#         background_len = len(self._get_background_seq())
#         if self.mark_deletion:
#             return background_len
#         else:
#             return background_len - self.deletion_size
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of deletion positions based on parameters.
        
#         Formula: (L - O - W + S) // S
#         where L=len(background), W=deletion_size, S=shift, O=offset%W
#         """
#         background = self._get_background_seq()
#         L = len(background)
#         W = self.deletion_size
#         S = self.shift
        
#         # Validate that deletion_size is not longer than background
#         if W > L:
#             raise ValueError(
#                 f"deletion_size ({W}) cannot be longer than "
#                 f"background_seq length ({L})"
#             )
        
#         O = self.offset % W
#         num_states = (L - O - W + S) // S
        
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute sequence with deletion at current state position.
        
#         Maps state to a deletion position and either marks or removes
#         the deletion region at that position.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         background = self._get_background_seq()
#         W = self.deletion_size
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate deletion position
#         O = self.offset % W
#         pos = O + (state * self.shift)
        
#         # Perform deletion
#         if self.mark_deletion:
#             # Replace W characters at position pos with deletion_character
#             deletion_marker = self.deletion_character * W
#             result = background[:pos] + deletion_marker + background[pos + W:]
#         else:
#             # First create marked version, then remove all deletion_character occurrences
#             deletion_marker = self.deletion_character * W
#             marked_result = background[:pos] + deletion_marker + background[pos + W:]
#             # Remove all deletion_character occurrences
#             result = marked_result.replace(self.deletion_character, '')
        
#         return result
    
#     def __repr__(self) -> str:
#         background_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
#         mark_str = "marked" if self.mark_deletion else "unmarked"
#         return f"DeletionScanPool(bg={background_str}, del_size={self.deletion_size}, mode={mark_str}, S={self.shift}, O={self.offset})"

