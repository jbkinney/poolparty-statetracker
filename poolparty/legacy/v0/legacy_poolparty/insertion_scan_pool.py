from typing import Union, List, Literal, Dict, Any
from .pool import Pool
import random

class InsertionScanPool(Pool):
    """A class for scanning an insertion sequence across a background sequence.
    
    Performs scanning mutagenesis by either overwriting positions in the background
    sequence or inserting between positions. Supports both random and sequential
    iteration through insertion positions.
    
    The pool supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically scans positions using start, end, and step_size parameters.
        
        When insert_or_overwrite='overwrite':
            Given L=len(background), W=len(insertion):
            - Scans positions where window [pos, pos+W) fits within [start, end)
            - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
            - Defaults: start=0, end=L, step_size=1
            
        When insert_or_overwrite='insert':
            Given L=len(background):
            - Can insert at positions 0 through L (inclusive)
            - num_internal_states = len(range(start, min(end, L) + 1, step_size))
            - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit positions to scan with optional importance weights.
        
        - Parameters: positions (required), position_weights (optional)
        - num_internal_states = len(positions)
        - Sequential mode: iterates through positions in order
        - Random mode: samples positions with optional weights (uniform by default)
    """
    
    def __init__(self, 
                 background_seq: Union[Pool, str],
                 insert_seq: Union[Pool, str],
                 start: int = None,
                 end: int = None,
                 step_size: int = None,
                 positions: List[int] = None,
                 position_weights: List[float] = None,
                 insert_or_overwrite: Literal['insert', 'overwrite'] = 'overwrite',
                 mark_changes: bool = False,
                 mode: str = 'random',
                 max_num_states: int = None,
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize an InsertionScanPool.
        
        Args:
            background_seq: Background sequence (string or Pool object) to scan across
            insert_seq: Sequence to insert or overwrite with (string or Pool object)

            # Range-based interface (mutually exclusive with positions):
            start: Starting position for first insertion (default: 0)
            end: Ending position (exclusive) for insertions. In overwrite mode, the last
                valid window starts at end-W. In insert mode, can insert up to position end.
                (default: len(background_seq))
            step_size: Number of positions to step between adjacent insertions (default: 1)

            # Position-based interface (mutually exclusive with start/end/step_size):
            positions: List of explicit positions to scan. Each position must be valid for
                the chosen insert_or_overwrite mode. Must be non-empty. (default: None)
            position_weights: Optional weights for random sampling of positions. Only valid
                with mode='random'. Weights are normalized to probabilities. Must have same
                length as positions if provided. (default: uniform weights)

            insert_or_overwrite: If "overwrite", overwrite characters; if "insert", insert between positions (default: "overwrite")
            mark_changes: If True, apply swapcase() to the inserted sequence for visualization (default: False)
            mode: Either 'random' or 'sequential' (default: 'random')
            max_num_states: Maximum number of states before treating as infinite
            iteration_order: Order for sequential iteration
            
        Raises:
            ValueError: If both range-based (start/end/step_size) and position-based 
                (positions) parameters are provided
            ValueError: If position_weights is provided without positions
            ValueError: If position_weights length doesn't match positions length
            ValueError: If position_weights are provided with mode='sequential'
            ValueError: If positions list is empty
            ValueError: If any position in positions is out of valid bounds
            ValueError: If insert_or_overwrite is not 'insert' or 'overwrite'
            ValueError: In overwrite mode with range-based interface, if insert_seq 
                is longer than background_seq
        """
        self.background_seq = background_seq
        self.insert_seq = insert_seq
        self.insert_or_overwrite = insert_or_overwrite
        self.mark_changes = mark_changes
        
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
        
        # Validate that weights are not provided with sequential mode
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
            
            # Normalize weights to probabilities
            total_weight = sum(self.position_weights)
            if total_weight <= 0:
                raise ValueError("Sum of position_weights must be positive")
            self.position_probabilities = [w / total_weight for w in self.position_weights]
            
            # Not used in position mode
            self.start = None
            self.end = None
            self.step_size = None
        else:
            # Range-based interface (use V2 defaults)
            self.use_positions = False
            self.positions = None
            self.position_weights = None
            self.position_probabilities = None
            
            self.start = start if start is not None else 0
            self.end = end
            self.step_size = step_size if step_size is not None else 1

        # Validate insert_or_overwrite
        if self.insert_or_overwrite not in ['insert', 'overwrite']:
            raise ValueError(
                f"insert_or_overwrite must be 'insert' or 'overwrite', "
                f"got '{self.insert_or_overwrite}'"
            )
        
        # Collect parents for computation graph
        parents = []
        if isinstance(background_seq, Pool):
            parents.append(background_seq)
        if isinstance(insert_seq, Pool):
            parents.append(insert_seq)
        
        # Get sequences and lengths BEFORE super().__init__() to cache positions
        background = self._get_background_seq()
        insertion = self._get_insert_seq()
        L = len(background)
        W = len(insertion)

        # For range-based interface: compute and cache positions BEFORE super().__init__()
        if not self.use_positions:
            end_boundary = self.end if self.end is not None else L
            end_boundary = min(end_boundary, L)
            
            # Compute and assign to self.positions
            if self.insert_or_overwrite == 'overwrite':
                # Validate that insertion is not longer than background in overwrite mode
                if W > L:
                    raise ValueError(
                        f"insert_seq length ({W}) cannot be longer than "
                        f"background_seq length ({L}) when insert_or_overwrite='overwrite'"
                    )
                self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
            else:  # insert mode
                self.positions = list(range(self.start, end_boundary + 1, self.step_size))
            
            if len(self.positions) == 0:
                raise ValueError(
                    f"Range [start={self.start}, end={end_boundary}) with "
                    f"step_size={self.step_size} produces no valid insertion positions"
                )
        
        super().__init__(
            parents=tuple(parents), 
            op='insertion_scan', 
            max_num_states=max_num_states, 
            mode=mode, 
            iteration_order=iteration_order,
            name=name,
            metadata=metadata
        )
        
        # Design cards: cached position from last _compute_seq call
        self._cached_pos: int | None = None
        self._cached_insert: str | None = None
        self._cached_state: int | None = None

        # Validate position bounds (after super().__init__() so parent pools are initialized)
        if self.use_positions:
            # Position-based: validate user-provided positions
            if self.insert_or_overwrite == 'overwrite':
                # In overwrite mode, window [pos, pos+W) must fit within background
                for pos in self.positions:
                    if pos < 0 or pos + W > L:
                        raise ValueError(
                            f"Position {pos} is invalid: in overwrite mode, "
                            f"window [{pos}, {pos + W}) must fit within "
                            f"background sequence of length {L}"
                        )
            else:  # insert mode
                # In insert mode, can insert at positions 0 through L (inclusive)
                for pos in self.positions:
                    if pos < 0 or pos > L:
                        raise ValueError(
                            f"Position {pos} is invalid: in insert mode, "
                            f"position must be in range [0, {L}] for "
                            f"background sequence of length {L}"
                        )
            if len(self.positions) != len(set(self.positions)):
                raise ValueError("positions must not contain duplicates")
    
    def _get_background_seq(self) -> str:
        """Get the current background sequence from either a Pool or string."""
        if isinstance(self.background_seq, Pool):
            return self.background_seq.seq
        return self.background_seq
    
    def _get_insert_seq(self) -> str:
        """Get the current insertion sequence from either a Pool or string."""
        if isinstance(self.insert_seq, Pool):
            return self.insert_seq.seq
        return self.insert_seq
    
    def _calculate_seq_length(self) -> int:
        """Calculate the output sequence length."""
        background_len = len(self._get_background_seq())
        if self.insert_or_overwrite == 'overwrite':
            return background_len
        else:
            insertion_len = len(self._get_insert_seq())
            return background_len + insertion_len
    
    def _calculate_num_internal_states(self) -> int:
        """Calculate number of insertion positions."""
        # Both interfaces now have positions cached in self.positions after __init__
        num_positions = len(self.positions) if self.positions else 0
        return max(0, num_positions)
    
    def _compute_seq(self) -> str:
        """Compute sequence with insertion at current state position."""
        background = self._get_background_seq()
        insertion = self._get_insert_seq()
        
        # Determine position (use weighted random only for position-based + random mode)
        if self.use_positions and self.mode == 'random':
            rng = random.Random(self.get_state())
            pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
            pos = self.positions[state]
        
        # Cache for design cards
        self._cached_pos = pos
        self._cached_insert = insertion
        self._cached_state = self.get_state()
        
        # Apply case change if requested
        if self.mark_changes:
            insertion_to_use = insertion.swapcase()
        else:
            insertion_to_use = insertion
        
        # Perform insertion or overwrite
        if self.insert_or_overwrite == 'overwrite':
            # Overwrite W characters at position pos
            W = len(insertion)
            result = background[:pos] + insertion_to_use + background[pos + W:]
        else:
            # Insert between positions
            result = background[:pos] + insertion_to_use + background[pos:]
        
        return result
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this InsertionScanPool at the current state.
        
        Extends base Pool metadata with insertion position information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + pos, pos_abs, insert (default)
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
        
        # Add InsertionScanPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['pos'] = self._cached_pos
            metadata['pos_abs'] = abs_start + self._cached_pos if abs_start is not None else None
            metadata['insert'] = self._cached_insert
        
        return metadata
    
    def __repr__(self) -> str:
        background_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
        insertion_str = self.insert_seq.seq if isinstance(self.insert_seq, Pool) else self.insert_seq
        mode_str = self.insert_or_overwrite
        
        if self.use_positions:
            weights_info = "" if all(w == self.position_weights[0] for w in self.position_weights) else f", weights={self.position_weights}"
            return f"InsertionScanPool(bg={background_str}, ins={insertion_str}, mode={mode_str}, positions={self.positions}{weights_info})"
        else:
            return f"InsertionScanPool(bg={background_str}, ins={insertion_str}, mode={mode_str}, start={self.start}, end={self.end}, step={self.step_size})"



# Legacy
# class InsertionScanPool(Pool):
#     """A class for scanning an insertion sequence across a background sequence.
    
#     Performs scanning mutagenesis by either overwriting positions in the background
#     sequence or inserting between positions. Supports both random and sequential
#     iteration through all possible insertion positions.
    
#     The pool always has a finite number of states determined by the background
#     length, insertion length, shift, and offset parameters.
    
#     When overwrite_insertion_site=True:
#         Given L=len(background), W=len(insertion), S=shift, and O=offset%W:
#         num_states = (L - O - W + S) // S
        
#     When overwrite_insertion_site=False:
#         Given L=len(background), S=shift, and O=offset:
#         num_states = (L - O + 1 + S) // S
#     """
    
#     def __init__(self, 
#                  background_seq: Union[Pool, str],
#                  insertion_seq: Union[Pool, str],
#                  overwrite_insertion_site: bool = True,
#                  change_case_of_insert: bool = False,
#                  mode: str = 'random',
#                  shift: int = 1,
#                  offset: int = 0,
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize an InsertionScanPool.
        
#         Args:
#             background_seq: Background sequence (string or Pool object) to scan across
#             insertion_seq: Sequence to insert or overwrite with (string or Pool object)
#             overwrite_insertion_site: If True, replace characters; if False, insert between positions (default: True)
#             change_case_of_insert: If True, apply swapcase() to the inserted sequence (default: False)
#             mode: Either 'random' or 'sequential' (default: 'random')
#             shift: Number of positions to shift between adjacent insertions (default: 1)
#             offset: Starting position offset for first insertion (default: 0)
#             max_num_states: Maximum number of states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
#         """
#         self.background_seq = background_seq
#         self.insertion_seq = insertion_seq
#         self.overwrite_insertion_site = overwrite_insertion_site
#         self.change_case_of_insert = change_case_of_insert
#         self.shift = shift
#         self.offset = offset
        
#         # Collect parents for computation graph
#         parents = []
#         if isinstance(background_seq, Pool):
#             parents.append(background_seq)
#         if isinstance(insertion_seq, Pool):
#             parents.append(insertion_seq)
        
#         super().__init__(
#             parents=tuple(parents), 
#             op='insertion_scan', 
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
    
#     def _get_insertion_seq(self) -> str:
#         """Get the current insertion sequence from either a Pool or string."""
#         if isinstance(self.insertion_seq, Pool):
#             return self.insertion_seq.seq
#         return self.insertion_seq
    
#     def _calculate_seq_length(self) -> int:
#         """Calculate the output sequence length.
        
#         When overwrite_insertion_site=True: length stays same as background
#         When overwrite_insertion_site=False: length is background + insertion
#         """
#         background_len = len(self._get_background_seq())
#         if self.overwrite_insertion_site:
#             return background_len
#         else:
#             insertion_len = len(self._get_insertion_seq())
#             return background_len + insertion_len
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of insertion positions based on parameters.
        
#         When overwrite_insertion_site=True:
#             Formula: (L - O - W + S) // S
#             where L=len(background), W=len(insertion), S=shift, O=offset%W
            
#         When overwrite_insertion_site=False:
#             Formula: (L - O + 1 + S) // S
#             where L=len(background), S=shift, O=offset
#         """
#         background = self._get_background_seq()
#         insertion = self._get_insertion_seq()
#         L = len(background)
#         W = len(insertion)
#         S = self.shift
        
#         # Validate that insertion is not longer than background in overwrite mode
#         if self.overwrite_insertion_site and W > L:
#             raise ValueError(
#                 f"insertion_seq length ({W}) cannot be longer than "
#                 f"background_seq length ({L}) when overwrite_insertion_site=True"
#             )
        
#         if self.overwrite_insertion_site:
#             O = self.offset % W
#             num_states = (L - O - W + S) // S
#         else:
#             O = self.offset
#             num_states = (L - O + 1 + S) // S
        
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute sequence with insertion at current state position.
        
#         Maps state to an insertion position and either overwrites or inserts
#         the insertion sequence at that position.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         background = self._get_background_seq()
#         insertion = self._get_insertion_seq()
#         W = len(insertion)
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate insertion position
#         if self.overwrite_insertion_site:
#             O = self.offset % W
#         else:
#             O = self.offset
        
#         pos = O + (state * self.shift)
        
#         # Apply case change if requested
#         if self.change_case_of_insert:
#             insertion_to_use = insertion.swapcase()
#         else:
#             insertion_to_use = insertion
        
#         # Perform insertion or overwrite
#         if self.overwrite_insertion_site:
#             # Replace W characters at position pos
#             result = background[:pos] + insertion_to_use + background[pos + W:]
#         else:
#             # Insert between positions
#             result = background[:pos] + insertion_to_use + background[pos:]
        
#         return result
    
#     def __repr__(self) -> str:
#         background_str = self.background_seq.seq if isinstance(self.background_seq, Pool) else self.background_seq
#         insertion_str = self.insertion_seq.seq if isinstance(self.insertion_seq, Pool) else self.insertion_seq
#         mode_str = "overwrite" if self.overwrite_insertion_site else "insert"
#         return f"InsertionScanPool(bg={background_str}, ins={insertion_str}, mode={mode_str}, S={self.shift}, O={self.offset})"

