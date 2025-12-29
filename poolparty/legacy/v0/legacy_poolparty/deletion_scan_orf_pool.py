from typing import Union, List, Dict, Any
from .pool import Pool
from .orf_pool import ORFPool
import random


class DeletionScanORFPool(ORFPool):
    """Scan deletions across an ORF at the codon level.
    
    Performs deletion scanning mutagenesis on open reading frames by systematically
    removing or marking codon segments. Operations work at the codon level to
    maintain reading frame integrity.
    
    Supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically scans deletion positions using start, end, and step_size.
        All parameters are in codon units (not nucleotides).
        
        Given L=num_codons, W=deletion_size:
        - Scans positions where window [pos, pos+W) fits within [start, end)
        - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
        - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit codon positions with optional importance weights.
        
        - Parameters: positions (required), position_weights (optional)
        - num_internal_states = len(positions)
        - Sequential mode: iterates through positions in order
        - Random mode: samples positions with optional weights
    
    Inherits from ORFPool:
        - DNA validation and flanking region support
        - Codon splitting utilities
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
                 orf_start: int = 0,
                 orf_end: int = None,
                 mode: str = 'random',
                 max_num_states: int = None,
                 iteration_order: int = None,
                 name: str = None,
                 metadata: str = 'features'):
        """Initialize DeletionScanORFPool.
        
        Args:
            background_seq: Input sequence (string or Pool). Can include flanking regions (UTRs).
            deletion_size: Number of codons to delete at each position
            
            # Range-based interface (codon units, mutually exclusive with positions):
            start: Starting codon position for first deletion (default: 0)
            end: Ending codon position (exclusive). Last valid deletion starts at
                end - deletion_size. (default: num_codons)
            step_size: Codon positions to step between deletions (default: 1)
            
            # Position-based interface (mutually exclusive with start/end/step_size):
            positions: List of explicit codon positions to delete at. Each position
                must be valid (deletion window fits within ORF). (default: None)
            position_weights: Optional weights for random sampling. Only valid with
                mode='random'. (default: uniform weights)
            
            mark_changes: If True, mark deletions with deletion_character; if False,
                actually remove codons (default: True)
            deletion_character: Character for marking deletions (default: '-')
            orf_start: Start index of ORF within seq (nucleotide, 0-based). Default: 0
            orf_end: End index of ORF within seq (nucleotide, exclusive). Default: len(seq)
            mode: 'random' or 'sequential' (default: 'random')
            max_num_states: Maximum states before treating as infinite
            iteration_order: Order for sequential iteration
            name: Optional pool name
        """
        # Handle Pool vs string input
        if isinstance(background_seq, Pool):
            actual_seq = background_seq.seq
            parents_to_pass = (background_seq,)
            allow_lowercase = True
        else:
            actual_seq = background_seq
            parents_to_pass = (background_seq,)
            allow_lowercase = False
        
        # Initialize ORFPool (handles validation, codon splitting, flanks)
        # Note: We don't need codon tables for deletion scanning
        super().__init__(
            seq=actual_seq,
            orf_start=orf_start,
            orf_end=orf_end,
            codon_table=None,  # Scan pools don't need codon lookups
            mark_changes=mark_changes,
            parents=parents_to_pass,
            op='deletion_scan_orf',
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            allow_lowercase=allow_lowercase,
            metadata=metadata
        )
        
        self.deletion_size = deletion_size
        self.deletion_character = deletion_character
        
        # Validate deletion_size
        if deletion_size <= 0:
            raise ValueError(f"deletion_size must be > 0, got {deletion_size}")
        if deletion_size > self.num_codons:
            raise ValueError(
                f"deletion_size ({deletion_size}) cannot exceed "
                f"number of codons ({self.num_codons})"
            )
        
        # Determine interface type
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
                "Sequential mode iterates deterministically, ignoring weights."
            )
        
        # Set up interface
        if position_params_provided:
            # Position-based interface
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            
            # Validate positions
            W = self.deletion_size
            L = self.num_codons
            for pos in positions:
                if not isinstance(pos, int):
                    raise ValueError(f"All positions must be integers, got {type(pos)}")
                if pos < 0 or pos + W > L:
                    raise ValueError(
                        f"Position {pos} is invalid: deletion window "
                        f"[{pos}, {pos + W}) must fit within ORF ({L} codons)"
                    )
            
            if len(positions) != len(set(positions)):
                raise ValueError("positions must not contain duplicates")
            
            self.positions = list(positions)
            self.use_positions = True
            
            # Set up weights
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
            self.position_weights = None
            self.position_probabilities = None
            
            self.start = start if start is not None else 0
            self.end = end if end is not None else self.num_codons
            self.step_size = step_size if step_size is not None else 1
            
            # Compute positions
            W = self.deletion_size
            end_boundary = min(self.end, self.num_codons)
            self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
            
            if len(self.positions) == 0:
                raise ValueError(
                    f"Range [start={self.start}, end={end_boundary}) with deletion_size={W} "
                    f"and step_size={self.step_size} produces no valid deletion positions"
                )

        # Update state counts (Pool.__init__ was called before we set positions)
        self.num_internal_states = self._calculate_num_internal_states()
        self.num_states = self._calculate_num_states()
        
        # Design cards: cached values from last _compute_seq call
        self._cached_codon_pos: int | None = None
        self._cached_del_codons: str | None = None
        self._cached_del_aa: str | None = None
        self._cached_state: int | None = None
    
    def _calculate_num_internal_states(self) -> int:
        """Number of deletion positions."""
        if not hasattr(self, 'positions') or self.positions is None:
            return 1  # Temporary during __init__
        return max(0, len(self.positions))
    
    def _calculate_seq_length(self) -> int:
        """Output sequence length."""
        base_len = len(self.upstream_flank) + len(self.downstream_flank)
        if self.mark_changes:
            return base_len + len(self.orf_seq)  # Same length (marked)
        else:
            return base_len + len(self.orf_seq) - (self.deletion_size * 3)  # Shorter
    
    def _compute_seq(self) -> str:
        """Compute sequence with deletion at current state."""
        # Get current codons (handle transformer pattern)
        if isinstance(self.parents[0], Pool):
            parent_seq = self.parents[0].seq
            orf_start_idx = len(self.upstream_flank)
            orf_end_idx = orf_start_idx + len(self.orf_seq)
            current_orf = parent_seq[orf_start_idx:orf_end_idx]
            base_codons = [current_orf[i:i+3] for i in range(0, len(current_orf), 3)]
        else:
            base_codons = self.codons.copy()
        
        W = self.deletion_size
        
        # Determine position
        if self.use_positions and self.mode == 'random':
            rng = random.Random(self.get_state())
            pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
            pos = self.positions[state]
        
        # Cache for design cards
        self._cached_codon_pos = pos
        deleted_codons = base_codons[pos:pos + W]
        self._cached_del_codons = ''.join(deleted_codons)
        # Translate to amino acids if codon table available
        if hasattr(self, 'codon_table') and self.codon_table:
            self._cached_del_aa = ''.join(
                self.codon_table.get(c.upper(), 'X') for c in deleted_codons
            )
        else:
            self._cached_del_aa = None
        self._cached_state = self.get_state()
        
        # Perform deletion
        if self.mark_changes:
            deletion_marker = [self.deletion_character * 3 for _ in range(W)]
            result_codons = base_codons[:pos] + deletion_marker + base_codons[pos + W:]
        else:
            result_codons = base_codons[:pos] + base_codons[pos + W:]
        
        return self._reassemble_with_flanks(''.join(result_codons))
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this DeletionScanORFPool at the current state.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + codon_pos, codon_pos_abs, del_codons, del_aa (default)
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
        
        # Add DeletionScanORFPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['codon_pos'] = self._cached_codon_pos
            # Compute absolute codon position (accounting for upstream flank)
            if abs_start is not None:
                flank_len = len(self.upstream_flank) if hasattr(self, 'upstream_flank') else 0
                metadata['codon_pos_abs'] = abs_start + flank_len + (self._cached_codon_pos * 3)
            else:
                metadata['codon_pos_abs'] = None
            metadata['del_codons'] = self._cached_del_codons
            metadata['del_aa'] = self._cached_del_aa
        
        return metadata
    
    def __repr__(self) -> str:
        seq_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
        if self.use_positions:
            return f"DeletionScanORFPool(orf={seq_preview}, del_size={self.deletion_size}, positions={self.positions})"
        else:
            return f"DeletionScanORFPool(orf={seq_preview}, del_size={self.deletion_size}, start={self.start}, end={self.end}, step={self.step_size})"



# Legacy
# class DeletionScanORFPool(Pool):
#     """A class for scanning deletions across an ORF background at the codon level.
    
#     Performs deletion scanning mutagenesis on open reading frames by systematically removing or
#     marking codon segments from the ORF sequence. The shift and offset parameters work at the
#     codon level (not nucleotide level) to maintain reading frame integrity.
    
#     Both marked deletions (with a character like '-') and unmarked deletions (actual removal)
#     are supported. Supports both random and sequential iteration through all possible deletion positions.
    
#     The orf_seq must be a DNA sequence (ACGT only) with length divisible by 3, or a Pool object.
    
#     The pool always has a finite number of states determined by the ORF codon count,
#     deletion codon count, shift, and offset parameters.
    
#     Given L=num_codons(orf), W=deletion_size, S=shift, and O=offset%W:
#     num_states = (L - O - W + S) // S
#     """
    
#     def __init__(self, 
#                  orf_seq: Union[Pool, str],
#                  deletion_size: int,
#                  mark_deletion: bool = True,
#                  deletion_character: str = '-',
#                  mode: str = 'random',
#                  shift: int = 1,
#                  offset: int = 0,
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a DeletionScanORFPool.
        
#         Args:
#             orf_seq: Background ORF sequence (string or Pool object) to delete from
#                      (must be ACGT only and length divisible by 3)
#             deletion_size: Size of the deletion region (number of codons to delete)
#             mark_deletion: If True, mark deletions with deletion_character; if False, actually remove them (default: True)
#             deletion_character: Character to mark deletions when mark_deletion=True (default: '-')
#             mode: Either 'random' or 'sequential' (default: 'random')
#             shift: Number of codon positions to shift between adjacent deletions (default: 1)
#             offset: Starting codon position offset for first deletion (default: 0)
#             max_num_states: Maximum number of states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
            
#         Raises:
#             ValueError: If orf_seq is not a valid DNA sequence or not divisible by 3
#         """
#         self.orf_seq = orf_seq
#         self.deletion_size = deletion_size
#         self.mark_deletion = mark_deletion
#         self.deletion_character = deletion_character
#         self.shift = shift
#         self.offset = offset
        
#         # Validate and parse orf_seq
#         if isinstance(orf_seq, Pool):
#             # If it's a Pool, validate when we get the sequence
#             pass
#         else:
#             # Validate orf_seq is a string
#             if not isinstance(orf_seq, str):
#                 raise ValueError("orf_seq must be a string or Pool object")
            
#             # Validate orf_seq is DNA
#             if not all(c in 'ACGTacgt' for c in orf_seq):
#                 raise ValueError(f"orf_seq must contain only ACGT characters, got '{orf_seq}'")
            
#             # Validate length is divisible by 3
#             if len(orf_seq) % 3 != 0:
#                 raise ValueError(
#                     f"orf_seq length must be divisible by 3, got length {len(orf_seq)}"
#                 )
        
#         # Collect parents for computation graph
#         parents = []
#         if isinstance(orf_seq, Pool):
#             parents.append(orf_seq)
        
#         super().__init__(
#             parents=tuple(parents), 
#             op='deletion_scan_orf', 
#             max_num_states=max_num_states, 
#             mode=mode, 
#             iteration_order=iteration_order,
#             name=name
#         )
    
#     def _get_orf_seq(self) -> str:
#         """Get the current ORF sequence from either a Pool or string."""
#         if isinstance(self.orf_seq, Pool):
#             return self.orf_seq.seq
#         return self.orf_seq
    
#     def _get_orf_codons(self):
#         """Get the ORF sequence split into codons."""
#         orf = self._get_orf_seq()
#         return [orf[i:i+3] for i in range(0, len(orf), 3)]
    
#     def _calculate_seq_length(self) -> int:
#         """Calculate the output sequence length in nucleotides.
        
#         When mark_deletion=True: length stays same as orf_seq
#         When mark_deletion=False: length is (orf_codons - deletion_size) * 3
#         """
#         orf_codons = self._get_orf_codons()
#         num_codons = len(orf_codons)
        
#         if self.mark_deletion:
#             return num_codons * 3
#         else:
#             return (num_codons - self.deletion_size) * 3
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of deletion positions based on codon-level parameters.
        
#         Formula: (L - O - W + S) // S
#         where L=num_codons(orf), W=deletion_size, S=shift, O=offset%W
#         """
#         orf_codons = self._get_orf_codons()
#         L = len(orf_codons)
#         W = self.deletion_size
#         S = self.shift
        
#         # Validate that deletion_size is not longer than orf
#         if W > L:
#             raise ValueError(
#                 f"deletion_size ({W}) cannot be longer than "
#                 f"orf_seq codon count ({L})"
#             )
        
#         O = self.offset % W
#         num_states = (L - O - W + S) // S
        
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute sequence with deletion at current state codon position.
        
#         Maps state to a codon-level deletion position and either marks or removes
#         the deletion region at that position.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         orf_codons = self._get_orf_codons()
#         W = self.deletion_size
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate deletion position (in codon units)
#         O = self.offset % W
#         codon_pos = O + (state * self.shift)
        
#         # Perform deletion at codon level
#         if self.mark_deletion:
#             # Replace W codons at codon position codon_pos with deletion_character
#             # Each codon becomes a 3-character deletion marker
#             deletion_marker_codons = [self.deletion_character * 3 for _ in range(W)]
#             result_codons = (
#                 orf_codons[:codon_pos] + 
#                 deletion_marker_codons + 
#                 orf_codons[codon_pos + W:]
#             )
#         else:
#             # First create marked version, then remove all deletion_character occurrences
#             deletion_marker_codons = [self.deletion_character * 3 for _ in range(W)]
#             marked_codons = (
#                 orf_codons[:codon_pos] + 
#                 deletion_marker_codons + 
#                 orf_codons[codon_pos + W:]
#             )
#             # Join and remove all deletion_character occurrences
#             marked_result = ''.join(marked_codons)
#             result = marked_result.replace(self.deletion_character, '')
#             return result
        
#         return ''.join(result_codons)
    
#     def __repr__(self) -> str:
#         orf_str = self.orf_seq.seq if isinstance(self.orf_seq, Pool) else self.orf_seq
#         orf_preview = orf_str[:12] + "..." if len(orf_str) > 12 else orf_str
#         mark_str = "marked" if self.mark_deletion else "unmarked"
#         return f"DeletionScanORFPool(orf={orf_preview}, del_size={self.deletion_size}, mode={mark_str}, S={self.shift}, O={self.offset})"

