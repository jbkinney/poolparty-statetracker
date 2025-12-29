from typing import Union, List, Dict, Any
from .pool import Pool
from .orf_pool import ORFPool
import random

class InsertionScanORFPool(ORFPool):
    """Scan an insertion sequence across an ORF at the codon level.
    
    Performs scanning mutagenesis on open reading frames by either overwriting
    codons or inserting between codon positions. Operations work at the codon
    level to maintain reading frame integrity.
    
    Supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically scans positions using start, end, and step_size.
        All parameters are in codon units.
        
        When insert_or_overwrite='overwrite':
            Given L=num_codons(orf), W=num_codons(insertion):
            - Scans positions where window [pos, pos+W) fits within [start, end)
            - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
            
        When insert_or_overwrite='insert':
            Given L=num_codons(orf):
            - Can insert at positions 0 through L (inclusive)
            - num_internal_states = len(range(start, min(end, L) + 1, step_size))
    
    **Position-based interface:**
        Directly specifies explicit codon positions with optional importance weights.
    
    Inherits from ORFPool:
        - DNA validation and flanking region support
        - Codon splitting utilities
    """
    
    def __init__(self,
                 background_seq: Union[Pool, str],
                 insert_seq: Union[Pool, str],
                 start: int = None,
                 end: int = None,
                 step_size: int = None,
                 positions: List[int] = None,
                 position_weights: List[float] = None,
                 insert_or_overwrite: str = 'overwrite',
                 mark_changes: bool = False,
                 orf_start: int = 0,
                 orf_end: int = None,
                 mode: str = 'random',
                 max_num_states: int = None,
                 iteration_order: int = None,
                 name: str = None,
                 metadata: str = 'features'):
        """Initialize InsertionScanORFPool.
        
        Args:
            background_seq: Background ORF sequence (string or Pool). Can include flanking regions.
            insert_seq: Sequence to insert (string or Pool). Must be DNA with
                length divisible by 3.
            
            # Range-based interface (codon units, mutually exclusive with positions):
            start: Starting codon position for first insertion (default: 0)
            end: Ending codon position (exclusive). (default: num_codons)
            step_size: Codon positions to step between insertions (default: 1)
            
            # Position-based interface:
            positions: List of explicit codon positions to insert at. (default: None)
            position_weights: Optional weights for random sampling. (default: uniform)
            
            insert_or_overwrite: 'overwrite' to replace codons, 'insert' to insert
                between positions (default: 'overwrite')
            mark_changes: If True, swapcase() the inserted codons (default: False)
            orf_start: Start index of ORF within seq (nucleotide). Default: 0
            orf_end: End index of ORF within seq (nucleotide). Default: len(seq)
            mode: 'random' or 'sequential' (default: 'random')
            max_num_states: Maximum states before treating as infinite
            iteration_order: Order for sequential iteration
            name: Optional pool name
        """
        # Handle Pool vs string for background
        if isinstance(background_seq, Pool):
            actual_seq = background_seq.seq
            parents_list = [background_seq]
            allow_lowercase = True
        else:
            actual_seq = background_seq
            parents_list = [background_seq]
            allow_lowercase = False
        
        # Handle Pool vs string for insertion
        if isinstance(insert_seq, Pool):
            insertion_str = insert_seq.seq
            parents_list.append(insert_seq)
        else:
            insertion_str = insert_seq
            parents_list.append(insert_seq)
        
        # Validate insert_seq is valid DNA with length divisible by 3
        if not isinstance(insertion_str, str):
            raise ValueError("insert_seq must be a string or Pool")
        if not all(c in 'ACGTacgt' for c in insertion_str):
            raise ValueError("insert_seq must contain only ACGT characters")
        if len(insertion_str) % 3 != 0:
            raise ValueError(
                f"insert_seq length must be divisible by 3, got {len(insertion_str)}"
            )
        
        # Initialize ORFPool
        super().__init__(
            seq=actual_seq,
            orf_start=orf_start,
            orf_end=orf_end,
            codon_table=None,  # Scan pools don't need codon lookups
            mark_changes=mark_changes,
            parents=tuple(parents_list),
            op='insertion_scan_orf',
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            allow_lowercase=allow_lowercase,
            metadata=metadata
        )
        
        # Store insertion info
        self.insert_seq = insert_seq
        self._insertion_str = insertion_str
        self.insertion_codons = [insertion_str[i:i+3] for i in range(0, len(insertion_str), 3)]
        self.num_insertion_codons = len(self.insertion_codons)
        
        # Validate insert_or_overwrite
        if insert_or_overwrite not in ['insert', 'overwrite']:
            raise ValueError(
                f"insert_or_overwrite must be 'insert' or 'overwrite', "
                f"got '{insert_or_overwrite}'"
            )
        self.insert_or_overwrite = insert_or_overwrite
        
        # Validate insertion size for overwrite mode
        if insert_or_overwrite == 'overwrite' and self.num_insertion_codons > self.num_codons:
            raise ValueError(
                f"insertion codon count ({self.num_insertion_codons}) cannot exceed "
                f"ORF codon count ({self.num_codons}) in overwrite mode"
            )
        
        # Determine interface type
        range_params_provided = any(p is not None for p in [start, end, step_size])
        position_params_provided = positions is not None
        
        if range_params_provided and position_params_provided:
            raise ValueError(
                "Cannot specify both range-based and position-based parameters."
            )
        
        if position_weights is not None and positions is None:
            raise ValueError("position_weights requires positions to be specified")
        
        if mode == 'sequential' and position_weights is not None:
            raise ValueError(
                "Cannot specify position_weights with mode='sequential'."
            )
        
        # Set up interface
        L = self.num_codons
        W = self.num_insertion_codons
        
        if position_params_provided:
            # Position-based interface
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            
            # Validate positions
            for pos in positions:
                if not isinstance(pos, int):
                    raise ValueError(f"All positions must be integers, got {type(pos)}")
                if insert_or_overwrite == 'overwrite':
                    if pos < 0 or pos + W > L:
                        raise ValueError(
                            f"Position {pos} invalid: window [{pos}, {pos + W}) "
                            f"must fit within ORF ({L} codons)"
                        )
                else:  # insert
                    if pos < 0 or pos > L:
                        raise ValueError(
                            f"Position {pos} invalid: must be in [0, {L}] for insert mode"
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
            self.end = end if end is not None else L
            self.step_size = step_size if step_size is not None else 1
            
            # Compute positions
            end_boundary = min(self.end, L)
            if insert_or_overwrite == 'overwrite':
                self.positions = list(range(self.start, end_boundary - W + 1, self.step_size))
            else:  # insert
                self.positions = list(range(self.start, end_boundary + 1, self.step_size))

            if len(self.positions) == 0:
                raise ValueError(
                    f"Range [start={self.start}, end={end_boundary}) with "
                    f"step_size={self.step_size} produces no valid insertion positions"
                )
        # Update state counts
        self.num_internal_states = self._calculate_num_internal_states()
        self.num_states = self._calculate_num_states()
        
        # Design cards: cached values from last _compute_seq call
        self._cached_codon_pos: int | None = None
        self._cached_insert: str | None = None
        self._cached_insert_aa: str | None = None
        self._cached_state: int | None = None
    
    def _get_insertion_codons(self) -> List[str]:
        """Get current insertion codons (handles Pool parent)."""
        if isinstance(self.insert_seq, Pool):
            ins_str = self.insert_seq.seq
            return [ins_str[i:i+3] for i in range(0, len(ins_str), 3)]
        return self.insertion_codons
    
    def _calculate_num_internal_states(self) -> int:
        """Number of insertion positions."""
        if not hasattr(self, 'positions') or self.positions is None:
            return 1  # Temporary during __init__
        return max(0, len(self.positions))
    
    def _calculate_seq_length(self) -> int:
        """Output sequence length."""
        base_len = len(self.upstream_flank) + len(self.downstream_flank)
        if self.insert_or_overwrite == 'overwrite':
            return base_len + len(self.orf_seq)
        else:
            return base_len + len(self.orf_seq) + len(self._insertion_str)
    
    def _compute_seq(self) -> str:
        """Compute sequence with insertion at current state."""
        # Get current ORF codons (handle transformer pattern)
        if isinstance(self.parents[0], Pool):
            parent_seq = self.parents[0].seq
            orf_start_idx = len(self.upstream_flank)
            orf_end_idx = orf_start_idx + len(self.orf_seq)
            current_orf = parent_seq[orf_start_idx:orf_end_idx]
            base_codons = [current_orf[i:i+3] for i in range(0, len(current_orf), 3)]
        else:
            base_codons = self.codons.copy()
        
        # Get current insertion codons
        insertion_codons = self._get_insertion_codons()
        W = len(insertion_codons)
        
        # Determine position
        if self.use_positions and self.mode == 'random':
            rng = random.Random(self.get_state())
            pos = rng.choices(self.positions, weights=self.position_probabilities)[0]
        else:
            state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
            pos = self.positions[state]
        
        # Cache for design cards
        self._cached_codon_pos = pos
        self._cached_insert = ''.join(insertion_codons)
        # Translate insertion to amino acids (if codon_table available)
        if hasattr(self, 'codon_table') and self.codon_table:
            self._cached_insert_aa = ''.join(
                self.codon_table.get(c.upper(), 'X') for c in insertion_codons
            )
        else:
            self._cached_insert_aa = None
        self._cached_state = self.get_state()
        
        # Apply mark_changes if requested
        if self.mark_changes:
            insertion_to_use = [c.swapcase() for c in insertion_codons]
        else:
            insertion_to_use = insertion_codons
        
        # Perform insertion or overwrite
        if self.insert_or_overwrite == 'overwrite':
            result_codons = base_codons[:pos] + insertion_to_use + base_codons[pos + W:]
        else:
            result_codons = base_codons[:pos] + insertion_to_use + base_codons[pos:]
        
        return self._reassemble_with_flanks(''.join(result_codons))
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this InsertionScanORFPool at the current state.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + codon_pos, codon_pos_abs, insert, insert_aa (default)
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
        
        # Add InsertionScanORFPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['codon_pos'] = self._cached_codon_pos
            # Compute absolute codon position (accounting for upstream flank)
            if abs_start is not None:
                flank_len = len(self.upstream_flank) if hasattr(self, 'upstream_flank') else 0
                metadata['codon_pos_abs'] = abs_start + flank_len + (self._cached_codon_pos * 3)
            else:
                metadata['codon_pos_abs'] = None
            metadata['insert'] = self._cached_insert
            metadata['insert_aa'] = self._cached_insert_aa
        
        return metadata
    
    def __repr__(self) -> str:
        orf_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
        ins_preview = self._insertion_str[:12] + "..." if len(self._insertion_str) > 12 else self._insertion_str
        mode_str = self.insert_or_overwrite
        if self.use_positions:
            return f"InsertionScanORFPool(orf={orf_preview}, ins={ins_preview}, mode={mode_str}, positions={self.positions})"
        else:
            return f"InsertionScanORFPool(orf={orf_preview}, ins={ins_preview}, mode={mode_str}, start={self.start}, end={self.end}, step={self.step_size})"

# Legacy
# class InsertionScanORFPool(Pool):
#     """A class for scanning an insertion sequence across an ORF background at the codon level.
    
#     Performs scanning mutagenesis on open reading frames by either overwriting codons in the
#     background ORF or inserting between codon positions. The shift and offset parameters work
#     at the codon level (not nucleotide level) to maintain reading frame integrity.
    
#     Both orf_seq and insertion_seq must be DNA sequences (ACGT only) with lengths divisible by 3.
    
#     The pool always has a finite number of states determined by the background codon count,
#     insertion codon count, shift, and offset parameters.
    
#     When overwrite_insertion_site=True:
#         Given L=num_codons(orf), W=num_codons(insertion), S=shift, and O=offset%W:
#         num_states = (L - O - W + S) // S
        
#     When overwrite_insertion_site=False:
#         Given L=num_codons(orf), S=shift, and O=offset:
#         num_states = (L - O + 1 + S) // S
#     """
    
#     def __init__(self, 
#                  orf_seq: str,
#                  insertion_seq: str,
#                  overwrite_insertion_site: bool = True,
#                  change_case_of_insert: bool = False,
#                  mode: str = 'random',
#                  shift: int = 1,
#                  offset: int = 0,
#                  max_num_states: int = None,
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize an InsertionScanORFPool.
        
#         Args:
#             orf_seq: Background ORF sequence (must be ACGT only and length divisible by 3)
#             insertion_seq: Sequence to insert or overwrite with (must be ACGT only and length divisible by 3)
#             overwrite_insertion_site: If True, replace codons; if False, insert between codon positions (default: True)
#             change_case_of_insert: If True, apply swapcase() to the inserted codons (default: False)
#             mode: Either 'random' or 'sequential' (default: 'random')
#             shift: Number of codon positions to shift between adjacent insertions (default: 1)
#             offset: Starting codon position offset for first insertion (default: 0)
#             max_num_states: Maximum number of states before treating as infinite
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
            
#         Raises:
#             ValueError: If orf_seq or insertion_seq are not valid DNA sequences or not divisible by 3
#         """
#         # Validate orf_seq is DNA
#         if not isinstance(orf_seq, str):
#             raise ValueError("orf_seq must be a string")
        
#         if not all(c in 'ACGTacgt' for c in orf_seq):
#             raise ValueError(f"orf_seq must contain only ACGT characters, got '{orf_seq}'")
        
#         # Validate length is divisible by 3
#         if len(orf_seq) % 3 != 0:
#             raise ValueError(
#                 f"orf_seq length must be divisible by 3, got length {len(orf_seq)}"
#             )
        
#         # Validate insertion_seq is DNA
#         if not isinstance(insertion_seq, str):
#             raise ValueError("insertion_seq must be a string")
        
#         if not all(c in 'ACGTacgt' for c in insertion_seq):
#             raise ValueError(f"insertion_seq must contain only ACGT characters, got '{insertion_seq}'")
        
#         # Validate insertion length is divisible by 3
#         if len(insertion_seq) % 3 != 0:
#             raise ValueError(
#                 f"insertion_seq length must be divisible by 3, got length {len(insertion_seq)}"
#             )
        
#         self.orf_seq = orf_seq
#         self.insertion_seq = insertion_seq
#         self.overwrite_insertion_site = overwrite_insertion_site
#         self.change_case_of_insert = change_case_of_insert
#         self.shift = shift
#         self.offset = offset
        
#         # Split into codons
#         self.orf_codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
#         self.insertion_codons = [insertion_seq[i:i+3] for i in range(0, len(insertion_seq), 3)]
        
#         super().__init__(
#             parents=(orf_seq, insertion_seq), 
#             op='insertion_scan_orf', 
#             max_num_states=max_num_states, 
#             mode=mode, 
#             iteration_order=iteration_order,
#             name=name
#         )
    
#     def _calculate_seq_length(self) -> int:
#         """Calculate the output sequence length in nucleotides.
        
#         When overwrite_insertion_site=True: length stays same as orf_seq
#         When overwrite_insertion_site=False: length is orf_seq + insertion_seq
#         """
#         orf_len = len(self.orf_seq)
#         if self.overwrite_insertion_site:
#             return orf_len
#         else:
#             insertion_len = len(self.insertion_seq)
#             return orf_len + insertion_len
    
#     def _calculate_num_internal_states(self) -> int:
#         """Calculate number of insertion positions based on codon-level parameters.
        
#         When overwrite_insertion_site=True:
#             Formula: (L - O - W + S) // S
#             where L=num_codons(orf), W=num_codons(insertion), S=shift, O=offset%W
            
#         When overwrite_insertion_site=False:
#             Formula: (L - O + 1 + S) // S
#             where L=num_codons(orf), S=shift, O=offset
#         """
#         L = len(self.orf_codons)
#         W = len(self.insertion_codons)
#         S = self.shift
        
#         # Validate that insertion is not longer than orf in overwrite mode
#         if self.overwrite_insertion_site and W > L:
#             raise ValueError(
#                 f"insertion_seq codon count ({W}) cannot be longer than "
#                 f"orf_seq codon count ({L}) when overwrite_insertion_site=True"
#             )
        
#         if self.overwrite_insertion_site:
#             O = self.offset % W
#             num_states = (L - O - W + S) // S
#         else:
#             O = self.offset
#             num_states = (L - O + 1 + S) // S
        
#         return max(0, num_states)  # Ensure non-negative
    
#     def _compute_seq(self) -> str:
#         """Compute sequence with insertion at current state codon position.
        
#         Maps state to a codon-level insertion position and either overwrites or inserts
#         the insertion codons at that position.
#         Uses state % num_internal_states to ensure bounds are never exceeded.
#         """
#         W = len(self.insertion_codons)
        
#         # Ensure state stays within valid range
#         state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        
#         # Calculate insertion position (in codon units)
#         if self.overwrite_insertion_site:
#             O = self.offset % W
#         else:
#             O = self.offset
        
#         codon_pos = O + (state * self.shift)
        
#         # Get insertion codons, optionally with case change
#         if self.change_case_of_insert:
#             insertion_codons_to_use = [codon.swapcase() for codon in self.insertion_codons]
#         else:
#             insertion_codons_to_use = self.insertion_codons.copy()
        
#         # Perform insertion or overwrite at codon level
#         if self.overwrite_insertion_site:
#             # Replace W codons at codon position codon_pos
#             result_codons = (
#                 self.orf_codons[:codon_pos] + 
#                 insertion_codons_to_use + 
#                 self.orf_codons[codon_pos + W:]
#             )
#         else:
#             # Insert between codon positions
#             result_codons = (
#                 self.orf_codons[:codon_pos] + 
#                 insertion_codons_to_use + 
#                 self.orf_codons[codon_pos:]
#             )
        
#         return ''.join(result_codons)
    
#     def __repr__(self) -> str:
#         orf_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
#         insertion_preview = self.insertion_seq[:12] + "..." if len(self.insertion_seq) > 12 else self.insertion_seq
#         mode_str = "overwrite" if self.overwrite_insertion_site else "insert"
#         return f"InsertionScanORFPool(orf={orf_preview}, ins={insertion_preview}, mode={mode_str}, S={self.shift}, O={self.offset})"

