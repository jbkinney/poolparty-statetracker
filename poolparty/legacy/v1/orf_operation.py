"""ORFOperation - Base class for codon-level operations on open reading frames.

Provides common functionality for:
- ORF sequence validation (DNA, divisible by 3)
- Flanking region extraction and reassembly
- Codon splitting and utilities
- Hybrid codon table caching (efficient + flexible)

Key Features:
- Standard genetic code cached at class level (shared, efficient)
- Custom genetic codes built per instance (isolated, flexible)
- Opt-in codon tables (scan pools don't pay overhead)
- Universal flanking region support (realistic constructs)
"""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Union, List, Dict, Optional

from .operation import Operation

if TYPE_CHECKING:
    from .pool import Pool


class ORFOp(Operation):
    """Base class for all ORF-related operations.
    
    Subclasses must:
        - Call super().__init__() with appropriate codon_table setting
        - Set self.seq_length based on output sequence length
        - Set self.num_states based on state space
        - Implement compute_seq()
    
    The hybrid caching strategy provides:
    - 0ms overhead for standard genetic code (after first instance)
    - 5ms overhead for custom genetic codes (per instance)
    - Thread-safe by design (each instance has independent references)
    """
    
    op_name: str = 'orf_operation'
    
    # Class-level cache for standard genetic code (shared by all instances)
    _STANDARD_TABLES_CACHE: Optional[Dict] = None
    
    # Human codon usage table - codons sorted by frequency (high → low)
    # Source: Kazusa Codon Usage Database - Homo sapiens
    # https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606
    # 
    # This ordering is important for mutation types like 'missense_only_first'
    # and 'nonsynonymous_first' which select the first (most frequent) codon.
    STANDARD_GENETIC_CODE = {
        "F": ["TTC", "TTT"],
        "L": ["CTG", "CTC", "CTT", "TTG", "TTA", "CTA"],
        "I": ["ATC", "ATT", "ATA"],
        "M": ["ATG"],
        "V": ["GTG", "GTC", "GTT", "GTA"],
        "S": ["AGC", "TCC", "TCT", "TCA", "AGT", "TCG"],
        "P": ["CCC", "CCT", "CCA", "CCG"],
        "T": ["ACC", "ACA", "ACT", "ACG"],
        "A": ["GCC", "GCT", "GCA", "GCG"],
        "Y": ["TAC", "TAT"],
        "H": ["CAC", "CAT"],
        "Q": ["CAG", "CAA"],
        "N": ["AAC", "AAT"],
        "K": ["AAG", "AAA"],
        "D": ["GAC", "GAT"],
        "E": ["GAG", "GAA"],
        "C": ["TGC", "TGT"],
        "W": ["TGG"],
        "R": ["AGA", "AGG", "CGG", "CGC", "CGA", "CGT"],
        "G": ["GGC", "GGA", "GGG", "GGT"],
        "*": ["TGA", "TAA", "TAG"]
    }
    
    def __init__(
        self,
        parent_pools: list['Pool'],
        num_states: int,
        mode: str,
        seq_length: Optional[int] = None,
        name: Optional[str] = None,
        parent: Union['Pool', str] = None,
        orf_start: int = 0,
        orf_end: Optional[int] = None,
        codon_table: Union[dict, str, None] = 'standard',
        mark_changes: bool = False,
    ):
        """Initialize ORFOperation base class.
        
        Args:
            parent_pools: List of parent Pool objects for DAG traversal (required)
            num_states: Number of internal states (required)
            mode: Either 'random' or 'sequential' (required)
            seq_length: Sequence length (int for fixed, None for variable)
            name: Optional name for this operation
            parent: Input sequence (Pool or string). Can include flanking regions (UTRs).
            orf_start: Start index of ORF within seq (0-based, inclusive). Default: 0
            orf_end: End index of ORF within seq (0-based, exclusive). Default: len(seq)
            codon_table: Codon table specification:
                - None: Don't build codon tables (for scan pools that don't need them)
                - 'standard': Use cached standard genetic code (efficient, recommended)
                - dict: Custom genetic code as {AA: [codon, ...]} (builds per-instance)
            mark_changes: If True, apply swapcase() to changed codons/regions
        
        Raises:
            ValueError: If seq is not valid DNA, or ORF boundaries are invalid
        """
        # Handle Pool vs string input
        if isinstance(parent, str):
            self._input_seq = parent
            self._parent_pool = parent_pools[0] if parent_pools else None
        else:
            self._parent_pool = parent if parent is not None else (parent_pools[0] if parent_pools else None)
            self._input_seq = None  # Will resolve at compute time
        
        # Get the sequence for validation and setup
        seq = self._input_seq if self._input_seq is not None else parent.seq
        
        # Validate full sequence is DNA
        self._validate_dna_sequence(seq)
        
        # Handle ORF boundaries
        if orf_end is None:
            orf_end = len(seq)
        self._validate_orf_boundaries(seq, orf_start, orf_end)
        
        # Store boundary offsets for computing at runtime
        self.orf_start = orf_start
        self.orf_end = orf_end
        
        # Extract regions (for validation and initial setup)
        self.upstream_flank = seq[:orf_start]
        orf_seq = seq[orf_start:orf_end]
        self.downstream_flank = seq[orf_end:]
        
        # Validate ORF region is divisible by 3
        if len(orf_seq) % 3 != 0:
            raise ValueError(
                f"ORF region length must be divisible by 3, got {len(orf_seq)} "
                f"(orf_start={orf_start}, orf_end={orf_end})"
            )
        
        # Store ORF sequence
        self.orf_seq = orf_seq
        self.mark_changes = mark_changes
        
        # Build codon tables if requested
        self.codon_table = codon_table
        if codon_table is not None:
            if codon_table == 'standard':
                # Use class-level cached tables for efficiency
                self._use_cached_standard_tables()
            else:
                # Build custom tables for this instance
                self._build_codon_tables(codon_table)
                self._build_mutation_lookup()
        
        # Initialize base Operation class
        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
        )
    
    # =========================================================================
    # Codon Table Management
    # =========================================================================
    
    def _use_cached_standard_tables(self):
        """Use class-level cached standard genetic code tables.
        
        Builds tables once on first call, then reuses for all instances.
        Thread-safe: Each instance gets its own reference to shared data.
        """
        if ORFOp._STANDARD_TABLES_CACHE is None:
            # First time: build and cache at class level
            ORFOp._STANDARD_TABLES_CACHE = self._build_codon_tables_static(
                self.STANDARD_GENETIC_CODE
            )
        
        cached = ORFOp._STANDARD_TABLES_CACHE
        # Copy cached data so instances can mutate safely without affecting others
        self.aa_to_codon_dict = {
            aa: codons[:] for aa, codons in cached['aa_to_codon'].items()
        }
        self.codon_to_aa_dict = cached['codon_to_aa'].copy()
        self.codon_to_synonymous_dict = {
            codon: syns[:] for codon, syns in cached['synonymous'].items()
        }
        self.stop_codons = cached['stop_codons'][:]
        self.all_codons = cached['all_codons'][:]
        self.mutation_lookup = copy.deepcopy(cached['mutation_lookup'])
    
    @staticmethod
    def _build_codon_tables_static(aa_to_codon_dict: dict) -> dict:
        """Build all codon tables and return as a dict (for caching).
        
        Args:
            aa_to_codon_dict: Dict mapping amino acid -> list of codons
        
        Returns:
            Dict with keys: aa_to_codon, codon_to_aa, synonymous, 
                           stop_codons, all_codons, mutation_lookup
        """
        # Build basic tables
        codon_to_aa = {}
        for aa, codon_list in aa_to_codon_dict.items():
            for codon in codon_list:
                codon_to_aa[codon] = aa
        
        synonymous = {}
        for aa, codon_list in aa_to_codon_dict.items():
            for codon in codon_list:
                synonymous[codon] = [c for c in codon_list if c != codon]
        
        stop_codons = aa_to_codon_dict.get("*", [])
        all_codons = list(codon_to_aa.keys())
        
        # Build mutation lookup tables
        mutation_lookup = ORFOp._build_mutation_lookup_static(
            aa_to_codon_dict, codon_to_aa, synonymous, stop_codons, all_codons
        )
        
        return {
            'aa_to_codon': aa_to_codon_dict,
            'codon_to_aa': codon_to_aa,
            'synonymous': synonymous,
            'stop_codons': stop_codons,
            'all_codons': all_codons,
            'mutation_lookup': mutation_lookup
        }
    
    def _build_codon_tables(self, aa_to_codon_dict: dict):
        """Build codon tables for this instance (for custom genetic codes).
        
        Same logic as static version, but stores in instance attributes.
        
        Args:
            aa_to_codon_dict: Dict mapping amino acid -> list of codons
        """
        self.aa_to_codon_dict = aa_to_codon_dict
        
        self.codon_to_aa_dict = {}
        for aa, codon_list in aa_to_codon_dict.items():
            for codon in codon_list:
                self.codon_to_aa_dict[codon] = aa
        
        self.codon_to_synonymous_dict = {}
        for aa, codon_list in aa_to_codon_dict.items():
            for codon in codon_list:
                self.codon_to_synonymous_dict[codon] = [c for c in codon_list if c != codon]
        
        self.stop_codons = aa_to_codon_dict.get("*", [])
        self.all_codons = list(self.codon_to_aa_dict.keys())
    
    @staticmethod
    def _build_mutation_lookup_static(
        aa_to_codon: dict, 
        codon_to_aa: dict, 
        codon_to_syn: dict, 
        stops: list, 
        codons: list
    ) -> dict:
        """Build mutation lookup tables (static version for caching).
        
        Args:
            aa_to_codon: Dict mapping AA -> list of codons
            codon_to_aa: Dict mapping codon -> AA
            codon_to_syn: Dict mapping codon -> list of synonymous codons
            stops: List of stop codons
            codons: List of all codons
        
        Returns:
            Dict mapping mutation_type -> codon -> list of mutations
            
        Mutation types:
        - any_codon: All 63 other codons
        - nonsynonymous_first: First codon of each different amino acid (includes stop)
        - nonsynonymous_random: All codons encoding different amino acids (includes stop)
        - missense_only_first: First codon of each different AA, excluding stop
        - missense_only_random: All codons for different AAs, excluding stop
        - synonymous: Synonymous codons
        - nonsense: Stop codons for non-stop codons, empty for stops
        """
        lookup = {}
        
        # 1. any_codon: all other codons
        lookup['any_codon'] = {
            codon: [c for c in codons if c != codon]
            for codon in codons
        }
        
        # 2. nonsynonymous_first: different AA/stop, first codon
        lookup['nonsynonymous_first'] = {}
        for codon in codons:
            current_aa = codon_to_aa[codon]
            lookup['nonsynonymous_first'][codon] = [
                aa_to_codon[aa][0]
                for aa in aa_to_codon.keys()
                if aa != current_aa
            ]
        
        # 3. nonsynonymous_random: different AA/stop, all codons
        lookup['nonsynonymous_random'] = {}
        for codon in codons:
            current_aa = codon_to_aa[codon]
            mutations = []
            for aa, codon_list in aa_to_codon.items():
                if aa != current_aa:
                    mutations.extend(codon_list)
            lookup['nonsynonymous_random'][codon] = mutations
        
        # 4. missense_only_first: different AA, first codon, NO stop
        lookup['missense_only_first'] = {}
        for codon in codons:
            current_aa = codon_to_aa[codon]
            lookup['missense_only_first'][codon] = [
                aa_to_codon[aa][0]
                for aa in aa_to_codon.keys()
                if aa != current_aa and aa != '*'
            ]
        
        # 5. missense_only_random: different AA, all codons, NO stop
        lookup['missense_only_random'] = {}
        for codon in codons:
            current_aa = codon_to_aa[codon]
            mutations = []
            for aa, codon_list in aa_to_codon.items():
                if aa != current_aa and aa != '*':
                    mutations.extend(codon_list)
            lookup['missense_only_random'][codon] = mutations
        
        # 6. synonymous: synonymous codons
        lookup['synonymous'] = dict(codon_to_syn)
        
        # 7. nonsense: stop codons
        lookup['nonsense'] = {
            codon: [] if codon in stops else stops.copy()
            for codon in codons
        }
        
        return lookup
    
    def _build_mutation_lookup(self):
        """Build mutation lookup for this instance (for custom codes).
        
        Calls the static version with instance-level codon tables.
        """
        self.mutation_lookup = self._build_mutation_lookup_static(
            self.aa_to_codon_dict, self.codon_to_aa_dict,
            self.codon_to_synonymous_dict, self.stop_codons, self.all_codons
        )
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    @staticmethod
    def _validate_dna_sequence(seq: str, allow_lowercase: bool = False):
        """Validate sequence is valid DNA (ACGT only).
        
        Args:
            seq: DNA sequence to validate
            allow_lowercase: If True, accept lowercase bases
            
        Raises:
            ValueError: If seq is not a string or contains non-ACGT characters
        """
        if not isinstance(seq, str):
            raise ValueError("seq must be a string")
        alphabet = 'ACGTacgt' if allow_lowercase else 'ACGT'
        if not all(c in alphabet for c in seq):
            raise ValueError("seq must contain only ACGT characters")
    
    @staticmethod
    def _validate_orf_boundaries(seq: str, orf_start: int, orf_end: int):
        """Validate ORF boundaries are within sequence.
        
        Args:
            seq: Full DNA sequence
            orf_start: Start index of ORF (0-based, inclusive)
            orf_end: End index of ORF (0-based, exclusive)
            
        Raises:
            ValueError: If boundaries are invalid
        """
        if orf_start < 0:
            raise ValueError(f"orf_start must be >= 0, got {orf_start}")
        if orf_end > len(seq):
            raise ValueError(
                f"orf_end ({orf_end}) cannot exceed sequence length ({len(seq)})"
            )
        if orf_start >= orf_end:
            raise ValueError(
                f"orf_start ({orf_start}) must be < orf_end ({orf_end})"
            )
    
    # =========================================================================
    # Codon Utilities
    # =========================================================================
    
    @property
    def codons(self) -> list[str]:
        """Get ORF split into codons.
        
        Returns:
            List of 3-nucleotide codon strings
        """
        return [self.orf_seq[i:i+3] for i in range(0, len(self.orf_seq), 3)]
    
    @property
    def num_codons(self) -> int:
        """Get number of codons in ORF.
        
        Returns:
            Number of codons (ORF length / 3)
        """
        return len(self.orf_seq) // 3
    
    def _get_codons_from_seq(self, seq: str) -> list[str]:
        """Extract codons from a sequence using stored ORF boundaries.
        
        Args:
            seq: Full sequence (with flanks)
        
        Returns:
            List of codons from the ORF region
        """
        orf_seq = seq[self.orf_start:self.orf_end]
        return [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
    
    def _reassemble_with_flanks(self, mutated_orf: str, base_seq: str) -> str:
        """Reassemble full sequence with preserved flanking regions.
        
        Args:
            mutated_orf: Modified ORF sequence
            base_seq: Original full sequence (to extract flanks from)
            
        Returns:
            Full sequence: upstream_flank + mutated_orf + downstream_flank
        """
        upstream = base_seq[:self.orf_start]
        downstream = base_seq[self.orf_end:]
        return upstream + mutated_orf + downstream
    
    # =========================================================================
    # Operation Interface
    # =========================================================================
    
    def compute_seq(
        self, 
        input_strings: Sequence[str], 
        state: int
    ) -> str:
        """Compute output sequence.
        
        Base implementation returns the input sequence unchanged.
        Subclasses should override to apply their transformations.
        
        Args:
            input_strings: Resolved sequences from parent pools (in order)
            state: Internal state number for this operation
        
        Returns:
            Output sequence
        """
        return input_strings[0]
