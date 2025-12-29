import copy
from typing import Union, List, Dict
from .pool import Pool


class ORFPool(Pool):
    """Base class for all ORF-related pools.
    
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
    
    The hybrid caching strategy provides:
    - 0ms overhead for standard genetic code (after first instance)
    - 5ms overhead for custom genetic codes (per instance)
    - Thread-safe by design (each instance has independent references)
    """
    
    # Class-level cache for standard genetic code (shared by all instances)
    _STANDARD_TABLES_CACHE = None
    
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
    
    def __init__(self, 
                 seq: str,
                 orf_start: int = 0,
                 orf_end: int = None,
                 codon_table: Union[dict, str, None] = None,
                 mark_changes: bool = False,
                 parents: tuple = (),
                 op: str = 'orf_pool',
                 max_num_states: int = None,
                 mode: str = 'random',
                 iteration_order: int = None,
                 name: str = None,
                 allow_lowercase: bool = False,
                 metadata: str = 'features'):
        """Initialize ORFPool base class.
        
        Args:
            seq: Full DNA sequence (ACGT only). Can include flanking regions (UTRs).
            orf_start: Start index of ORF within seq (0-based, inclusive). Default: 0
            orf_end: End index of ORF within seq (0-based, exclusive). Default: len(seq)
            codon_table: Codon table specification:
                - None: Don't build codon tables (for scan pools that don't need them)
                - 'standard': Use cached standard genetic code (efficient, recommended)
                - dict: Custom genetic code as {AA: [codon, ...]} (builds per-instance)
            mark_changes: If True, apply swapcase() to changed codons/regions
            parents: Tuple of parents for Pool computation graph. Default: () (empty tuple)
                Subclasses should explicitly pass parents=(seq,) or parents=(orf_seq,) when 
                calling super().__init__() to track the input sequence as a dependency.
            op: Operation name for Pool (e.g., 'k_mutate_orf', 'deletion_scan_orf')
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential'. Default: 'random'
            iteration_order: Order for sequential iteration. Default: auto-assigned
            name: Optional name for the pool
            allow_lowercase: If True, accept lowercase bases (useful when chaining pools that marked changes via swapcase)
        
        Raises:
            ValueError: If seq is not valid DNA, or ORF boundaries are invalid
        """
        # Validate full sequence is DNA
        self._validate_dna_sequence(seq, allow_lowercase=allow_lowercase)
        
        # Handle ORF boundaries
        if orf_end is None:
            orf_end = len(seq)
        self._validate_orf_boundaries(seq, orf_start, orf_end)
        
        # Extract regions
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
        if codon_table is not None:
            if codon_table == 'standard':
                # Use class-level cached tables for efficiency
                self._use_cached_standard_tables()
            else:
                # Build custom tables for this instance
                self._build_codon_tables(codon_table)
                self._build_mutation_lookup()
        
        # Call Pool.__init__
        # Note: ORFPool behaves like Pool - it accepts parents as-is (default: empty tuple).
        # Subclasses should explicitly pass parents=(seq,) to track input dependencies.
        super().__init__(
            parents=parents,
            op=op,
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            metadata=metadata
        )
    
    def _use_cached_standard_tables(self):
        """Use class-level cached standard genetic code tables.
        
        Builds tables once on first call, then reuses for all instances.
        Thread-safe: Each instance gets its own reference to shared data.
        """
        if ORFPool._STANDARD_TABLES_CACHE is None:
            # First time: build and cache at class level
            ORFPool._STANDARD_TABLES_CACHE = self._build_codon_tables_static(
                self.STANDARD_GENETIC_CODE
            )
        
        cached = ORFPool._STANDARD_TABLES_CACHE
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
        mutation_lookup = ORFPool._build_mutation_lookup_static(
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
    def _build_mutation_lookup_static(aa_to_codon: dict, codon_to_aa: dict, 
                                     codon_to_syn: dict, stops: list, codons: list) -> dict:
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
        lookup['synonymous'] = codon_to_syn
        
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
    
    @staticmethod
    def _validate_dna_sequence(seq: str, allow_lowercase: bool = False):
        """Validate sequence is valid DNA (ACGT only).
        
        Args:
            seq: DNA sequence to validate
            
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
    
    @property
    def codons(self) -> List[str]:
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
    
    def _reassemble_with_flanks(self, mutated_orf: str) -> str:
        """Reassemble full sequence with preserved flanking regions.
        
        Args:
            mutated_orf: Modified ORF sequence
            
        Returns:
            Full sequence: upstream_flank + mutated_orf + downstream_flank
        """
        return self.upstream_flank + mutated_orf + self.downstream_flank
    
    def _calculate_seq_length(self) -> int:
        """Calculate output sequence length (ORF + flanks).
        
        Returns:
            Total length: len(upstream_flank) + len(orf_seq) + len(downstream_flank)
        """
        return len(self.upstream_flank) + len(self.orf_seq) + len(self.downstream_flank)
    
    def _compute_seq(self) -> str:
        """Compute the output sequence.
        
        Base implementation returns the full sequence unchanged (ORF + flanks).
        Subclasses should override this to apply their transformations (mutations, deletions, etc.).
        
        Note: We override Pool's default _compute_seq() which would raise ValueError 
        for our op='orf_pool'. Pool's default _calculate_num_internal_states() 
        already returns 1 for us, so we don't need to override it.
        
        Returns:
            Full sequence with flanks: upstream_flank + orf_seq + downstream_flank
        """
        return self._reassemble_with_flanks(self.orf_seq)

