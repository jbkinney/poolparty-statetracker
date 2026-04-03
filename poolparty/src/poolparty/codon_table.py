"""Codon table utilities for ORF-aware operations."""

from .types import Union

# Mutation types that are uniform for valid ORF sequences (no internal stop codons)
# Maps mutation_type -> number of alternatives per codon
# Note: missense_only_first and nonsense are uniform for non-stop codons only
UNIFORM_MUTATION_TYPES = {
    "any_codon": 63,  # All 64 - 1 (self)
    "nonsynonymous_first": 20,  # 21 AAs - 1 (current)
    "missense_only_first": 19,  # 20 non-stop AAs - 1 (current)
    "nonsense": 3,  # 3 stop codons
}

# All valid mutation types
VALID_MUTATION_TYPES = {
    "any_codon",
    "nonsynonymous_first",
    "nonsynonymous_random",
    "missense_only_first",
    "missense_only_random",
    "synonymous",
    "nonsense",
}

# Human codon usage table - codons sorted by frequency (high → low)
# Source: Kazusa Codon Usage Database - Homo sapiens
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606
#
# This ordering is important for mutation types like 'missense_only_first'
# and 'nonsynonymous_first' which select the first codon in each list
# (the most frequent when using the built-in standard genetic code).
STANDARD_GENETIC_CODE: dict[str, list[str]] = {
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
    "*": ["TGA", "TAA", "TAG"],
}


class CodonTable:
    """Codon table with pre-computed mutation lookup tables.

    Attributes:
        aa_to_codons: Dict mapping amino acid -> list of codons
        codon_to_aa: Dict mapping codon -> amino acid
        synonymous: Dict mapping codon -> list of synonymous codons
        stop_codons: List of stop codons
        all_codons: List of all codons
        mutation_lookup: Dict mapping mutation_type -> codon -> list of alternative codons
    """

    def __init__(self, genetic_code: Union[str, dict] = "standard"):
        """Initialize a CodonTable.

        Args:
            genetic_code: Either 'standard' to use the built-in frequency-ordered
                table, or a dict mapping amino acids to codon lists, e.g.
                ``{"M": ["ATG"], "F": ["TTC", "TTT"], ...}``. For custom dicts,
                the order of codons in each list matters: ``_first`` mutation
                types and ``codon_selection="first"`` pick the first codon in
                each list.
        """
        if genetic_code == "standard":
            aa_to_codons = STANDARD_GENETIC_CODE
        elif isinstance(genetic_code, dict):
            aa_to_codons = genetic_code
        else:
            raise ValueError(f"genetic_code must be 'standard' or a dict, got {type(genetic_code)}")

        # Build the lookup tables
        tables = self._build_tables(aa_to_codons)
        self.aa_to_codons = tables["aa_to_codons"]
        self.codon_to_aa = tables["codon_to_aa"]
        self.synonymous = tables["synonymous"]
        self.stop_codons = tables["stop_codons"]
        self.all_codons = tables["all_codons"]
        self.mutation_lookup = tables["mutation_lookup"]

    @staticmethod
    def _build_tables(aa_to_codons: dict) -> dict:
        """Build all codon tables from an AA -> codons mapping.

        Args:
            aa_to_codons: Dict mapping amino acid -> list of codons

        Returns:
            Dict with keys: aa_to_codons, codon_to_aa, synonymous,
                           stop_codons, all_codons, mutation_lookup
        """
        # Build codon -> AA mapping
        codon_to_aa = {}
        for aa, codon_list in aa_to_codons.items():
            for codon in codon_list:
                codon_to_aa[codon] = aa

        # Build synonymous codon lists
        synonymous = {}
        for aa, codon_list in aa_to_codons.items():
            for codon in codon_list:
                synonymous[codon] = [c for c in codon_list if c != codon]

        # Get the stop codons and all codons
        stop_codons = aa_to_codons.get("*", [])
        all_codons = list(codon_to_aa.keys())

        # Build the mutation lookup table
        mutation_lookup = CodonTable._build_mutation_lookup(
            aa_to_codons, codon_to_aa, synonymous, stop_codons, all_codons
        )

        return {
            "aa_to_codons": aa_to_codons,
            "codon_to_aa": codon_to_aa,
            "synonymous": synonymous,
            "stop_codons": stop_codons,
            "all_codons": all_codons,
            "mutation_lookup": mutation_lookup,
        }

    @staticmethod
    def _build_mutation_lookup(
        aa_to_codons: dict,
        codon_to_aa: dict,
        synonymous: dict,
        stop_codons: list,
        all_codons: list,
    ) -> dict:
        """Build mutation lookup tables for all mutation types.

        Mutation types:
        - any_codon: All 63 other codons (uniform)
        - nonsynonymous_first: First codon of each different amino acid including stop (uniform)
        - nonsynonymous_random: All codons encoding different amino acids including stop (non-uniform)
        - missense_only_first: First codon of each different AA, excluding stop (uniform)
        - missense_only_random: All codons for different AAs, excluding stop (non-uniform)
        - synonymous: Synonymous codons only (non-uniform)
        - nonsense: Stop codons for non-stop codons, empty for stops (uniform)

        Returns:
            Dict mapping mutation_type -> codon -> list of alternative codons
        """
        lookup: dict[str, dict[str, list[str]]] = {}

        # 1. any_codon: all other codons
        lookup["any_codon"] = {codon: [c for c in all_codons if c != codon] for codon in all_codons}

        # 2. nonsynonymous_first: different AA/stop, first codon
        lookup["nonsynonymous_first"] = {}
        for codon in all_codons:
            current_aa = codon_to_aa[codon]
            lookup["nonsynonymous_first"][codon] = [
                aa_to_codons[aa][0] for aa in aa_to_codons.keys() if aa != current_aa
            ]

        # 3. nonsynonymous_random: different AA/stop, all codons
        lookup["nonsynonymous_random"] = {}
        for codon in all_codons:
            current_aa = codon_to_aa[codon]
            mutations = []
            for aa, codon_list in aa_to_codons.items():
                if aa != current_aa:
                    mutations.extend(codon_list)
            lookup["nonsynonymous_random"][codon] = mutations

        # 4. missense_only_first: different AA, first codon, NO stop
        lookup["missense_only_first"] = {}
        for codon in all_codons:
            current_aa = codon_to_aa[codon]
            lookup["missense_only_first"][codon] = [
                aa_to_codons[aa][0] for aa in aa_to_codons.keys() if aa != current_aa and aa != "*"
            ]

        # 5. missense_only_random: different AA, all codons, NO stop
        lookup["missense_only_random"] = {}
        for codon in all_codons:
            current_aa = codon_to_aa[codon]
            mutations = []
            for aa, codon_list in aa_to_codons.items():
                if aa != current_aa and aa != "*":
                    mutations.extend(codon_list)
            lookup["missense_only_random"][codon] = mutations

        # 6. synonymous: synonymous codons (copy to avoid mutation)
        lookup["synonymous"] = {codon: syns[:] for codon, syns in synonymous.items()}

        # 7. nonsense: stop codons for non-stop codons
        lookup["nonsense"] = {
            codon: [] if codon in stop_codons else stop_codons[:] for codon in all_codons
        }

        return lookup

    def get_mutations(self, codon: str, mutation_type: str) -> list[str]:
        """Get available mutations for a codon."""
        if mutation_type not in self.mutation_lookup:
            valid_types = list(self.mutation_lookup.keys())
            raise ValueError(f"mutation_type must be one of {valid_types}, got '{mutation_type}'")
        return self.mutation_lookup[mutation_type][codon.upper()]

    def num_mutations(self, codon: str, mutation_type: str) -> int:
        """Get number of available mutations for a codon."""
        return len(self.get_mutations(codon, mutation_type))

    def is_uniform(self, mutation_type: str) -> bool:
        """Check if a mutation type has uniform alternatives across all codons."""
        counts = [len(alts) for alts in self.mutation_lookup[mutation_type].values()]
        return len(set(counts)) == 1

    def uniform_num_mutations(self, mutation_type: str) -> int | None:
        """Get the uniform number of mutations for a mutation type."""
        if not self.is_uniform(mutation_type):
            return None
        # All counts are the same, return the first
        first_codon = next(iter(self.mutation_lookup[mutation_type]))
        return len(self.mutation_lookup[mutation_type][first_codon])
