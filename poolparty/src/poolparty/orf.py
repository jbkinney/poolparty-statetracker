
# Mutation types that are uniform for valid ORF sequences (no internal stop codons)
# Maps mutation_type -> number of alternatives per codon
# Note: missense_only_first and nonsense are uniform for non-stop codons only
UNIFORM_MUTATION_TYPES = {
    'any_codon': 63,           # All 64 - 1 (self)
    'nonsynonymous_first': 20, # 21 AAs - 1 (current)
    'missense_only_first': 19, # 20 non-stop AAs - 1 (current)
    'nonsense': 3,             # 3 stop codons
}

# All valid mutation types
VALID_MUTATION_TYPES = {
    'any_codon',
    'nonsynonymous_first',
    'nonsynonymous_random',
    'missense_only_first',
    'missense_only_random',
    'synonymous',
    'nonsense',
}

# Human codon usage table - codons sorted by frequency (high → low)
# Source: Kazusa Codon Usage Database - Homo sapiens
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606
#
# This ordering is important for mutation types like 'missense_only_first'
# and 'nonsynonymous_first' which select the first (most frequent) codon.
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