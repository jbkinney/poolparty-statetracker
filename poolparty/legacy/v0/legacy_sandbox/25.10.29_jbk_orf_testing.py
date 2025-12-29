# Test KMutationORFPool: Generate a library of ORF variants with codon-level mutations.
# This demonstrates different mutation types and integration with other pools.

from poolparty import (
    KMutationORFPool,
    RandomMutationORFPool,
    InsertionScanORFPool,
    RandomMutationPool,
    KmerPool,
    Pool,
    visualize_computation_graph,
)
from poolparty.utils import codon_to_aa_dict
import textwrap

# Example ORF: A 15-codon gene (45 nucleotides)
# This could represent a small protein or peptide tag
orf_seq = 'ATGGCCAAACCCGGGTTTCTGCATCAGAACGACGAATGCTACCCCGGG'  # 15 codons
print("=" * 80)
print("KMutationORFPool Testing")
print("=" * 80)
print("\nOriginal ORF sequence:")
print(f"DNA: {orf_seq}")
codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
print(f"Codons: {'-'.join(codons)}")
aas = [codon_to_aa_dict[c] for c in codons]
print(f"Translation: {'-'.join(aas)}")
print(f"Length: {len(codons)} codons ({len(orf_seq)} nucleotides)")

# ============================================================================
# Test 1: Missense mutations with first codon (uniform, sequential compatible)
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Missense mutations (first codon) - k=2")
print("=" * 80)
missense_pool = KMutationORFPool(orf_seq, k=2, mutation_type='missense_first_codon', mode='sequential')
print(f"Mutation type: missense_first_codon")
print(f"Mode: {missense_pool.mode}")
print(f"Is uniform: {missense_pool.is_uniform}")
print(f"Alternatives per codon: {missense_pool.uniform_num_possible_mutations}")
print(f"Total states: {missense_pool.num_states:,}")

print("\nFirst 5 missense variants:")
for i in range(5):
    missense_pool.set_state(i)
    seq = missense_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Highlight mutated positions
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    mutated_pos = [idx for idx, (o, m) in enumerate(zip(orig_codons, codons)) if o != m]
    
    print(f"  Variant {i+1}: {seq}")
    print(f"    Translation: {'-'.join(aas)}")
    print(f"    Mutated codon positions: {mutated_pos}")

# ============================================================================
# Test 2: All-by-codon mutations (uniform, sequential compatible)
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: All-by-codon mutations - k=3")
print("=" * 80)
allcodon_pool = KMutationORFPool(orf_seq, k=3, mutation_type='all_by_codon', mode='sequential')
print(f"Mutation type: all_by_codon")
print(f"Alternatives per codon: {allcodon_pool.uniform_num_possible_mutations}")
print(f"Total states: {allcodon_pool.num_states:,}")

print("\nFirst 3 all-codon variants:")
for i in range(3):
    allcodon_pool.set_state(i)
    seq = allcodon_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")

# ============================================================================
# Test 3: Nonsense mutations (creating stop codons)
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Nonsense mutations (introducing stop codons) - k=1")
print("=" * 80)
nonsense_pool = KMutationORFPool(orf_seq, k=1, mutation_type='nonsense', mode='sequential')
print(f"Mutation type: nonsense")
print(f"Alternatives per codon: {nonsense_pool.uniform_num_possible_mutations}")
print(f"Total states: {nonsense_pool.num_states}")

print("\nFirst 5 nonsense variants (each introduces one stop codon):")
for i in range(5):
    nonsense_pool.set_state(i)
    seq = nonsense_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Find the stop codon
    stop_pos = [idx for idx, aa in enumerate(aas) if aa == '*']
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")
    print(f"    Stop codon at position: {stop_pos}")

# ============================================================================
# Test 4: Synonymous mutations (non-uniform, random mode only)
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: Synonymous mutations (preserving amino acids) - k=2")
print("=" * 80)
synonymous_pool = KMutationORFPool(orf_seq, k=2, mutation_type='synonymous', mode='random')
print(f"Mutation type: synonymous")
print(f"Is uniform: {synonymous_pool.is_uniform}")
print(f"Alternatives per codon: {synonymous_pool.num_possible_mutations}")
print(f"Total states: {synonymous_pool.num_states}")

print("\nFirst 5 synonymous variants (amino acids unchanged):")
for i in range(5):
    synonymous_pool.set_state(i)
    seq = synonymous_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")

# ============================================================================
# Test 5: Integration with other pools - Adding barcodes and primers
# ============================================================================
print("\n" + "=" * 80)
print("Test 5: Integration - ORF library with primers and barcodes")
print("=" * 80)

# Create a smaller ORF for clearer output
small_orf = 'ATGGCCAAACCCGGG'  # 5 codons
orf_variants = KMutationORFPool(small_orf, k=1, mutation_type='missense_first_codon', mode='sequential')

# Add fixed primers and variable barcodes
left_primer = 'AAAA.'
right_primer = '.TTTT'
barcodes = KmerPool(length=4, mode='random')

# Construct full oligo
oligo_pool = left_primer + orf_variants + '.' + barcodes + right_primer

print(f"Original ORF: {small_orf}")
print(f"ORF Translation: {'-'.join([codon_to_aa_dict[small_orf[i:i+3]] for i in range(0, len(small_orf), 3)])}")
print(f"Oligo structure: {left_primer}[ORF_variant].[barcode]{right_primer}")
print(f"Total combinations: {oligo_pool.num_states:,}")

print("\nFirst 10 complete oligos:")
seqs = oligo_pool.generate_seqs(num_seqs=10)
for i, seq in enumerate(seqs, 1):
    # Extract components
    orf_part = seq[5:20]  # After "AAAA." and 15nt ORF
    barcode_part = seq[21:25]  # After ORF and "."
    
    orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
    orf_aas = [codon_to_aa_dict[c] for c in orf_codons]
    
    print(f"  Oligo {i:2d}: {seq}")
    print(f"    ORF: {orf_part} -> {'-'.join(orf_aas)}, Barcode: {barcode_part}")

# ============================================================================
# Test 6: Computation graph visualization
# ============================================================================
print("\n" + "=" * 80)
print("Test 6: Computation graph for integrated pool")
print("=" * 80)

results = oligo_pool.generate_seqs(num_seqs=3, return_computation_graph=True)
print("\nComputation graph:")
visualize_computation_graph(results['graph'], results['node_sequences'], show_first_only=True, seq_display_length=100)

# ============================================================================
# Test 7: RandomMutationORFPool - Random number of codon mutations
# ============================================================================
print("\n" + "=" * 80)
print("Test 7: RandomMutationORFPool - Random codon-level mutations")
print("=" * 80)
print("\nComparison: KMutationORFPool vs RandomMutationORFPool")
print("  KMutationORFPool: Exactly k codon mutations (finite states)")
print("  RandomMutationORFPool: Random number of mutations (infinite states)")

# Test 7a: Missense mutations with uniform rate
print("\n" + "-" * 80)
print("Test 7a: Missense mutations with uniform mutation rate")
print("-" * 80)
rand_missense_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='missense_first_codon', 
    mutation_rate=0.25
)
print(f"Mutation type: missense_first_codon")
print(f"Mutation rate: {rand_missense_pool.mutation_rate} per codon")
print(f"Total states: {rand_missense_pool.num_states}")

print("\nFirst 5 random missense variants:")
orig_aas = [codon_to_aa_dict[orf_seq[i:i+3]] for i in range(0, len(orf_seq), 3)]
for i in range(5):
    rand_missense_pool.set_state(i * 10)
    seq = rand_missense_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Count mutations
    mutated_pos = [idx for idx, (o, m) in enumerate(zip(orig_aas, aas)) if o != m]
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")
    print(f"    Mutated positions: {mutated_pos} ({len(mutated_pos)} mutations)")

# Test 7b: Synonymous mutations with high rate
print("\n" + "-" * 80)
print("Test 7b: Synonymous mutations (preserving amino acids)")
print("-" * 80)
rand_synonymous_pool = RandomMutationORFPool(
    orf_seq,
    mutation_type='synonymous',
    mutation_rate=0.4
)
print(f"Mutation type: synonymous")
print(f"Mutation rate: {rand_synonymous_pool.mutation_rate} per codon")

print("\nFirst 5 synonymous variants (amino acids preserved):")
for i in range(5):
    rand_synonymous_pool.set_state(i * 20)
    seq = rand_synonymous_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Count codon changes
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    num_codon_changes = sum(1 for o, m in zip(orig_codons, codons) if o != m)
    aa_preserved = (aas == orig_aas)
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Codon changes: {num_codon_changes}, AA preserved: {aa_preserved}")

# Test 7c: Position-specific mutation rates
print("\n" + "-" * 80)
print("Test 7c: Position-specific mutation rates")
print("-" * 80)
num_codons = len([orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)])
# High mutation at ends, low in middle
position_rates = [0.8] * 3 + [0.1] * (num_codons - 6) + [0.8] * 3
print(f"Mutation rates per codon: {position_rates}")
print(f"  (High mutation rate at N- and C-terminus, low in middle)")

rand_position_pool = RandomMutationORFPool(
    orf_seq,
    mutation_type='all_by_codon',
    mutation_rate=position_rates
)

print("\nFirst 5 variants with position-specific rates:")
for i in range(5):
    rand_position_pool.set_state(i * 15)
    seq = rand_position_pool.seq
    
    # Find mutated positions
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    mutated_pos = [idx for idx, (o, m) in enumerate(zip(orig_codons, codons)) if o != m]
    
    print(f"  Variant {i+1}: Mutations at codon positions {mutated_pos}")
    print(f"    {seq[:30]}...{seq[-15:]}")

# Test 7d: Comparison of mutation rates
print("\n" + "-" * 80)
print("Test 7d: Effect of different mutation rates")
print("-" * 80)
test_orf = 'ATGGCCAAACCCGGG'  # 5 codons
rates = [0.0, 0.2, 0.5, 0.8, 1.0]

print(f"Testing with {test_orf}")
print(f"Generating 20 variants at each mutation rate:\n")

for rate in rates:
    pool = RandomMutationORFPool(test_orf, mutation_type='all_by_codon', mutation_rate=rate)
    seqs = pool.generate_seqs(num_seqs=20)
    
    # Count mutations
    orig_codons = [test_orf[i:i+3] for i in range(0, len(test_orf), 3)]
    mutation_counts = []
    for seq in seqs:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        num_muts = sum(1 for o, m in zip(orig_codons, codons) if o != m)
        mutation_counts.append(num_muts)
    
    avg_muts = sum(mutation_counts) / len(mutation_counts)
    unique_seqs = len(set(seqs))
    
    print(f"  Rate {rate:.1f}: Avg mutations = {avg_muts:.2f}/5, Unique seqs = {unique_seqs}/20")

# Test 7e: Integration example with RandomMutationORFPool
print("\n" + "-" * 80)
print("Test 7e: Integration with other pools")
print("-" * 80)
rand_orf_variants = RandomMutationORFPool(
    small_orf,
    mutation_type='missense_first_codon',
    mutation_rate=0.4
)

# Build oligo with random mutations
rand_oligo_pool = left_primer + rand_orf_variants + '.' + KmerPool(length=4) + right_primer
print(f"Oligo structure: {left_primer}[RandomMutationORF].[barcode]{right_primer}")
print(f"Total combinations: {rand_oligo_pool.num_states}")

print("\nFirst 5 oligos with random ORF mutations:")
rand_seqs = rand_oligo_pool.generate_seqs(num_seqs=5)
for i, seq in enumerate(rand_seqs, 1):
    orf_part = seq[5:20]
    barcode_part = seq[21:25]
    orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
    orf_aas = [codon_to_aa_dict[c] for c in orf_codons]
    print(f"  Oligo {i}: {seq}")
    print(f"    ORF: {orf_part} -> {'-'.join(orf_aas)}, Barcode: {barcode_part}")

# Test 7f: change_case_of_mutations feature
print("\n" + "-" * 80)
print("Test 7f: change_case_of_mutations (visual marking of mutations)")
print("-" * 80)

print("Demonstrating the change_case_of_mutations feature:")
print("  - Mutated codons appear in lowercase")
print("  - Unmutated codons remain in uppercase")
print("  - Makes mutations easy to spot visually\n")

# Create two pools: one with and one without case change
test_orf = 'ATGGCCAAACCCGGG'  # 5 codons: M-A-K-P-G
pool_normal = RandomMutationORFPool(
    test_orf,
    mutation_type='missense_first_codon',
    mutation_rate=0.6,
    change_case_of_mutations=False
)

pool_case_change = RandomMutationORFPool(
    test_orf,
    mutation_type='missense_first_codon',
    mutation_rate=0.6,
    change_case_of_mutations=True
)

print("Original ORF:", test_orf)
print("Translation:  M-A-K-P-G\n")

print("Comparison (5 variants each):")
for i in range(5):
    state = i * 20
    
    pool_normal.set_state(state)
    seq_normal = pool_normal.seq
    
    pool_case_change.set_state(state)
    seq_case_change = pool_case_change.seq
    
    # Get translations
    codons_normal = [seq_normal[j:j+3] for j in range(0, len(seq_normal), 3)]
    aas_normal = [codon_to_aa_dict[c] for c in codons_normal]
    
    codons_case = [seq_case_change[j:j+3] for j in range(0, len(seq_case_change), 3)]
    aas_case = [codon_to_aa_dict[c.upper()] for c in codons_case]
    
    print(f"  Variant {i+1}:")
    print(f"    Normal:      {seq_normal} -> {'-'.join(aas_normal)}")
    print(f"    Case change: {seq_case_change} -> {'-'.join(aas_case)}")
    print(f"                 ↑ lowercase codons = mutated")

# Demonstrate with position-specific rates
print("\n" + "-" * 80)
print("Using position-specific rates with change_case_of_mutations:")
print("-" * 80)

position_rates = [1.0, 0.0, 1.0, 0.0, 1.0]  # Alternate high/low
print(f"Mutation rates: {position_rates}")
print("  (Positions 0, 2, 4 will always mutate; 1, 3 won't)\n")

pool_position_case = RandomMutationORFPool(
    test_orf,
    mutation_type='all_by_codon',
    mutation_rate=position_rates,
    change_case_of_mutations=True
)

print("3 examples:")
for i in range(3):
    pool_position_case.set_state(i * 7)
    seq = pool_position_case.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    
    print(f"  Example {i+1}: {seq}")
    print(f"    Codons: {' '.join(codons)}")
    print(f"            ↑     ↑     ↑  (mutated positions)")

# ============================================================================
# Test 8: InsertionScanORFPool - Overwrite mode (scanning a peptide tag)
# ============================================================================
print("\n" + "=" * 80)
print("Test 8: InsertionScanORFPool - Overwrite mode (codon-level scanning)")
print("=" * 80)

# Use a smaller ORF for this demo
small_orf = 'ATGGCCAAACCCTTTGGG'  # 6 codons: M-A-K-P-F-G
tag_insert = 'CTGCTG'  # 2 codons: L-L

print(f"\nTarget ORF: {small_orf}")
orig_codons = [small_orf[j:j+3] for j in range(0, len(small_orf), 3)]
orig_aas = [codon_to_aa_dict[c] for c in orig_codons]
print(f"  Codons: {'-'.join(orig_codons)}")
print(f"  Translation: {'-'.join(orig_aas)}")

print(f"\nInsertion tag: {tag_insert}")
insert_codons = [tag_insert[j:j+3] for j in range(0, len(tag_insert), 3)]
insert_aas = [codon_to_aa_dict[c] for c in insert_codons]
print(f"  Codons: {'-'.join(insert_codons)}")
print(f"  Translation: {'-'.join(insert_aas)}")

# Overwrite mode: replaces 2 codons at each position
insertion_pool = InsertionScanORFPool(
    small_orf,
    tag_insert,
    overwrite_insertion_site=True,
    shift=1,  # Shift by 1 codon each time
    offset=0
)

print(f"\nMode: overwrite (replaces codons)")
print(f"Shift: {insertion_pool.shift} codon(s)")
print(f"Total scan positions: {insertion_pool.num_states}")

print("\nAll scan positions:")
for i in range(insertion_pool.num_states):
    insertion_pool.set_state(i)
    seq = insertion_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    print(f"  Position {i}: {'-'.join(codons)} -> {'-'.join(aas)}")

# ============================================================================
# Test 9: InsertionScanORFPool - Insert mode with case marking
# ============================================================================
print("\n" + "=" * 80)
print("Test 9: InsertionScanORFPool - Insert mode with case marking")
print("=" * 80)

# Use a FLAG-like tag
flag_tag = 'GACTACAAG'  # 3 codons: D-Y-K (part of FLAG tag)
target_orf = 'ATGGCCAAACCCTTTGGG'  # 6 codons: M-A-K-P-F-G

print(f"\nTarget protein: {target_orf}")
target_codons = [target_orf[j:j+3] for j in range(0, len(target_orf), 3)]
target_aas = [codon_to_aa_dict[c] for c in target_codons]
print(f"  Codons: {''.join(target_codons)}")
print(f"  Translation: {''.join(target_aas)}")

print(f"\nEpitope tag: {flag_tag}")
flag_codons = [flag_tag[j:j+3] for j in range(0, len(flag_tag), 3)]
flag_aas = [codon_to_aa_dict[c] for c in flag_codons]
print(f"  Codons: {''.join(flag_codons)}")
print(f"  Translation: {''.join(flag_aas)}")

# Insert mode: adds codons without replacing
insertion_pool_insert = InsertionScanORFPool(
    target_orf,
    flag_tag,
    overwrite_insertion_site=False,  # Insert, don't overwrite
    change_case_of_insert=True,  # Mark inserted codons
    shift=2,  # Skip every other position
    offset=0
)

print(f"\nMode: insert (adds codons, marked with lowercase)")
print(f"Shift: {insertion_pool_insert.shift} codon(s)")
print(f"Total scan positions: {insertion_pool_insert.num_states}")
print(f"Result length: {insertion_pool_insert.seq_length} nt ({insertion_pool_insert.seq_length // 3} codons)")

print("\nInsertion at different positions (lowercase = inserted tag):")
for i in range(insertion_pool_insert.num_states):
    insertion_pool_insert.set_state(i)
    seq = insertion_pool_insert.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    
    # Build display with uppercase/lowercase indication
    codon_display = []
    for codon in codons:
        if codon.islower():
            codon_display.append(f"{codon}")  # Lowercase (inserted)
        else:
            codon_display.append(codon)
    
    # Get amino acids (uppercase for translation lookup)
    aas = [codon_to_aa_dict[c.upper()] for c in codons]
    
    print(f"  Position {i}: {''.join(codon_display)}")
    print(f"             -> {''.join(aas)}")

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
