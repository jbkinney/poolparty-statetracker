# Test RandomMutationORFPool: Generate ORF libraries with random codon-level mutations.
# This demonstrates different mutation types and integration with other pools.

from poolparty import (
    RandomMutationORFPool,
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
print("RandomMutationORFPool Testing")
print("=" * 80)
print("\nOriginal ORF sequence:")
print(f"DNA: {orf_seq}")
codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
print(f"Codons: {'-'.join(codons)}")
aas = [codon_to_aa_dict[c] for c in codons]
print(f"Translation: {'-'.join(aas)}")
print(f"Length: {len(codons)} codons ({len(orf_seq)} nucleotides)")

# ============================================================================
# Test 1: All-by-codon mutations with uniform mutation rate
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: All-by-codon mutations (uniform rate)")
print("=" * 80)
all_codon_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='all_by_codon', 
    mutation_rate=0.2
)
print(f"Mutation type: all_by_codon")
print(f"Mutation rate: {all_codon_pool.mutation_rate} (uniform)")
print(f"Total states: {all_codon_pool.num_states}")

print("\nFirst 5 all-by-codon variants:")
for i in range(5):
    all_codon_pool.set_state(i)
    seq = all_codon_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Count mutations
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    num_mutations = sum(1 for o, m in zip(orig_codons, codons) if o != m)
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")
    print(f"    Number of mutations: {num_mutations}/15 codons")

# ============================================================================
# Test 2: Missense mutations (first codon)
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Missense mutations (first codon) - uniform rate")
print("=" * 80)
missense_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='missense_first_codon', 
    mutation_rate=0.3
)
print(f"Mutation type: missense_first_codon")
print(f"Mutation rate: {missense_pool.mutation_rate}")

print("\nFirst 5 missense variants:")
orig_aas = [codon_to_aa_dict[orf_seq[i:i+3]] for i in range(0, len(orf_seq), 3)]
for i in range(5):
    missense_pool.set_state(i * 100)  # Use different states
    seq = missense_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Find positions where amino acid changed
    aa_changes = [idx for idx, (o, m) in enumerate(zip(orig_aas, aas)) if o != m]
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")
    print(f"    AA changes at positions: {aa_changes}")

# ============================================================================
# Test 3: Synonymous mutations (preserving amino acids)
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Synonymous mutations (preserving amino acids)")
print("=" * 80)
synonymous_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='synonymous', 
    mutation_rate=0.5
)
print(f"Mutation type: synonymous")
print(f"Mutation rate: {synonymous_pool.mutation_rate}")

print("\nFirst 5 synonymous variants (amino acids should be preserved):")
for i in range(5):
    synonymous_pool.set_state(i * 50)
    seq = synonymous_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Count codon mutations
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    num_codon_changes = sum(1 for o, m in zip(orig_codons, codons) if o != m)
    
    # Verify amino acids unchanged
    all_aas_same = (aas == orig_aas)
    
    print(f"  Variant {i+1}: {seq[:30]}...{seq[-15:]}")
    print(f"    Translation: {'-'.join(aas[:5])}...{'-'.join(aas[-3:])}")
    print(f"    Codon changes: {num_codon_changes}, AA preserved: {all_aas_same}")

# ============================================================================
# Test 4: Position-specific mutation rates
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: Position-specific mutation rates")
print("=" * 80)

# Create position-specific rates: high mutation at ends, low in middle
num_codons = len([orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)])
position_rates = [0.8] * 3 + [0.1] * (num_codons - 6) + [0.8] * 3
print(f"Position rates: {position_rates}")
print(f"  (High mutation rate at ends, low in middle)")

position_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='all_by_codon', 
    mutation_rate=position_rates
)

print("\nGenerating 10 variants with position-specific rates:")
seqs = position_pool.generate_seqs(num_seqs=10)
for i, seq in enumerate(seqs, 1):
    orig_codons = [orf_seq[j:j+3] for j in range(0, len(orf_seq), 3)]
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    
    # Find which positions mutated
    mutated_positions = [idx for idx, (o, m) in enumerate(zip(orig_codons, codons)) if o != m]
    
    print(f"  Variant {i:2d}: Mutations at codon positions {mutated_positions}")

# ============================================================================
# Test 5: Nonsense mutations (introducing stop codons)
# ============================================================================
print("\n" + "=" * 80)
print("Test 5: Nonsense mutations (introducing stop codons)")
print("=" * 80)
nonsense_pool = RandomMutationORFPool(
    orf_seq, 
    mutation_type='nonsense', 
    mutation_rate=0.15
)
print(f"Mutation type: nonsense")
print(f"Mutation rate: {nonsense_pool.mutation_rate}")

print("\nFirst 10 nonsense variants:")
for i in range(10):
    nonsense_pool.set_state(i * 10)
    seq = nonsense_pool.seq
    codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    aas = [codon_to_aa_dict[c] for c in codons]
    
    # Find stop codons
    stop_positions = [idx for idx, aa in enumerate(aas) if aa == '*']
    
    if stop_positions:
        print(f"  Variant {i+1}: Stop codons at positions {stop_positions}")
        print(f"    Translation: {'-'.join(aas)}")
    else:
        print(f"  Variant {i+1}: No stop codons introduced")

# ============================================================================
# Test 6: Integration with other pools - Building complete oligos
# ============================================================================
print("\n" + "=" * 80)
print("Test 6: Integration - Building complete oligos with primers and barcodes")
print("=" * 80)

# Create a smaller ORF for clearer output
small_orf = 'ATGGCCAAACCCGGG'  # 5 codons
orf_variants = RandomMutationORFPool(
    small_orf, 
    mutation_type='missense_first_codon', 
    mutation_rate=0.4
)

# Add fixed primers and variable barcodes
left_primer = 'AAAA.'
right_primer = '.TTTT'
barcodes = KmerPool(length=4, mode='random')

# Construct full oligo
oligo_pool = left_primer + orf_variants + '.' + barcodes + right_primer

print(f"Original ORF: {small_orf}")
print(f"ORF Translation: {'-'.join([codon_to_aa_dict[small_orf[i:i+3]] for i in range(0, len(small_orf), 3)])}")
print(f"Oligo structure: {left_primer}[ORF_variant].[barcode]{right_primer}")
print(f"Total combinations: {oligo_pool.num_states}")

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
# Test 7: Comparison of mutation rates
# ============================================================================
print("\n" + "=" * 80)
print("Test 7: Comparing different mutation rates")
print("=" * 80)

test_orf = 'ATGGCCAAACCCGGG'  # 5 codons
rates_to_test = [0.0, 0.2, 0.5, 0.8, 1.0]

print(f"Testing ORF: {test_orf}")
print(f"Mutation type: all_by_codon")
print(f"\nGenerating 20 variants at each mutation rate:\n")

for rate in rates_to_test:
    pool = RandomMutationORFPool(test_orf, mutation_type='all_by_codon', mutation_rate=rate)
    seqs = pool.generate_seqs(num_seqs=20)
    
    # Count mutations in each sequence
    orig_codons = [test_orf[i:i+3] for i in range(0, len(test_orf), 3)]
    mutation_counts = []
    for seq in seqs:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        num_muts = sum(1 for o, m in zip(orig_codons, codons) if o != m)
        mutation_counts.append(num_muts)
    
    avg_mutations = sum(mutation_counts) / len(mutation_counts)
    unique_seqs = len(set(seqs))
    
    print(f"  Rate {rate:.1f}: Avg mutations = {avg_mutations:.2f}/{len(orig_codons)}, "
          f"Unique sequences = {unique_seqs}/20")

# ============================================================================
# Test 8: Computation graph visualization
# ============================================================================
print("\n" + "=" * 80)
print("Test 8: Computation graph for integrated pool")
print("=" * 80)

results = oligo_pool.generate_seqs(num_seqs=3, return_computation_graph=True)
print("\nComputation graph:")
visualize_computation_graph(results['graph'], results['node_sequences'], show_first_only=True, seq_display_length=100)

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)

