"""
Demonstration of InsertionScanORFPool functionality.

This shows how to use InsertionScanORFPool to scan an insertion sequence across
an ORF background at the codon level, maintaining reading frame integrity.
"""

from poolparty import InsertionScanORFPool
from poolparty.utils import codon_to_aa_dict

print("=" * 80)
print("InsertionScanORFPool Demonstration")
print("=" * 80)

# Example: Small ORF representing a peptide tag
orf_seq = "ATGGCCAAACCCTTTGGG"  # 6 codons: ATG-GCC-AAA-CCC-TTT-GGG
insertion_seq = "CTGCTG"  # 2 codons: CTG-CTG (Leucine-Leucine)

# Display original sequence
print("\nOriginal ORF:")
print(f"DNA: {orf_seq}")
codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
print(f"Codons: {'-'.join(codons)}")
aas = [codon_to_aa_dict[c] for c in codons]
print(f"Translation: {'-'.join(aas)}")

print("\nInsertion sequence:")
print(f"DNA: {insertion_seq}")
ins_codons = [insertion_seq[i:i+3] for i in range(0, len(insertion_seq), 3)]
print(f"Codons: {'-'.join(ins_codons)}")
ins_aas = [codon_to_aa_dict[c] for c in ins_codons]
print(f"Translation: {'-'.join(ins_aas)}")

# ============================================================================
# Test 1: Overwrite mode (default) - replaces codons
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Overwrite mode - scanning with shift=1")
print("=" * 80)

pool = InsertionScanORFPool(
    orf_seq, 
    insertion_seq, 
    overwrite_insertion_site=True,
    shift=1,
    offset=0
)

print(f"\nTotal states: {pool.num_states}")
print(f"Mode: {'overwrite' if pool.overwrite_insertion_site else 'insert'}")
print(f"Shift: {pool.shift} codons")
print(f"Offset: {pool.offset} codons")

print("\nFirst 5 variants:")
for i in range(min(5, pool.num_states)):
    pool.set_state(i)
    seq = pool.seq
    seq_codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    seq_aas = [codon_to_aa_dict[c] for c in seq_codons]
    print(f"State {i}: {'-'.join(seq_codons)} -> {'-'.join(seq_aas)}")

# ============================================================================
# Test 2: Insert mode - adds codons without replacement
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Insert mode - adding codons at different positions")
print("=" * 80)

pool2 = InsertionScanORFPool(
    orf_seq, 
    insertion_seq, 
    overwrite_insertion_site=False,
    shift=2,  # Skip every other position
    offset=0
)

print(f"\nTotal states: {pool2.num_states}")
print(f"Mode: {'overwrite' if pool2.overwrite_insertion_site else 'insert'}")
print(f"Shift: {pool2.shift} codons")
print(f"Result length: {pool2.seq_length} nucleotides ({pool2.seq_length // 3} codons)")

print("\nAll variants:")
for i in range(pool2.num_states):
    pool2.set_state(i)
    seq = pool2.seq
    seq_codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    seq_aas = [codon_to_aa_dict[c] for c in seq_codons]
    print(f"State {i}: {'-'.join(seq_codons)} -> {'-'.join(seq_aas)}")

# ============================================================================
# Test 3: Case marking - visualize inserted codons
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Using change_case_of_insert to mark insertions")
print("=" * 80)

pool3 = InsertionScanORFPool(
    orf_seq, 
    insertion_seq, 
    overwrite_insertion_site=True,
    change_case_of_insert=True,  # Mark inserted codons
    shift=1,
    offset=0
)

print(f"\nInsertion marked with lowercase (original is uppercase)")
print("\nFirst 5 variants:")
for i in range(min(5, pool3.num_states)):
    pool3.set_state(i)
    seq = pool3.seq
    seq_codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    print(f"State {i}: {'-'.join(seq_codons)}")

# ============================================================================
# Test 4: Practical example - scanning epitope tag
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: Practical example - scanning a FLAG tag across a protein")
print("=" * 80)

# FLAG tag: DYKDDDDK (common epitope tag)
flag_seq = "GACTACAAGGACGACGACGACAAG"  # 8 codons encoding DYKDDDDK

# Small target protein
target_orf = "ATGGCCTTTAAACCCTTTGGGAAACCCTTTGGG"  # 11 codons

pool4 = InsertionScanORFPool(
    target_orf,
    flag_seq,
    overwrite_insertion_site=False,  # Insert, don't replace
    change_case_of_insert=True,  # Mark the tag
    shift=3,  # Insert every 3 codons
    offset=0
)

print(f"\nTarget protein: {len(target_orf) // 3} codons")
print(f"FLAG tag: {len(flag_seq) // 3} codons")
print(f"Total insertion positions: {pool4.num_states}")

flag_codons = [flag_seq[i:i+3] for i in range(0, len(flag_seq), 3)]
flag_aas = [codon_to_aa_dict[c] for c in flag_codons]
print(f"FLAG sequence: {''.join(flag_aas)}")

print("\nInserting FLAG at different positions (lowercase = FLAG tag):")
for i in range(pool4.num_states):
    pool4.set_state(i)
    seq = pool4.seq
    seq_codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
    # Show compact version
    compact = ''.join(['[' + c + ']' if c.islower() else c for c in seq_codons])
    print(f"Position {i}: {compact}")

print("\n" + "=" * 80)
print("Demo complete!")
print("=" * 80)

