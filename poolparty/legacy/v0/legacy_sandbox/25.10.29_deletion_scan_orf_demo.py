"""Demo script for DeletionScanORFPool class."""

from poolparty import DeletionScanORFPool

# Example 1: Basic marked deletion scanning
print("Example 1: Basic marked deletion scanning")
print("=" * 60)
orf = "ATGGCCAAACCCTTTGGG"  # 6 codons: ATG-GCC-AAA-CCC-TTT-GGG
pool = DeletionScanORFPool(
    orf, 
    deletion_size=2,  # Delete 2 codons at a time
    mark_deletion=True,
    deletion_character='-',
    shift=1,
    offset=0
)

print(f"ORF sequence: {orf}")
print(f"Number of codons: {len(orf)//3}")
print(f"Deletion size: 2 codons")
print(f"Number of states: {pool.num_states}\n")

for i in range(pool.num_states):
    pool.set_state(i)
    print(f"State {i}: {pool.seq}")

# Example 2: Unmarked deletion scanning (actual removal)
print("\n\nExample 2: Unmarked deletion scanning")
print("=" * 60)
pool2 = DeletionScanORFPool(
    orf, 
    deletion_size=2,  # Delete 2 codons at a time
    mark_deletion=False,  # Actually remove the codons
    shift=1,
    offset=0
)

print(f"ORF sequence: {orf}")
print(f"Deletion size: 2 codons (actually removed)")
print(f"Number of states: {pool2.num_states}\n")

for i in range(pool2.num_states):
    pool2.set_state(i)
    print(f"State {i}: {pool2.seq} (length: {len(pool2.seq)} nt)")

# Example 3: Single codon deletion with shift
print("\n\nExample 3: Single codon deletion with shift=2")
print("=" * 60)
pool3 = DeletionScanORFPool(
    orf, 
    deletion_size=1,  # Delete 1 codon at a time
    mark_deletion=True,
    deletion_character='X',
    shift=2,  # Skip every other codon position
    offset=0
)

print(f"ORF sequence: {orf}")
print(f"Deletion size: 1 codon, shift: 2")
print(f"Number of states: {pool3.num_states}\n")

for i in range(pool3.num_states):
    pool3.set_state(i)
    print(f"State {i}: {pool3.seq}")

# Example 4: With offset
print("\n\nExample 4: Deletion with offset=1")
print("=" * 60)
pool4 = DeletionScanORFPool(
    orf, 
    deletion_size=2,
    mark_deletion=True,
    deletion_character='-',
    shift=1,
    offset=1  # Start at codon position 1
)

print(f"ORF sequence: {orf}")
print(f"Deletion size: 2 codons, offset: 1")
print(f"Number of states: {pool4.num_states}\n")

for i in range(pool4.num_states):
    pool4.set_state(i)
    print(f"State {i}: {pool4.seq}")

print("\n\n" + "=" * 60)
print("Demo complete!")

