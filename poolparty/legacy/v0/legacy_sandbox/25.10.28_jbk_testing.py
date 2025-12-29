# Split a 1kb genomic region up into 100 nt segments with 20 nt overlap.
# Then generate mutations at 10% per nucleotide.
# Then add shared primer sequences to each end, with the right primer containing a 20 nt barcode. 

import random
from poolparty import SubseqPool, RandomMutationPool, KmerPool, DeletionScanPool, ShuffleScanPool, InsertionScanPool, DiShuffleScanPool, DiShufflePool
import textwrap
from poolparty import visualize_computation_graph, get_alphabet

dna_alphabet = get_alphabet('dna')

genomic_seq = ''.join([random.choice(dna_alphabet) for _ in range(100)])
segments_pool = SubseqPool(genomic_seq, width=50, shift=20, mode='sequential', iteration_order=1)
#variants_pool = RandomMutationPool(segments_pool, alphabet=dna_alphabet, mutation_rate=0.1, change_case_of_mutations=True)
#variants_pool = InsertionScanPool(segments_pool, insertion_seq="AAAAAAAA", change_case_of_insert=True, shift=5, offset=1, mode='sequential')
#variants_pool = DeletionScanPool(segments_pool, deletion_size=5, shift=5, offset=1, mode='sequential', mark_deletion=True, iteration_order=0)
#variants_pool = ShuffleScanPool(segments_pool, shuffle_size=5, shift=5, offset=1, change_case_of_shuffle=True, mode='sequential', iteration_order=2)
variants_pool = DiShuffleScanPool(segments_pool, shuffle_size=5, shift=5, offset=1, change_case_of_shuffle=True, mode='random', iteration_order=3)
barcodes_pool = KmerPool(length=7, mode='random')
#barcodes_pool = DiShufflePool(seq='ACTAGCTGA', mode='random')
left_fixed_seq = 'AAAAA.'
spacer_fixed_seq = '.BBBBB.'
right_fixed_seq = '.CCCCC'

oligo_pool = left_fixed_seq + variants_pool + spacer_fixed_seq + barcodes_pool + right_fixed_seq
#
print("Genomic sequence:")
for line in textwrap.wrap(genomic_seq, width=80):
    print(line)

num_to_print = 10
index_width = len(str(num_to_print))

print("\nSequences in segments_pool:")
seqs = segments_pool.generate_seqs(num_seqs=num_to_print)
for i, seq in enumerate(seqs, 1):
    print(f"Segment {i:0{index_width}}: {seq}")

print("\nSequences in variants_pool:")
seqs = variants_pool.generate_seqs(num_seqs=num_to_print)
for i, seq in enumerate(seqs, 1):
    print(f"Variant {i:0{index_width}}: {seq}")
    
print("\nSequences in barcodes_pool:")
seqs = barcodes_pool.generate_seqs(num_seqs=num_to_print)
for i, seq in enumerate(seqs, 1):
    print(f"Barcode {i:0{index_width}}: {seq}")

seqs = oligo_pool.generate_seqs(num_seqs=num_to_print)
print("\nSequences in oligo_pool:")
for i, seq in enumerate(seqs, 1):
    print(f"Oligo {i:0{index_width}}: {seq}")

results = oligo_pool.generate_seqs(num_seqs=num_to_print, return_computation_graph=True)
print("\nComputation graph:")
visualize_computation_graph(results['graph'], results['node_sequences'], show_first_only=True, seq_display_length=1000)
