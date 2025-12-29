from poolparty import ListPool, KMutationPool

# Test 1: Variable-length parent (should have infinite states)
print("Test 1: Variable-length parent")
a_variable = ListPool(['AAA', 'TT', 'GGG', 'CCC'])
b_variable = KMutationPool(a_variable, alphabet='ACGT', k=1)
print(f"  a_variable.seq_length: {a_variable.seq_length}")
print(f"  b_variable.num_states: {b_variable.num_states}")
print(f"  Can use in combinatorial iteration: {b_variable.num_states < float('inf')}")

# Random generation should still work
random_seqs = b_variable.generate_seqs(num_seqs=10)
print(f"  Random sequences: {random_seqs}")
print()

# Test 2: Fixed-length parent (should have finite states)
print("Test 2: Fixed-length parent")
a_fixed = ListPool(['AAA', 'TTT', 'GGG', 'CCC'])
b_fixed = KMutationPool(a_fixed, alphabet='ACGT', k=1)
print(f"  a_fixed.seq_length: {a_fixed.seq_length}")
print(f"  b_fixed.num_states: {b_fixed.num_states}")
print(f"  Can use in combinatorial iteration: {b_fixed.num_states < float('inf')}")

# Both random and sequential generation should work
sequential_seqs = b_fixed.generate_seqs(combinatorially_complete_pools=[b_fixed, a_fixed], num_complete_iterations=1)
print(f"  Sequential sequences: {sequential_seqs}")
random_seqs = b_fixed.generate_seqs(num_seqs=10)
print(f"  Random sequences: {random_seqs}")
