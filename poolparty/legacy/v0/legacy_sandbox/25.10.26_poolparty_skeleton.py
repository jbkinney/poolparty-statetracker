from typing import Tuple, List, Union
import random
import itertools    

class Pool:
    """A class representing an oligonucleotide sequence with lazy evaluation.
    
    Supports sequence concatenation (+) and repetition (*) operations through
    a computation graph structure. Properly handles cases where the same Pool
    appears multiple times in the computation graph.
    """
    def __init__(self, 
                 seq: str | None = None, 
                 parents: Tuple[Union['Pool', str, int],...] = (), 
                 op: str | None = None):
        self._seq = seq
        self.parents = parents
        self.op = op
        self._state = 0
        
        # Collect all unique Pool ancestors (primitive pools with independent state)
        # This ensures each unique ancestor is counted only once
        self.ancestors = self._collect_ancestors()
        
        # Calculate total number of states based on unique ancestors only
        self.num_states = self._calculate_num_states()
        
    def _collect_ancestors(self) -> list:
        """Collect all unique primitive Pool ancestors in the computation graph.
        
        Returns a sorted list of all Pool objects that have independent state. 
        For composite Pools (those created by operations like +, *, etc.), 
        recursively collects ancestors from parents. For primitive Pools (leaf nodes), 
        returns [self].
        
        The list is sorted by id() to ensure deterministic ordering for state 
        decomposition. This ordering is computed once during initialization.
        """
        if not self.parents:
            # This is a leaf/primitive pool with independent state
            return [self]
        
        # Composite pool: collect unique ancestors from all Pool parents
        ancestors_set = set()
        for parent in self.parents:
            if isinstance(parent, Pool):
                # Parent ancestors are already lists, add them to set for uniqueness
                ancestors_set.update(parent.ancestors)
        
        # Sort by id() for deterministic ordering
        return sorted(ancestors_set, key=id)
    
    def _calculate_num_states(self) -> int:
        """Calculate total number of distinct states based on unique ancestors.
        
        For primitive pools (SequentialPool): returns finite number (overridden)
        For stochastic pools (RandomPool, etc.): returns -1 for infinite (overridden)
        For composite pools: product of unique ancestor states if all finite, else -1
        """
        if not self.parents:
            # Base Pool with fixed sequence has 1 state (can be overridden)
            return 1
        
        # For composite pools, multiply states of unique ancestors
        total = 1
        for ancestor in self.ancestors:
            if ancestor.num_states == -1:
                return -1  # Any infinite ancestor makes the whole pool infinite
            total *= ancestor.num_states
        return total
        
    @property
    def seq(self) -> str:
        if self._seq is None:
            return self._compute_seq()
        else:
            return self._seq
    
    def is_finite(self) -> bool:
        """Check if this pool has a finite number of states."""
        return self.num_states > 0
        
    def set_state(self, state: int) -> None:
        """Set the state using mixed-radix decomposition over unique ancestors.
        
        Decomposes the global state index into individual states for each unique
        ancestor Pool. This ensures each ancestor's state is set exactly once,
        even if it appears multiple times in the computation graph.
        
        The ancestor ordering was determined during initialization in _collect_ancestors(),
        so we just use it directly here.
        """
        self._state = state
        
        if not self.ancestors:
            return
        
        # Decompose state using mixed-radix numeral system
        # Iterate in reverse (least significant digit first)
        remaining = state
        for ancestor in reversed(self.ancestors):
            if ancestor.num_states > 0:
                ancestor_state = remaining % ancestor.num_states
                ancestor._state = ancestor_state
                remaining //= ancestor.num_states
            else:
                # For infinite pools, pass state through
                ancestor._state = remaining

    def _advance_state(self) -> None:
        """Advance to the next state."""
        self.set_state(self._state + 1)

    def _compute_seq(self) -> str:
        match self.op, self.parents:
            case None, ():
                return self.seq
            case '+', (Pool(), str()):
                return self.parents[0].seq + self.parents[1]
            case '+', (Pool(), Pool()):
                return self.parents[0].seq + self.parents[1].seq
            case '*', (Pool(), int()):
                return self.parents[0].seq * self.parents[1]
            case 'slice', (Pool(), slice()):
                return self.parents[0].seq[self.parents[1]]
            case _:
                raise ValueError(f"Invalid op: {self.op}")
    
    def __next__(self) -> str:
        """Get current sequence and advance to next state."""
        seq = self.seq
        self._advance_state()
        return seq
    
    def __iter__(self):
        """Enable iteration through all states (if finite)."""
        if self.is_finite():
            for state in range(self.num_states):
                self.set_state(state)
                yield self.seq
        else:
            # For infinite pools, iterate indefinitely
            state = 0
            while True:
                self.set_state(state)
                yield self.seq
                state += 1

    def __add__(self, other: 'Pool') -> 'Pool':
        return Pool(parents=(self, other), op='+')
    
    def __radd__(self, other: 'Pool') -> 'Pool':
        return self.__add__(other)
    
    def __mul__(self, other: int) -> 'Pool':
        return Pool(parents=(self, other), op='*')
    
    def __rmul__(self, other: int) -> 'Pool':
        return self.__mul__(other)
    
    def __str__(self) -> str:
        return self._compute_seq()

    def __repr__(self) -> str:
        return f"Pool(seq={repr(self.seq)})"

    def __len__(self) -> int:
        return len(self._compute_seq())
    
    def __getitem__(self, key: Union[int, slice]) -> Union[str, 'Pool']:
        return Pool(None, parents=(self, key), op='slice')
        
    def visualize_graph(self, indent=0):
        """Print the computation graph structure."""
        states_info = f"({self.num_states} states)" if self.num_states > 0 else "(infinite)"
        unique_ancestors = f"[{len(self.ancestors)} unique]" if self.ancestors else ""
        print("  " * indent + f"{self.op or 'input'}: {str(self)} {states_info} {unique_ancestors}")
        for parent in self.parents:
            if isinstance(parent, Pool):
                parent.visualize_graph(indent + 1)
            else: 
                print("  " * (indent + 1) + f"input: {str(parent)}")


class RandomPool(Pool):
    """A class for generating random oligonucleotide sequences.
    
    Inherits from Pool and generates a new random sequence of specified length
    and alphabet each time the sequence is accessed. Has infinite states.
    """
    def __init__(self, length: int, alphabet: str = 'ACGT'):
        self.length = length
        self.alphabet = alphabet
        super().__init__(op='random')
    
    def _calculate_num_states(self) -> int:
        """RandomPool has infinite states."""
        return -1
    
    def _compute_seq(self) -> str:
        random.seed(self._state)
        seq = ''.join(random.choice(self.alphabet) for _ in range(self.length))
        return seq
    
    def __repr__(self) -> str:
        return f"RandomPool(L={self.length}, alphabet='{self.alphabet}')"


class ShuffledPool(Pool):
    """A class for generating shuffled versions of an input oligonucleotide sequence.
    
    Inherits from Pool and generates a new shuffled version of the input sequence
    each time the sequence is accessed. Has infinite states.
    """
    def __init__(self, seq: Union[Pool, str]):
        super().__init__(parents=(seq,), op='shuffle')
    
    def _calculate_num_states(self) -> int:
        """ShuffledPool has infinite states (different random shuffles)."""
        return -1
    
    def _compute_seq(self) -> str:
        if isinstance(self.parents[0], str):
            seq_list = list(self.parents[0])
        else:
            seq_list = list(self.parents[0]._compute_seq())
        random.seed(self._state)
        random.shuffle(seq_list)
        return ''.join(seq_list)   
    
    def __repr__(self) -> str:
        parent_seq = self.parents[0].seq if isinstance(self.parents[0], Pool) else self.parents[0]
        return f"ShuffledPool(seq={parent_seq})"


class SequentialPool(Pool):
    """A class for selecting sequences sequentially from a list of oligonucleotides.
    
    Inherits from Pool and selects sequences in order from the provided list.
    Has a FINITE number of states equal to the number of sequences in the list,
    enabling combinatorially complete iteration when composed with other pools.
    """
    def __init__(self, seqs: List[Union[Pool, str]]):
        self.seqs = seqs
        super().__init__(op='sequential_choice')
    
    def _calculate_num_states(self) -> int:
        """SequentialPool has finite states equal to the number of sequences."""
        return len(self.seqs)
    
    def _compute_seq(self) -> str:
        index = self._state % len(self.seqs)
        og = self.seqs[index]
        return og.seq if isinstance(og, Pool) else og
    
    def __repr__(self) -> str:
        seqs_preview = ', '.join(repr(s)[:15] for s in self.seqs[:3])
        if len(self.seqs) > 3:
            seqs_preview += '...'
        return f"SequentialPool({len(self.seqs)} seqs)"


class RandomSequentialPool(SequentialPool):
    """A class that iterates over all possible sequences of a given length from an alphabet.
    
    Inherits from SequentialPool and generates all possible sequences (alphabet^length total).
    This is a FINITE pool with num_states = len(alphabet)^length, enabling combinatorially 
    complete iteration.
    
    For example:
        - length=2, alphabet='AB' → ['AA', 'AB', 'BA', 'BB'] (4 sequences)
        - length=3, alphabet='ACGT' → 64 sequences (4^3)
    """
    def __init__(self, length: int, alphabet: str = 'ACGT'):
        """Initialize a RandomSequentialPool.
        
        Args:
            length: Length of sequences to generate
            alphabet: String of characters to use (default: 'ACGT')
        """
        self.length = length
        self.alphabet = alphabet
        
        # Generate all possible sequences using itertools.product
        all_seqs = [''.join(combo) for combo in itertools.product(alphabet, repeat=length)]
        
        # Initialize parent SequentialPool with all sequences
        super().__init__(seqs=all_seqs)
    
    def __repr__(self) -> str:
        return f"RandomSequentialPool(L={self.length}, alphabet='{self.alphabet}', {self.num_states} seqs)"


class RandomSelectionPool(Pool):
    """A class for randomly selecting sequences from a list of oligonucleotides.
    
    Inherits from Pool and randomly selects a sequence from the provided list
    each time the sequence is accessed. Has infinite states (random selection).
    """
    def __init__(self, seqs: List[Union[Pool, str]]):
        self.seqs = seqs
        super().__init__(op='random_choice')
    
    def _calculate_num_states(self) -> int:
        """RandomSelectionPool has infinite states (random selection)."""
        return -1
    
    def _compute_seq(self) -> str:
        random.seed(self._state)
        og = random.choice(self.seqs)
        return og.seq if isinstance(og, Pool) else og
    
    def __repr__(self) -> str:
        return f"RandomSelectionPool({len(self.seqs)} seqs)"

    
def shuffle(seq: Union[Pool, str]) -> Pool:
    """Create a ShuffledPool from an input pool or string.
    
    Args:
        seq: Input oligonucleotide sequence, either as Pool object or string
        
    Returns:
        ShuffledPool object that generates shuffled versions of the input sequence
    """
    return ShuffledPool(seq)


def generate_seqs(oligo_generator: Pool, size: int = None) -> List[str]:
    """Generate a list of sequences from a Pool.
    
    Args:
        oligo_generator: The Pool object to generate from
        size: Number of sequences to generate. If None and pool is finite,
              generates all possible sequences.
    
    Returns:
        List of generated sequences
    """
    if size is None:
        if oligo_generator.is_finite():
            size = oligo_generator.num_states
        else:
            raise ValueError("Must specify size for infinite pools")
    
    oligo_generator.set_state(0)  # Reset to beginning
    return [next(oligo_generator) for _ in range(size)]


def generate_complete_seqs(oligo_generator: Pool) -> List[str]:
    """Generate all possible sequences from a finite pool.
    
    Args:
        oligo_generator: The Pool object (must be finite)
    
    Returns:
        List of all possible sequences
        
    Raises:
        ValueError: If the pool is infinite
    """
    if not oligo_generator.is_finite():
        raise ValueError(f"Cannot generate complete sequences from infinite pool. "
                        f"Pool has num_states={oligo_generator.num_states}")
    
    result = []
    for state in range(oligo_generator.num_states):
        oligo_generator.set_state(state)
        result.append(oligo_generator.seq)
    return result
    
if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Combinatorially complete iteration with SequentialPools")
    print("=" * 70)
    
    # Two sequential pools - should give all combinations
    pool1 = SequentialPool(['A', 'B', 'C'])
    pool2 = SequentialPool(['1', '2'])
    combined = pool1 + '-' + pool2
    
    print(f"\nCombined pool has {combined.num_states} states")
    print(f"Unique ancestors: {len(combined.ancestors)}")
    combined.visualize_graph()
    
    print("\nAll sequences (combinatorially complete):")
    all_seqs = generate_complete_seqs(combined)
    for i, seq in enumerate(all_seqs):
        print(f"  State {i}: {seq}")
    
    print("\n" + "=" * 70)
    print("Example 2: Reusing the same Pool multiple times in the graph")
    print("=" * 70)
    
    # Same pool appears twice - should be counted only once
    a = SequentialPool(['X', 'Y'])
    repeated = a + '-' + a + '-' + a
    
    print(f"\nRepeated pool has {repeated.num_states} states")
    print(f"Unique ancestors: {len(repeated.ancestors)} (should be 1)")
    repeated.visualize_graph()
    
    print("\nAll sequences (same value repeated 3 times):")
    all_seqs = generate_complete_seqs(repeated)
    for i, seq in enumerate(all_seqs):
        print(f"  State {i}: {seq}")
    
    print("\n" + "=" * 70)
    print("Example 3: Complex composition with multiple SequentialPools")
    print("=" * 70)
    
    prefix = SequentialPool(['AAA', 'TTT'])
    middle = SequentialPool(['GGG', 'CCC'])
    suffix = SequentialPool(['XX', 'YY', 'ZZ'])
    complex_pool = prefix + '.' + middle + '.' + suffix
    
    print(f"\nComplex pool has {complex_pool.num_states} states (2×2×3=12)")
    print(f"Unique ancestors: {len(complex_pool.ancestors)}")
    print("\nAll sequences:")
    all_seqs = generate_complete_seqs(complex_pool)
    for i, seq in enumerate(all_seqs):
        print(f"  State {i:2d}: {seq}")
    
    print("\n" + "=" * 70)
    print("Example 4: RandomSequentialPool - all possible sequences")
    print("=" * 70)
    
    # Generate all possible 2-mers from alphabet 'AB'
    dimers = RandomSequentialPool(length=2, alphabet='AB')
    
    print(f"\nDimers pool has {dimers.num_states} states (2^2=4)")
    print("All sequences:")
    all_dimers = generate_complete_seqs(dimers)
    for i, seq in enumerate(all_dimers):
        print(f"  State {i}: {seq}")
    
    # Generate all possible 3-mers from alphabet 'XYZ'
    print("\n")
    trimers = RandomSequentialPool(length=3, alphabet='XYZ')
    print(f"Trimers pool has {trimers.num_states} states (3^3=27)")
    print("First 10 sequences:")
    trimers.set_state(0)
    for i in range(10):
        print(f"  State {i}: {next(trimers)}")
    
    print("\n" + "=" * 70)
    print("Example 5: Combining RandomSequentialPools")
    print("=" * 70)
    
    # Combine two RandomSequentialPools
    pool_a = RandomSequentialPool(length=2, alphabet='AB')
    pool_b = RandomSequentialPool(length=1, alphabet='XY')
    combined = pool_a + '-' + pool_b
    
    print(f"\nCombined pool has {combined.num_states} states (4×2=8)")
    print(f"Unique ancestors: {len(combined.ancestors)}")
    print("All sequences:")
    all_seqs = generate_complete_seqs(combined)
    for i, seq in enumerate(all_seqs):
        print(f"  State {i}: {seq}")
    
    print("\n" + "=" * 70)
    print("Example 6: Mixed finite and infinite pools")
    print("=" * 70)
    
    # This will be infinite because of RandomPool
    finite_part = SequentialPool(['A', 'B'])
    infinite_part = RandomPool(4, 'ACGT')
    mixed = finite_part + '-' + infinite_part
    
    print(f"\nMixed pool has num_states={mixed.num_states} (infinite)")
    print(f"Unique ancestors: {len(mixed.ancestors)}")
    print("First 10 sequences:")
    pool_samples = generate_seqs(mixed, size=10)
    for i, seq in enumerate(pool_samples):
        print(f"  Sample {i}: {seq}")