from typing import Tuple, List, Union
import random    

class OligoGenerator:
    """A class representing an oligonucleotide sequence with lazy evaluation.
    
    Supports sequence concatenation (+) and repetition (*) operations through
    a computation graph structure.
    """
    def __init__(self, 
                 seq: str | None = None, 
                 parents: Tuple[Union['OligoGenerator', str, int],...] = (), 
                 op: str | None = None):
        self._seq = seq
        self.parents = parents
        self.op = op
        self.ancestors = set()
        for parent in self.parents:
            if isinstance(parent, OligoGenerator):
                self.ancestors.update(parent.ancestors)
        self._state = 0
        self.num_states = 1
        for parent in self.parents:
            if isinstance(parent, OligoGenerator):
                self.num_states *= parent.num_states
        
    @property
    def seq(self) -> str:
        if self._seq is None:
            return self._compute_seq()
        else:
            return self._seq
        
    def set_state(self, state: int) -> None:
        self._state = state
        # FIGURE OUT HOW TO ITERATE OVER THE STATES

    def _advance_state(self) -> None:
        self._state += 1

    def _compute_seq(self) -> str:
        match self.op, self.parents:
            case None, ():
                return self.seq
            case '+', (OligoGenerator(), str()):
                return self.parents[0].seq + self.parents[1]
            case '+', (OligoGenerator(), OligoGenerator()):
                return self.parents[0].seq + self.parents[1].seq
            case '*', (OligoGenerator(), int()):
                return self.parents[0].seq * self.parents[1]
            case 'slice', (OligoGenerator(), slice()):
                return self.parents[0].seq[self.parents[1]]
            case _:
                raise ValueError(f"Invalid op: {self.op}")
    
    def __next__(self) -> str:
        seq = self.seq
        self._advance_state() 
        for ancestor in self.ancestors:
            ancestor._advance_state()   
        return seq

    def __add__(self, other: 'OligoGenerator') -> 'OligoGenerator':
        return OligoGenerator(parents=(self, other), op='+')
    
    def __radd__(self, other: 'OligoGenerator') -> 'OligoGenerator':
        return self.__add__(other)
    
    def __mul__(self, other: int) -> 'OligoGenerator':
        return OligoGenerator(parents=(self, other), op='*')
    
    def __rmul__(self, other: int) -> 'OligoGenerator':
        return self.__mul__(other)
    
    def __str__(self) -> str:
        return self._compute_seq()

    def __repr__(self) -> str:
        return f"Oligo(seq={repr(self.seq)})"

    def __len__(self) -> int:
        return len(self._compute_seq())
    
    def __getitem__(self, key: Union[int, slice]) -> Union[str, 'OligoGenerator']:
        return OligoGenerator(None, parents=(self, key), op='slice')
        
    def visualize_graph(self, indent=0):
        """Print the computation graph structure."""
        print("  " * indent + f"{self.op or 'input'}: {str(self)}")
        for parent in self.parents:
            if isinstance(parent, OligoGenerator):
                parent.visualize_graph(indent + 1)
            else: 
                print("  " * (indent + 1) + f"input: {str(parent)}")


class RandomOG(OligoGenerator):
    """A class for generating random oligonucleotide sequences.
    
    Inherits from Oligo and generates a new random sequence of specified length
    and alphabet each time the sequence is accessed.
    """
    def __init__(self, length: int, alphabet: str = 'ACGT'):
        self.length = length
        self.alphabet = alphabet
        super().__init__(op='random')
    
    def _compute_seq(self) -> None:
        random.seed(self._state)
        seq = ''.join(random.choice(self.alphabet) for _ in range(self.length))
        return seq
    
    def __repr__(self) -> str:
        return f"RandomOG(L={self.length}, alphabet='{self.alphabet}')"


class ShuffledOG(OligoGenerator):
    """A class for generating shuffled versions of an input oligonucleotide sequence.
    +
    Inherits from Oligo and generates a new shuffled version of the input sequence
    each time the sequence is accessed.
    """
    def __init__(self, seq: Union[OligoGenerator, str]):
        super().__init__(parents=(seq,), op='shuffle')
    
    def _compute_seq(self) -> str:
        if isinstance(self.parents[0], str):
            seq_list = list(self.parents[0])
        else:
            seq_list = list(self.parents[0]._compute_seq())
        random.seed(self._state)
        random.shuffle(seq_list)
        return ''.join(seq_list)   
    
    def __repr__(self) -> str:
        parent_seq = self.parents[0].seq if isinstance(self.parents[0], OligoGenerator) else self.parents[0]
        return f"ShuffledOG(seq={parent_seq})"


class ChosenOG(OligoGenerator):
    """A class for selecting sequences from a list of oligonucleotides.
    
    Inherits from Oligo and can either select sequences sequentially in order
    or randomly from the provided list each time the sequence is accessed.
    """
    def __init__(self, seqs: List[Union[OligoGenerator, str]], mode: str = 'sequential'):
        self.seqs = seqs
        if mode not in ['sequential', 'random']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode    
        super().__init__(op=f'{mode}_choice')
    
    def _compute_seq(self) -> str:
        if self.mode == 'sequential':
            self.index = self._state % len(self.seqs)
            og = self.seqs[self.index]
        elif self.mode == 'random':
            random.seed(self._state)
            og = random.choice(self.seqs)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return og.seq if isinstance(og, OligoGenerator) else og
    
    def __repr__(self) -> str:
        seqs_str = ', '.join(repr(s) for s in self.seqs)
        return f"ChosenOG(seqs=[...], mode={self.mode})"

    
def shuffle(seq: Union[OligoGenerator, str]) -> OligoGenerator:
    """Create a ShuffledOligo from an input oligo or string.
    
    Args:
        seq: Input oligonucleotide sequence, either as Oligo object or string
        
    Returns:
        ShuffledOligo object that generates shuffled versions of the input sequence
    """
    return ShuffledOG(seq)

def generate_pool(oligo_generator: OligoGenerator, size: int) -> List[str]:
    return [next(oligo_generator) for _ in range(size)]
    
if __name__ == "__main__":
    a = RandomOG(length=4, alphabet='ABCD')
    b = ShuffledOG('EFGH')
    c = ChosenOG(['IJKL', 'MNOP', 'QRST', 'UVWX'], mode='sequential')
    d = ChosenOG(['01', '23', '45', '67', '89'], mode='random')
    e = 2*a + '.' + b + '.' + c + '.' + d + '.' + a[::-1]
    e.visualize_graph()
    pool = generate_pool(e, 5)
    for seq in pool:
        print(repr(seq))