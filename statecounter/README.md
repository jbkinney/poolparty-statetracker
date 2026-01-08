# StateCounter

[![PyPI version](https://badge.fury.io/py/statecounter.svg)](https://badge.fury.io/py/statecounter)
[![Documentation Status](https://readthedocs.org/projects/statecounter/badge/?version=latest)](https://statecounter.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StateCounter** is a Python library for creating composable counters with unidirectional state propagation. It provides a powerful way to enumerate combinatorial spaces through counter algebra operations.

## Features

- **Composable Counters**: Build complex iteration patterns from simple counters
- **Unidirectional State Propagation**: Set child state and parent states update automatically
- **Counter Algebra**: Product (×), sum (+), slice, repeat, shuffle, and synchronize operations
- **Conflict Detection**: Automatic detection of conflicting state assignments
- **Tree Visualization**: Built-in ASCII tree visualization for debugging

## Installation

```bash
pip install statecounter
```

## Quick Start

```python
from statecounter import Counter, Manager

with Manager():
    # Create leaf counters
    A = Counter(num_states=2, name='A')
    B = Counter(num_states=3, name='B')
    
    # Combine with product (Cartesian product)
    C = A * B  # 6 states total
    
    # Iterate and see parent states update
    for state in C:
        print(f"C={state}, A={A.state}, B={B.state}")
```

Output:
```
C=0, A=0, B=0
C=1, A=1, B=0
C=2, A=0, B=1
C=3, A=1, B=1
C=4, A=0, B=2
C=5, A=1, B=2
```

## Counter Operations

### Product (Cartesian Product)

```python
with Manager():
    A = Counter(num_states=2, name='A')
    B = Counter(num_states=3, name='B')
    C = A * B  # 6 states (2 × 3)
```

### Stack 

```python
with Manager():
    A = Counter(num_states=2, name='A')
    B = Counter(num_states=3, name='B')
    C = stack([A,B])  # 5 states (2 + 3)
    
    for _ in C:
        # Only one of A or B is active (non-None) at a time
        print(f"A={A.state}, B={B.state}")
```

### Slicing

```python
with Manager():
    A = Counter(num_states=10, name='A')
    B = A[2:5]   # States 2, 3, 4
    C = A[::2]   # Even states: 0, 2, 4, 6, 8
    D = A[::-1]  # Reversed: 9, 8, 7, ..., 0
```

### Repeat

```python
with Manager():
    A = Counter(num_states=3, name='A')
    B = A * 4  # Repeat A four times (12 states)
```

### Shuffle

```python
from statecounter import Counter, Manager, shuffle_counter

with Manager():
    A = Counter(num_states=5, name='A')
    B = shuffle_counter(A, seed=42)  # Randomly permuted order
```

### Synchronize

```python
from statecounter import Counter, Manager, synchronize_counters

with Manager():
    A = Counter(num_states=4, name='A')
    B = Counter(num_states=4, name='B')
    C = synchronize_counters(A, B)  # A and B always have same state
```

## State Propagation

StateCounter uses **unidirectional state propagation**. When you set a child counter's state, all parent counters automatically update:

```python
with Manager():
    A = Counter(num_states=2, name='A')
    B = Counter(num_states=3, name='B')
    C = A * B
    
    C.state = 5  # Set child state
    print(A.state)  # 1 (automatically updated)
    print(B.state)  # 2 (automatically updated)
```

## Visualization

Visualize counter dependencies with ASCII trees:

```python
with Manager():
    A = Counter(num_states=2, name='A')
    B = Counter(num_states=3, name='B')
    C = A * B
    C.name = 'C'
    
    C.print_dag()
```

Output:
```
C [Multiply, n=6]
├── A [Leaf, n=2]
└── B [Leaf, n=3]
```

## Documentation

Full documentation is available at [statecounter.readthedocs.io](https://statecounter.readthedocs.io).

## License

MIT License - see [LICENSE](LICENSE) for details.
