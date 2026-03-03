# StateTracker

[![PyPI version](https://badge.fury.io/py/statetracker.svg)](https://badge.fury.io/py/statetracker)
[![Documentation Status](https://readthedocs.org/projects/statetracker/badge/?version=latest)](https://statetracker.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StateTracker** is a Python library for creating composable states with unidirectional value propagation. It provides a powerful way to enumerate combinatorial spaces through state algebra operations.

## Why StateTracker?

StateTracker was developed to support the design of complex DNA sequence libraries (see [PoolParty](https://github.com/jbkinney/poolparty-statetracker/tree/main/poolparty)), but it solves a general problem: **random access to combinatorial spaces**.

If you've ever written nested loops to enumerate a Cartesian product and then wished you could shuffle the order, sample a subset, or split into train/test sets—all while tracking which component indices correspond to each item—StateTracker is for you. Build your combinatorial structure once using state algebra, and StateTracker handles the index math automatically.

## Features

- **Composable States**: Build complex iteration patterns from simple states
- **Unidirectional Value Propagation**: Set child value and parent values update automatically
- **State Algebra**: Product (×), sum (+), slice, repeat, shuffle, and synchronize operations
- **Conflict Detection**: Automatic detection of conflicting value assignments
- **Tree Visualization**: Built-in ASCII tree visualization for debugging

## Installation

```bash
pip install statetracker
```

## Quick Start

```python
from statetracker import State, Manager

with Manager():
    # Create leaf states
    A = State(num_values=2, name='A')
    B = State(num_values=3, name='B')

    # Combine with product (Cartesian product)
    C = A * B  # 6 values total

    # Iterate and see parent values update
    for value in C:
        print(f"C={value}, A={A.value}, B={B.value}")
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

## State Operations

### Product (Cartesian Product)

```python
with Manager():
    A = State(num_values=2, name='A')
    B = State(num_values=3, name='B')
    C = A * B  # 6 values (2 × 3)
```

### Stack

```python
with Manager():
    A = State(num_values=2, name='A')
    B = State(num_values=3, name='B')
    C = stack([A,B])  # 5 values (2 + 3)

    for _ in C:
        # Only one of A or B is active (non-None) at a time
        print(f"A={A.value}, B={B.value}")
```

### Slicing

```python
with Manager():
    A = State(num_values=10, name='A')
    B = A[2:5]   # Values 2, 3, 4
    C = A[::2]   # Even values: 0, 2, 4, 6, 8
    D = A[::-1]  # Reversed: 9, 8, 7, ..., 0
```

### Repeat

```python
with Manager():
    A = State(num_values=3, name='A')
    B = A * 4  # Repeat A four times (12 values)
```

### Shuffle

```python
from statetracker import State, Manager, shuffle

with Manager():
    A = State(num_values=5, name='A')
    B = shuffle(A, seed=42)  # Randomly permuted order
```

### Synchronize

```python
from statetracker import State, Manager, sync

with Manager():
    A = State(num_values=4, name='A')
    B = State(num_values=4, name='B')
    C = sync([A, B])  # A and B always have same value
```

## Value Propagation

StateTracker uses **unidirectional value propagation**. When you set a child state's value, all parent states automatically update:

```python
with Manager():
    A = State(num_values=2, name='A')
    B = State(num_values=3, name='B')
    C = A * B

    C.value = 5  # Set child value
    print(A.value)  # 1 (automatically updated)
    print(B.value)  # 2 (automatically updated)
```

## Visualization

Visualize state dependencies with ASCII trees:

```python
with Manager():
    A = State(num_values=2, name='A')
    B = State(num_values=3, name='B')
    C = A * B
    C.name = 'C'

    C.print_dag()
```

Output:
```
C [Product, n=6]
├── A [State, n=2]
└── B [State, n=3]
```

## Documentation

Full documentation is available at [statetracker.readthedocs.io](https://statetracker.readthedocs.io).

## License

MIT License - see [LICENSE](LICENSE) for details.
