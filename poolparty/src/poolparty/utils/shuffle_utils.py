"""Utilities for sequence shuffling algorithms."""

from collections import defaultdict

import numpy as np


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Shuffle a sequence preserving dinucleotide frequencies.

    Uses the Altschul-Erikson algorithm: builds a directed multigraph where
    each edge represents a dinucleotide, finds a random spanning arborescence
    to guarantee the Euler path can be completed, then walks the path greedily.

    The first and last characters of the output are always identical to the
    input — this is a mathematical constraint of the Euler path.

    Parameters
    ----------
    seq : str
        Input sequence (any alphabet).
    rng : np.random.Generator
        Numpy random generator for reproducibility.

    Returns
    -------
    str
        Shuffled sequence with identical dinucleotide composition.
    """
    n = len(seq)
    if n <= 2:
        return seq

    # Build adjacency lists (multigraph edges) from consecutive character pairs
    edges: dict[str, list[str]] = defaultdict(list)
    for i in range(n - 1):
        edges[seq[i]].append(seq[i + 1])

    # Build a random spanning arborescence rooted at seq[-1] using
    # loop-erased random walk (Wilson's algorithm on the simple graph).
    # This designates one "last edge" per non-root node to ensure the
    # greedy Euler walk can always complete.
    chars = sorted(set(seq))
    root = seq[-1]
    unique_succs = {c: list(set(edges[c])) for c in chars}

    last_edge_target: dict[str, str] = {}
    in_tree = {root}

    for start_char in chars:
        if start_char in in_tree:
            continue
        # Loop-erased random walk until hitting a tree node
        path = [start_char]
        path_idx = {start_char: 0}
        current = start_char
        while current not in in_tree:
            candidates = unique_succs[current]
            succ = candidates[int(rng.integers(len(candidates)))]
            if succ in path_idx:
                loop_start = path_idx[succ]
                for removed in path[loop_start + 1 :]:
                    del path_idx[removed]
                path = path[: loop_start + 1]
                current = succ
            else:
                path.append(succ)
                path_idx[succ] = len(path) - 1
                current = succ
        # Add path nodes to tree with their arborescence edges
        for i in range(len(path) - 1):
            in_tree.add(path[i])
            last_edge_target[path[i]] = path[i + 1]

    # Arrange edges: put the arborescence ("last") edge at index 0 so it
    # is consumed last when we pop() from the end during the walk.
    for char, succs in edges.items():
        if char in last_edge_target:
            target = last_edge_target[char]
            idx = succs.index(target)
            succs[0], succs[idx] = succs[idx], succs[0]
            sub = succs[1:]
            rng.shuffle(sub)
            succs[1:] = sub
        else:
            rng.shuffle(succs)

    # Greedy Euler path walk
    result = [seq[0]]
    current = seq[0]
    for _ in range(n - 1):
        nxt = edges[current].pop()
        result.append(nxt)
        current = nxt

    return "".join(result)
