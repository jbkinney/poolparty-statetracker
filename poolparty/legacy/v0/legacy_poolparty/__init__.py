"""poolparty - A Python package for generating oligonucleotide sequence pools.

This package provides classes and utilities for creating and manipulating
oligonucleotide sequences with lazy evaluation and combinatorial iteration.
"""

# Global constant for maximum number of states before treating pool as infinite
DEFAULT_MAX_NUM_STATES = 1_000_000

from .pool import Pool, visualize_computation_graph, MixedPoolSegmentIndex, validate_unique_names
from .design_cards import DesignCards
from .orf_pool import ORFPool
from .kmer_pool import KmerPool
from .shuffle_pool import ShufflePool
from .random_mutation_pool import RandomMutationPool
from .k_mutation_pool import KMutationPool
from .k_mutation_orf_pool import KMutationORFPool
from .random_mutation_orf_pool import RandomMutationORFPool
from .motif_pool import MotifPool
from .iupac_pool import IUPACPool
from .subseq_pool import SubseqPool
from .insertion_scan_pool import InsertionScanPool
from .insertion_scan_orf_pool import InsertionScanORFPool
from .deletion_scan_pool import DeletionScanPool
from .deletion_scan_orf_pool import DeletionScanORFPool
from .shuffle_scan_pool import ShuffleScanPool
from .spacing_scan_pool import SpacingScanPool
from .mixed_pool import MixedPool
from .barcode_pool import BarcodePool
from .utils import shuffle, validate_alphabet, named_alphabets_dict, get_alphabet
from .visualization import visualize

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MAX_NUM_STATES",
    "Pool",
    "ORFPool",
    "DesignCards",
    "MixedPoolSegmentIndex",
    "validate_unique_names",
    "visualize_computation_graph",
    "visualize",
    "KmerPool",
    "ShufflePool",
    "RandomMutationPool",
    "KMutationPool",
    "KMutationORFPool",
    "RandomMutationORFPool",
    "MotifPool",
    "IUPACPool",
    "SubseqPool",
    "InsertionScanPool",
    "InsertionScanORFPool",
    "DeletionScanPool",
    "DeletionScanORFPool",
    "ShuffleScanPool",
    "SpacingScanPool",
    "MixedPool",
    "BarcodePool",
    "shuffle",
    "validate_alphabet",
    "named_alphabets_dict",
]

