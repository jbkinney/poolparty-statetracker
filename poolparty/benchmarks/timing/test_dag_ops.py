"""DAG operation workloads for poolparty benchmarking."""
from ._utils import DEFAULT_NUM_SEQS, DEFAULT_SEQ_LEN

# Pre-warm imports before profiling
import poolparty as pp
pp.init()


def workload_chain_of_joins(
    num_joins: int = 5,
    seg_len: int = 5,
    num_seqs: int = DEFAULT_NUM_SEQS,
    use_styles: bool = False,
    use_cards: bool = False,
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    segments = ['A'*seg_len,'C'*seg_len,'G'*seg_len,'T'*seg_len]
    pool = pp.from_seqs(segments, mode='random')
    for i in range(num_joins):
        new_pool = pp.from_seqs(segments, mode='random')
        pool = pp.join([pool, new_pool])
    return pool.generate_library(num_seqs=num_seqs)

workload_chain_of_joins.benchmark_specs = [
    ("TestDAGSize", "num_joins", [1, 2, 4, 8, 16]),
]


def workload_tree_of_joins(
    num_levels: int = 3,
    seg_len: int = 5,
    num_seqs: int = DEFAULT_NUM_SEQS,
    use_styles: bool = False,
    use_cards: bool = False,
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    segments = ['A'*seg_len,'C'*seg_len,'G'*seg_len,'T'*seg_len]
    pool = pp.from_seqs(segments, mode='random')
    for i in range(num_levels):
        pool = pp.join([pool, pool.deepcopy()])
    return pool.generate_library(num_seqs=num_seqs)

workload_tree_of_joins.benchmark_specs = [
    ("TestDAGSize", "num_levels", [1, 2, 3, 4, 5]),
]


# Generate test classes for running this file directly with pytest
from ._utils import collect_local_specs, generate_benchmark_tests

globals().update(generate_benchmark_tests(collect_local_specs(globals())))
