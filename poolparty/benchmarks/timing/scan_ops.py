"""Scan operation workloads for poolparty benchmarking."""
import poolparty as pp
from typing import Literal
from ._utils import make_sequence


def workload_deletion_scan(
    seq_len: int = 100,
    num_seqs: int = 100,
    del_len: int=5,
    positions = None,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    seq = make_sequence(seq_len)
    pool = pp.deletion_scan(seq, deletion_length=del_len, positions=positions, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_deletion_scan.benchmark_specs = [
    ("TestDeletionScan", "seq_len", [10, 30, 100, 300, 1000]),
]


def workload_insertion_scan(
    seq_len: int = 100,
    num_seqs: int = 100,
    num_ins: int = 10,
    ins_len: int=5,
    positions = None,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    seq = make_sequence(seq_len)
    ins_seqs = ['A'*ins_len]*num_ins
    ins_pool = pp.from_seqs(ins_seqs)
    pool = pp.insertion_scan(seq, ins_pool = ins_pool, positions=positions, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_insertion_scan.benchmark_specs = [
    ("TestInsertionScan", "seq_len", [10, 30, 100, 300, 1000]),
    ("TestInsertionScan", "ins_len", [1, 3, 10, 30, 100]),
]


# Generate test classes for running this file directly with pytest
from ._utils import collect_local_specs
from ..benchmark_utils import generate_benchmark_tests

globals().update(generate_benchmark_tests(collect_local_specs(globals())))