"""Base operation workloads for poolparty benchmarking."""
import poolparty as pp
from typing import Literal
from ._utils import make_sequence

# uv run python poolparty/benchmarks/run_benchmarks.py benchmark.py -c TestMutagenize -t    

def workload_mutagenize(
    seq_len: int = 100,
    mut_rate: float = None,
    num_mut: int = None,
    num_seqs: int = 100,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    seq = make_sequence(seq_len)
    pool = pp.mutagenize(seq, mutation_rate=mut_rate, num_mutations=num_mut, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_mutagenize.benchmark_specs = [
    ("TestMutagenize", "num_mut", [1, 3, 10, 30, 100]),
    ("TestMutagenize", "mut_rate", [0.01, 0.03, 0.10, 0.30, 1.00]),
    ("TestMutagenize", "seq_len", [10, 30, 100, 300, 1_000, 3_000], {"mut_rate": 0.1}),
]


def workload_shuffle_seq(
    seq_len: int = 100,
    num_seqs: int = 100,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    seq = make_sequence(seq_len)
    pool = pp.shuffle_seq(seq, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_shuffle_seq.benchmark_specs = [
    ("TestShuffleSeq", "seq_len", [10, 30, 100, 300, 1000]),
]


def workload_get_kmers(
    kmer_len: int = 5,
    num_seqs: int = 100,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    pool = pp.get_kmers(length=kmer_len, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_get_kmers.benchmark_specs = [
    ("TestGetKmers", "kmer_len", [1, 3, 10, 30, 100]),
]


def workload_from_iupac(
    seq_len: int = 5,
    num_seqs: int = 100,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    seq = 'N'*seq_len
    pool = pp.from_iupac(iupac_seq=seq, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_from_iupac.benchmark_specs = [
    ("TestFromIupac", "seq_len", [1, 3, 10, 30, 100]),
]


def workload_recombine(
    seq_len: int = 100,
    num_breakpoints: int = 3,
    num_sources: int = 4,
    num_seqs: int = 100,
    mode: Literal['random', 'sequential'] = 'random',
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    # Create source sequences of specified length
    sources = [make_sequence(seq_len) for _ in range(num_sources)]
    pool = pp.recombine(sources=sources, num_breakpoints=num_breakpoints, mode=mode)
    return pool.generate_library(num_seqs=num_seqs)

workload_recombine.benchmark_specs = [
    ("TestRecombine", "seq_len", [10, 30, 100, 300, 1000]),
    ("TestRecombine", "num_breakpoints", [1, 3, 10, 30, 99]),
    ("TestRecombine", "num_sources", [2, 4, 10, 30, 100]),
]


# Generate test classes for running this file directly with pytest
from ._utils import collect_local_specs
from ..benchmark_utils import generate_benchmark_tests

globals().update(generate_benchmark_tests(collect_local_specs(globals())))
