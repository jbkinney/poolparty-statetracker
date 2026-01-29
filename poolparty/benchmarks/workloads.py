"""Benchmark workloads for poolparty profiling."""
import poolparty as pp
from typing import Literal

def make_sequence(length: int) -> str:
    """Generate a DNA sequence of specified length."""
    bases = 'ACGT'
    return (bases * (length // 4 + 1))[:length]

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

def workload_mpra_example(
    num_seqs: int = 1000,
    use_styles: bool = False,
    use_cards: bool = False
):
    pp.init()
    #pp.load_config('default_config.toml')

    template_pool = pp.from_seq('TCCCGACT<cre>GGAAAGCGGGCAGTGAGCACACAGGA</cre>ATTACGG<bc/>AGATCGGA')\
                    .named('template_pool').stylize(style='lower', which='contents')

    mutated_pool = template_pool.stylize(region='cre', style='goldenrod')\
                                .mutagenize(region='cre',
                                            num_mutations=2, 
                                            style='yellow bold underline lower',
                                            mode='sequential', 
                                            num_states=5, 
                                            prefix='mutagenize').named('mutated_pool')\
                                .repeat_states(2, prefix='v', iter_order=-2)

    L = len('GGAAAGCGGGCAGTGAGCACACAGGA') 
    recomb_pool = template_pool.recombine(region='cre', 
                                        num_breakpoints=3,
                                        sources=['A'*L, 'C'*L, 'G'*L, 'T'*L],
                                        styles=['palegreen', 'springgreen', 'limegreen', 'forestgreen'],
                                        style_by='order',
                                        mode='random',
                                        num_states=5,
                                        prefix='recomb').named('recomb_pool')\
                                .repeat_states(2, prefix='v', iter_order=-2)


    deletion_pool = template_pool.stylize(region='cre', style='salmon')\
                                .deletion_scan(region='cre', 
                                                deletion_length=6, 
                                                positions=slice(None, None, 5), 
                                                mode='sequential', 
                                                style='red bold',
                                                prefix='delscan').named('deletion_pool')\
                                .repeat_states(2, prefix='v', iter_order=-2)

    sites_pool=pp.from_seqs(['AAAAAA','TTTTTT'], 
                            mode='sequential', 
                            iter_order=-1).named('sites_pool')

    insertion_pool = template_pool.stylize(region='cre', style='blue')\
                                .insertion_scan(region='cre', 
                                                ins_pool=sites_pool, 
                                                positions=slice(None,None,5), 
                                                replace=True, 
                                                mode='sequential',
                                                prefix='insscan',
                                                prefix_position='pos', 
                                                prefix_insert='ins',
                                                style='cyan bold').named('insertion_pool')
                                
    shuffle_pool = template_pool.stylize(region='cre', style='purple')\
                                .shuffle_scan(region='cre', 
                                            shuffle_length=6, 
                                            shuffles_per_position=2,
                                            positions=slice(None, None, 5), 
                                            mode='sequential', 
                                            style='magenta bold',
                                            prefix='shufscan',
                                            prefix_position='pos',
                                            prefix_shuffle='shuf')
                                

    combo_pool = pp.stack([mutated_pool, recomb_pool, deletion_pool, insertion_pool, shuffle_pool])\
        .named('stack_pool')\
        .insert_kmers(region='bc', mode='random', length=5, prefix='bc', style='green bold')\
        .named('combo_pool')\
        .stylize(which='tags', style='gray')

    pp.toggle_styles(on=use_styles)
    pp.toggle_cards(on=use_cards)
    return combo_pool.generate_library(num_seqs=num_seqs)


# Collect all workloads for easy iteration
ALL_WORKLOADS = {
    'mutagenize': workload_mutagenize,
    'shuffle_seq': workload_shuffle_seq,
    'deletion_scan': workload_deletion_scan,
    'insertion_scan': workload_insertion_scan,
    'get_kmers': workload_get_kmers,
    'from_iupac': workload_from_iupac,
    'recombine': workload_recombine,
    'mpra_example': workload_mpra_example,
}
