"""Example workloads for poolparty benchmarking."""
import poolparty as pp


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

workload_mpra_example.benchmark_specs = [
    ("TestMPRAExample", "num_seqs", [10, 100, 1_000]),
]


# Generate test classes for running this file directly with pytest
from ._utils import collect_local_specs
from ..benchmark_utils import generate_benchmark_tests

globals().update(generate_benchmark_tests(collect_local_specs(globals())))
